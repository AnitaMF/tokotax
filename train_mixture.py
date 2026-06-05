"""
Training script for mixture -> taxonomic abundance prediction.

Model: same two-branch GenomeClassifier architecture as baseline,
but output is a soft abundance vector over taxa (not a single class).

Label construction:
  For each mixture, sum the abundances of genomes belonging to the same taxon.
  Result: a vector of length num_taxa where values sum to 1.

Loss: KL divergence between predicted and true abundance distributions.
  - Predicted: softmax over output logits (ensures sum to 1)
  - Target: true abundance vector (already sums to 1)

Run as an LSF job (see submit_mixture.sh).
"""

import os
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# Model definitions (same architecture as baseline)
# ----------------------------------------------------------------------
class AdapterPool(nn.Module):
    """LoRA adapter on frozen embeddings + frequency-weighted pooling."""
    def __init__(self, frozen_embeddings, rank=8):
        super().__init__()
        self.register_buffer('E', frozen_embeddings)
        embed_dim = frozen_embeddings.shape[1]
        self.A = nn.Linear(embed_dim, rank, bias=False)
        self.B = nn.Linear(rank, embed_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, freqs):
        adapted = self.E + self.B(self.A(self.E))
        return adapted.mean(dim=0, keepdim=True).expand(freqs.shape[0], -1)    

## Weighted mean 
    # def forward(self, freqs):
    #     adapted = self.E + self.B(self.A(self.E))
    #     return freqs @ adapted


class FreqEncoder(nn.Module):
    """Encode raw frequency vector into a dense embedding."""
    def __init__(self, num_tokens, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_tokens, out_dim),
            nn.ReLU(),
        )

    def forward(self, freqs):
        return self.net(freqs)


class GenomeClassifier(nn.Module):
    """Two-branch model — now outputs abundance logits over taxa."""
    def __init__(self, frozen_embeddings, num_tokens, num_classes,
                 adapter_rank=8, freq_emb_dim=1024,
                 classifier_hidden=512, dropout=0.2):
        super().__init__()
        embed_dim = frozen_embeddings.shape[1]
        self.adapter_pool = AdapterPool(frozen_embeddings, rank=adapter_rank)
        self.freq_encoder = FreqEncoder(num_tokens, out_dim=freq_emb_dim)
        combined_dim = embed_dim + freq_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, freqs):
        adapter_emb = self.adapter_pool(freqs)
        freq_emb = self.freq_encoder(freqs)
        combined = torch.cat([adapter_emb, freq_emb], dim=1)
        return self.classifier(combined)   # raw logits — softmax applied in loss


# ----------------------------------------------------------------------
# Label builder
# ----------------------------------------------------------------------
def build_abundance_labels(mixture_meta, labels_df, level, taxon_to_idx):
    """
    For each mixture, build a taxon abundance vector by summing
    abundances of genomes that share the same taxon.

    Returns:
        torch.FloatTensor of shape [n_mixtures, num_taxa]
    """
    genome_to_taxon = dict(zip(labels_df['Genome'], labels_df[level]))
    num_taxa = len(taxon_to_idx)
    n = len(mixture_meta)
    Y = np.zeros((n, num_taxa), dtype=np.float32)

    for i, row in mixture_meta.iterrows():
        genome_ids  = json.loads(row['genome_ids'])
        abundances  = json.loads(row['abundances'])
        for genome, abund in zip(genome_ids, abundances):
            taxon = genome_to_taxon.get(genome)
            if taxon in taxon_to_idx:
                Y[i, taxon_to_idx[taxon]] += abund

    return torch.tensor(Y, dtype=torch.float32)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    """
    Return a dict of metrics over a loader:
      - val_kl:         KL divergence between predicted and true distributions
      - val_mae:        mean absolute error on abundances
      - val_topk_recall: fraction of true taxa recovered in top-k predictions,
                         where k = number of true taxa in each mixture
      - val_precision:  of taxa predicted present (pred > threshold), fraction truly present
      - val_f1:         harmonic mean of top-k recall and precision

    Threshold for presence/absence: 1 / num_taxa (i.e. above uniform)
    """
    model.eval()
    total_kl, total_mae, total = 0.0, 0.0, 0
    all_pred, all_true = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = F.softmax(logits, dim=1)

        kl  = F.kl_div((pred + 1e-10).log(), yb + 1e-10, reduction='batchmean')
        mae = (pred - yb).abs().mean()
        total_kl  += kl.item()  * xb.size(0)
        total_mae += mae.item() * xb.size(0)
        total     += xb.size(0)

        all_pred.append(pred.cpu())
        all_true.append(yb.cpu())

    all_pred = torch.cat(all_pred, dim=0)   # [n_val, num_taxa]
    all_true = torch.cat(all_true, dim=0)   # [n_val, num_taxa]
    num_taxa  = all_pred.shape[1]
    threshold = 1.0 / num_taxa              # above-uniform = predicted present

    # ── Top-k recall ────────────────────────────────────────────────────────
    # For each mixture, k = number of truly present taxa
    # Check how many of those k true taxa appear in the model's top-k predictions
    true_present = all_true > 0             # [n_val, num_taxa] bool
    k_per_mix    = true_present.sum(dim=1)  # [n_val] int
    topk_recalls = []
    for i in range(len(all_pred)):
        k = k_per_mix[i].item()
        if k == 0:
            continue
        topk_pred_idx  = all_pred[i].topk(k).indices          # model's top-k taxa
        true_idx       = true_present[i].nonzero(as_tuple=True)[0]
        hits = len(set(topk_pred_idx.tolist()) & set(true_idx.tolist()))
        topk_recalls.append(hits / k)
    topk_recall = float(np.mean(topk_recalls)) if topk_recalls else 0.0

    # ── Precision & F1 (threshold-based) ────────────────────────────────────
    pred_present = all_pred > threshold     # [n_val, num_taxa] bool
    tp = (pred_present & true_present).float().sum(dim=1)   # true positives per mixture
    fp = (pred_present & ~true_present).float().sum(dim=1)  # false positives per mixture
    fn = (~pred_present & true_present).float().sum(dim=1)  # false negatives per mixture

    precision = (tp / (tp + fp + 1e-10)).mean().item()
    recall_th = (tp / (tp + fn + 1e-10)).mean().item()
    f1 = (2 * precision * recall_th / (precision + recall_th + 1e-10))

    return {
        'val_kl':          total_kl  / total,
        'val_mae':         total_mae / total,
        'val_topk_recall': topk_recall,
        'val_precision':   precision,
        'val_f1':          f1,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/home/projects/roilab/anam/phD/tokenization_genomes/model_files')
    parser.add_argument('--out_dir', type=str,
                        default='/home/projects/roilab/anam/phD/tokenization_genomes/model_results/mixture_baseline')
    parser.add_argument('--level', type=str, default='order',
                        help='Taxonomic level to predict: order, family, genus')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adapter_rank', type=int, default=8)
    parser.add_argument('--freq_emb_dim', type=int, default=1024)
    parser.add_argument('--classifier_hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--embedding_file', type=str, default='embeddings_block28.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # -------------------- Load data --------------------
    print("Loading data...", flush=True)
    freqs       = load_npz(os.path.join(args.data_dir, 'mixtures.npz'))
    mixture_meta = pd.read_csv(os.path.join(args.data_dir, 'mixture_labels.csv'))
    labels_df   = pd.read_csv(os.path.join(args.data_dir, 'labels.csv'))
    embeddings  = torch.load(os.path.join(args.data_dir, args.embedding_file))

    freqs_torch = torch.tensor(freqs.toarray(), dtype=torch.float32)
    num_tokens  = freqs_torch.shape[1]
    print(f"  Mixtures: {freqs_torch.shape[0]}, Tokens: {num_tokens}", flush=True)

    # -------------------- Build taxon index --------------------
    all_taxa   = sorted(labels_df[args.level].dropna().unique())
    taxon_to_idx = {t: i for i, t in enumerate(all_taxa)}
    num_taxa   = len(taxon_to_idx)
    print(f"  Level={args.level}: {num_taxa} taxa", flush=True)

    # -------------------- Build labels --------------------
    print("Building abundance label matrix...", flush=True)
    Y = build_abundance_labels(mixture_meta, labels_df, args.level, taxon_to_idx)
    print(f"  Label matrix: {tuple(Y.shape)}, "
          f"avg taxa per mixture: {(Y > 0).float().sum(1).mean():.1f}", flush=True)

    # -------------------- Stratified split --------------------
    # Stratify by dominant taxon (argmax abundance) for balanced split
    #dominant = Y.argmax(1).numpy()
    train_idx, val_idx = train_test_split(
        np.arange(len(Y)), test_size=0.2, random_state=args.seed
    )
    X_train, Y_train = freqs_torch[train_idx], Y[train_idx]
    X_val,   Y_val   = freqs_torch[val_idx],   Y[val_idx]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}", flush=True)

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val),
                              batch_size=args.batch_size, shuffle=False)

    # -------------------- Model --------------------
    model = GenomeClassifier(
        frozen_embeddings=embeddings,
        num_tokens=num_tokens,
        num_classes=num_taxa,
        adapter_rank=args.adapter_rank,
        freq_emb_dim=args.freq_emb_dim,
        classifier_hidden=args.classifier_hidden,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------- Git hash --------------------
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    # -------------------- Training loop --------------------
    history = []
    best_val_kl = float('inf')
    print("\nTraining...", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            pred   = F.softmax(logits, dim=1)
            loss   = F.kl_div((pred + 1e-10).log(), yb + 1e-10, reduction='batchmean')
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_loss = running_loss / seen

        metrics = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'train_loss': train_loss, **metrics})

        if metrics['val_kl'] < best_val_kl:
            best_val_kl = metrics['val_kl']
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pt'))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f} | "
                  f"val_kl={metrics['val_kl']:.4f}  val_mae={metrics['val_mae']:.4f} | "
                  f"topk_recall={metrics['val_topk_recall']:.3f}  "
                  f"precision={metrics['val_precision']:.3f}  "
                  f"f1={metrics['val_f1']:.3f}", flush=True)

    # -------------------- Save --------------------
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, 'history.csv'), index=False)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'num_taxa': num_taxa, 'num_tokens': num_tokens,
                   'best_val_kl': best_val_kl, 'git_hash': git_hash}, f, indent=2)
    with open(os.path.join(args.out_dir, 'taxon_mapping.json'), 'w') as f:
        json.dump(taxon_to_idx, f, indent=2)

    print(f"\nDone. Best val_kl: {best_val_kl:.4f}", flush=True)
    print(f"Results saved to {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
