"""
Baseline training script for genome -> taxonomic order classification.

Model: two-branch GenomeClassifier
  - Branch 1: LoRA adapter on frozen Evo2 token embeddings + frequency-weighted pooling
  - Branch 2: small MLP encoder on the raw frequency vector
  - Concatenate both -> classifier MLP -> per-order logits

Baseline setup:
  - Input: frequencies
  - Filter: orders with >= 10 genomes
  - Split: stratified 80/20 by order
  - Loss: cross-entropy
  - Optimizer: Adam, lr=1e-3
  - Epochs: 100

Run as an LSF job (see submit_baseline.sh).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# Model definitions
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
        adapted = self.E + self.B(self.A(self.E))   # [num_tokens, embed_dim]
        return freqs @ adapted                       # [batch, embed_dim]


class FreqEncoder(nn.Module):
    """Encode raw frequency vector into a dense embedding."""
    def __init__(self, num_tokens, out_dim=1024, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_tokens, out_dim),
            nn.ReLU(),
        )

    def forward(self, freqs):
        return self.net(freqs)


class GenomeClassifier(nn.Module):
    """Two-branch genome classifier."""
    def __init__(self, frozen_embeddings, num_tokens, num_classes,
                 adapter_rank=8, freq_emb_dim=1024,
                 classifier_hidden=512, dropout=0.2):
        super().__init__()
        embed_dim = frozen_embeddings.shape[1]
        self.adapter_pool = AdapterPool(frozen_embeddings, rank=adapter_rank)
        self.freq_encoder = FreqEncoder(num_tokens, out_dim=freq_emb_dim, dropout=dropout)
        combined_dim = embed_dim + freq_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, classifier_hidden),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, freqs):
        adapter_emb = self.adapter_pool(freqs)
        freq_emb = self.freq_encoder(freqs)
        combined = torch.cat([adapter_emb, freq_emb], dim=1)
        return self.classifier(combined)


# ----------------------------------------------------------------------
# Evaluation helper
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Return (avg_loss, accuracy) over a loader. Sets model to eval mode."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/home/projects/roilab/anam/phD/tokenization_genomes/model_files')
    parser.add_argument('--out_dir', type=str,
                        default='/home/projects/roilab/anam/phD/tokenization_genomes/baseline_results')
    parser.add_argument('--level', type=str, default='order')
    parser.add_argument('--min_genomes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
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

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # -------------------- Load data --------------------
    print("Loading data...", flush=True)
    freqs = load_npz(os.path.join(args.data_dir, 'frequencies.npz'))
    labels_df = pd.read_csv(os.path.join(args.data_dir, 'labels.csv'))
    embeddings = torch.load(os.path.join(args.data_dir, args.embedding_file))

    freqs_torch = torch.tensor(freqs.toarray(), dtype=torch.float32)
    num_tokens = freqs_torch.shape[1]
    print(f"  freqs: {tuple(freqs_torch.shape)}, embeddings: {tuple(embeddings.shape)}", flush=True)

    # -------------------- Filter classes --------------------
    counts = labels_df[args.level].value_counts()
    valid = set(counts[counts >= args.min_genomes].index)
    keep_mask = labels_df[args.level].isin(valid).values

    class_list = sorted(valid)
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    num_classes = len(class_to_idx)

    X = freqs_torch[keep_mask]
    y = np.array([class_to_idx[c] for c in labels_df.loc[keep_mask, args.level]])
    print(f"  Level={args.level}: {num_classes} classes, {keep_mask.sum()} genomes kept", flush=True)

    # -------------------- Stratified split --------------------
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=args.seed
    )
    X_train, y_train = X[train_idx], torch.tensor(y[train_idx], dtype=torch.long)
    X_val, y_val = X[val_idx], torch.tensor(y[val_idx], dtype=torch.long)
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}", flush=True)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=args.batch_size, shuffle=False)

    # -------------------- Model --------------------
    model = GenomeClassifier(
        frozen_embeddings=embeddings,
        num_tokens=num_tokens,
        num_classes=num_classes,
        adapter_rank=args.adapter_rank,
        freq_emb_dim=args.freq_emb_dim,
        classifier_hidden=args.classifier_hidden,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}", flush=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------- Training loop --------------------
    history = []
    best_val_acc = 0.0
    print("\nTraining...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_loss = running_loss / seen

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'val_acc': val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pt'))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}", flush=True)

    # -------------------- Save results --------------------
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, 'history.csv'), index=False)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'num_classes': num_classes,
                   'num_tokens': num_tokens, 'best_val_acc': best_val_acc}, f, indent=2)
    with open(os.path.join(args.out_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"\nDone. Best val_acc: {best_val_acc:.4f}", flush=True)
    print(f"Results saved to {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
