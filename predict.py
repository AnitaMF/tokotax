"""
Inference script: predict per-sample taxonomy probabilities using a trained model.

Usage:
  python predict.py \
      --model_dir /path/to/baseline_results/block16_order_min5 \
      --samples /path/to/model_files/sample_frequencies.npz \
      --out_dir /path/to/predictions/

Outputs (in --out_dir):
  - predictions.csv : rows = samples, columns = taxonomic classes, values = probabilities
                      (each row sums to 1)
  - predictions_top.csv : top-3 predicted class + probability per sample (quick summary)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import load_npz


# ----------------------------------------------------------------------
# Model definitions — MUST match the training script that produced best_model.pt
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
        return freqs @ adapted


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
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, freqs):
        adapter_emb = self.adapter_pool(freqs)
        freq_emb = self.freq_encoder(freqs)
        combined = torch.cat([adapter_emb, freq_emb], dim=1)
        return self.classifier(combined)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing best_model.pt, config.json, class_mapping.json')
    parser.add_argument('--samples', type=str, required=True,
                        help='Path to sample frequency matrix (.npz, sparse format)')
    parser.add_argument('--sample_names', type=str, default=None,
                        help='Optional path to .npy file with sample names (one per row of samples)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Where to write predictions.csv')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference (small samples can use anything)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # -------------------- Load config and class mapping --------------------
    with open(os.path.join(args.model_dir, 'config.json')) as f:
        cfg = json.load(f)
    with open(os.path.join(args.model_dir, 'class_mapping.json')) as f:
        class_to_idx = json.load(f)
    # Reverse mapping: index -> class name
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"Model trained on level={cfg['level']}, {cfg['num_classes']} classes", flush=True)
    print(f"Best val_acc during training: {cfg.get('best_val_acc', 'unknown')}", flush=True)

    # -------------------- Load frozen embeddings (must match training) --------------------
    embedding_file = cfg['embedding_file']
    embeddings_path = os.path.join(cfg['data_dir'], embedding_file)
    print(f"Loading embeddings: {embeddings_path}", flush=True)
    embeddings = torch.load(embeddings_path)
    print(f"  embeddings: {tuple(embeddings.shape)}", flush=True)

    # -------------------- Load samples --------------------
    print(f"Loading samples: {args.samples}", flush=True)
    samples_sparse = load_npz(args.samples)
    print(f"  samples matrix: {samples_sparse.shape} "
          f"(samples × tokens)", flush=True)

    # Verify token alignment
    assert samples_sparse.shape[1] == cfg['num_tokens'], (
        f"Token mismatch: samples have {samples_sparse.shape[1]} tokens, "
        f"model expects {cfg['num_tokens']}. They must be in the SAME order as training."
    )
    print("  ✓ Token count matches training", flush=True)

    samples = torch.tensor(samples_sparse.toarray(), dtype=torch.float32)
    n_samples = samples.shape[0]

    # Optional sample names (row labels)
    if args.sample_names:
        sample_names = np.load(args.sample_names, allow_pickle=True)
        assert len(sample_names) == n_samples, "sample_names length mismatch"
        sample_names = [str(s) for s in sample_names]
    else:
        sample_names = [f"sample_{i:03d}" for i in range(n_samples)]

    # Verify rows look like proper frequency vectors (sum ~ 1)
    row_sums = samples.sum(dim=1)
    print(f"  Row sums — min: {row_sums.min():.4f}, "
          f"max: {row_sums.max():.4f}, "
          f"mean: {row_sums.mean():.4f}", flush=True)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
        print("  ⚠ WARNING: row sums are not ~1. Did you forget to normalize to frequencies?", flush=True)

    # -------------------- Rebuild model and load weights --------------------
    print("Building model and loading weights...", flush=True)
    model = GenomeClassifier(
        frozen_embeddings=embeddings,
        num_tokens=cfg['num_tokens'],
        num_classes=cfg['num_classes'],
        adapter_rank=cfg['adapter_rank'],
        freq_emb_dim=cfg['freq_emb_dim'],
        classifier_hidden=cfg['classifier_hidden'],
        dropout=cfg['dropout'],
    ).to(device)

    state_dict = torch.load(os.path.join(args.model_dir, 'best_model.pt'),
                            map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  ✓ Model loaded", flush=True)

    # -------------------- Inference --------------------
    print(f"\nRunning inference on {n_samples} samples...", flush=True)
    all_probs = []
    with torch.no_grad():
        for i in range(0, n_samples, args.batch_size):
            batch = samples[i:i + args.batch_size].to(device)
            logits = model(batch)                      # [B, num_classes]
            probs = torch.softmax(logits, dim=1)       # convert logits to probabilities
            all_probs.append(probs.cpu().numpy())
    all_probs = np.vstack(all_probs)                   # [n_samples, num_classes]
    print(f"  Output shape: {all_probs.shape}", flush=True)

    # -------------------- Save full probability matrix --------------------
    probs_df = pd.DataFrame(all_probs, columns=class_names, index=sample_names)
    probs_df.index.name = 'sample'
    probs_path = os.path.join(args.out_dir, 'predictions.csv')
    probs_df.to_csv(probs_path)
    print(f"\nSaved full probabilities: {probs_path}", flush=True)
    print(f"  Rows sum to: {probs_df.sum(axis=1).describe().loc[['min', 'max']].to_dict()}", flush=True)

    # -------------------- Save top-3 summary --------------------
    top3_rows = []
    for i, name in enumerate(sample_names):
        top_idx = np.argsort(all_probs[i])[::-1][:3]
        for rank, idx in enumerate(top_idx, start=1):
            top3_rows.append({
                'sample': name,
                'rank': rank,
                'class': class_names[idx],
                'probability': float(all_probs[i, idx]),
            })
    top_df = pd.DataFrame(top3_rows)
    top_path = os.path.join(args.out_dir, 'predictions_top3.csv')
    top_df.to_csv(top_path, index=False)
    print(f"Saved top-3 summary: {top_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
