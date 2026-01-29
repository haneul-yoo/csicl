import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

COLORS = {
    "en": "#ffb000",
    "ko": "#fe6100",
    "cot_en2ko": "#009e73",
}


def load(out_dir: str):
    out_dir = Path(out_dir)
    data = np.load(out_dir / "qwen3_hidden_reps.npz")
    meta = json.loads((out_dir / "meta.json").read_text())
    reps = {k: data[k] for k in meta["conds"]}  # each: (L, N, D)
    return reps, meta


def pca_scatter_for_layer(reps, layer: int, out_png: Path, max_points=500, seed=0):
    # reps: dict cond->(L,N,D)
    conds = ["ko", "cot_en2ko", "en"]
    L, N, D = reps["ko"].shape
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    if N > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    X_list = []
    for c in conds:
        if c not in reps:
            raise KeyError(f"Condition '{c}' not found in reps. Available: {list(reps.keys())}")
        X_list.append(reps[c][layer, idx, :])
    X = np.concatenate(X_list, axis=0)

    Xz = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2, random_state=seed).fit_transform(Xz)

    M = len(idx)
    Z_ko = Z[0*M:1*M]
    Z_en2ko = Z[1*M:2*M]
    Z_en = Z[2*M:3*M]

    plt.figure(figsize=(3, 3))
    plt.scatter(Z_ko[:, 0], Z_ko[:, 1], s=10, label="Tgt.", c=COLORS["ko"])
    plt.scatter(Z_en2ko[:, 0], Z_en2ko[:, 1], s=10, label="CSICL", c=COLORS["cot_en2ko"])
    plt.scatter(Z_en[:, 0], Z_en[:, 1], s=10, label="En", c=COLORS["en"])

    for j in range(min(80, M)):
        plt.plot(
            [Z_ko[j, 0], Z_en2ko[j, 0], Z_en[j, 0]],
            [Z_ko[j, 1], Z_en2ko[j, 1], Z_en[j, 1]],
            linewidth=0.6,
            alpha=0.25,
        )

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, format="pdf", bbox_inches="tight")
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_points", type=int, default=500)

    # Optional: control which layers to plot
    ap.add_argument("--layers", type=str, default=None,
                    help='Comma-separated layer indices, e.g., "0,8,16,24,32"')
    ap.add_argument("--num_layers", type=int, default=None,
                    help="Pick this many layers evenly from 0..L-1")
    ap.add_argument("--step", type=int, default=None,
                    help="Plot every `step` layers (includes last layer)")

    args = ap.parse_args()

    reps, meta = load(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L = meta["n_layers_including_emb"]

    # Decide layers_to_plot
    if args.layers is not None:
        layers_to_plot = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    elif args.num_layers is not None:
        layers_to_plot = np.linspace(0, L - 1, args.num_layers, dtype=int).tolist()
    elif args.step is not None:
        layers_to_plot = list(range(0, L, args.step))
        if (L - 1) not in layers_to_plot:
            layers_to_plot.append(L - 1)
    else:
        # default: 3 layers (emb, early, last)
        layers_to_plot = [0, (L - 1) // 8, L - 1]

    # clamp & unique
    layers_to_plot = sorted(set(max(0, min(L - 1, l)) for l in layers_to_plot))
    print("[info] layers_to_plot:", layers_to_plot)

    for l in layers_to_plot:
        pca_scatter_for_layer(reps, l, out_dir / f"pca_layer{l}.pdf", max_points=args.max_points)


if __name__ == "__main__":
    main()
