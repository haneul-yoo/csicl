import json
from pathlib import Path
import numpy as np


def main(in_dir: str, out_dir: str, num_shards: int):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta0 = None
    reps_acc = None

    for sid in range(num_shards):
        npz_path = in_dir / f"qwen3_hidden_reps.shard{sid:02d}.npz"
        meta_path = in_dir / f"meta.shard{sid:02d}.json"

        if not npz_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"missing shard files for sid={sid}: {npz_path} / {meta_path}")

        data = np.load(npz_path)
        meta = json.loads(meta_path.read_text())

        if meta0 is None:
            meta0 = meta
            conds = meta["conds"]
            reps_acc = {c: [] for c in conds}
        else:
            # sanity
            if meta["conds"] != meta0["conds"]:
                raise ValueError(f"conds mismatch at shard {sid}")

        for c in meta["conds"]:
            reps_acc[c].append(data[c])

    merged = {c: np.concatenate(reps_acc[c], axis=1) for c in reps_acc}
    out_npz = out_dir / "qwen3_hidden_reps.npz"
    np.savez_compressed(out_npz, **merged)

    meta0["N"] = int(merged[meta0["conds"][0]].shape[1])
    meta0.pop("shard_id", None)
    meta0.pop("num_shards", None)
    out_meta = out_dir / "meta.json"
    out_meta.write_text(json.dumps(meta0, ensure_ascii=False, indent=2))

    print("[saved]", out_npz)
    print("[saved]", out_meta)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_shards", type=int, default=8)
    args = ap.parse_args()
    main(args.in_dir, args.out_dir, args.num_shards)
