import os, json, random, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    BNB_OK = True
except Exception:
    BNB_OK = False


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_cap_sample(df: pd.DataFrame, cap_per_category: int, total_cap: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby('subject_category', sort=False):
        n = min(len(g), cap_per_category)
        parts.append(g.sample(n=n, random_state=seed) if len(g) > n else g)
    capped = pd.concat(parts, axis=0, ignore_index=True)
    if len(capped) > total_cap:
        capped = capped.sample(n=total_cap, random_state=seed).reset_index(drop=True)
    else:
        capped = capped.reset_index(drop=True)
    return capped


def load_global_mmlu(lang: str) -> pd.DataFrame:
    return load_dataset("CohereLabs/Global-MMLU", lang)['test'].to_pandas()


def build_fewshot_messages(csv_path: str, instructions: str, fs_name: str) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": instructions}]
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        real_q = str(row['question']).split('\n')[0].strip('1. ').strip()
        msgs.append({"role": "user", "content": f"{real_q}\n\n{row['options']}"})

        if 'cot' in fs_name:
            msgs.append({
                "role": "assistant",
                "content": (
                    "Let's gradually translate this non-English query into English, then think in English, and finally answer the question.\n"
                    f"{row['question']}\n\n"
                    f"The answer is {row['answer']}."
                )
            })
        else:
            msgs.append({"role": "assistant", "content": str(row['answer']).strip()})

    return msgs


def apply_chat(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def tokenize_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def get_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for key in ["model.embed_tokens", "embed_tokens", "transformer.wte", "model.tok_embeddings"]:
            if key in model.hf_device_map:
                return torch.device(model.hf_device_map[key])
        return torch.device(next(iter(model.hf_device_map.values())))
    return next(model.parameters()).device


@torch.inference_mode()
def forward_hidden(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.hidden_states


def mean_pool_span(hidden_seq: torch.Tensor, span: Tuple[int, int]) -> torch.Tensor:
    """
    hidden_seq: (seq, dim)
    span: (start, end)
    """
    s, e = span
    s = max(0, min(s, hidden_seq.shape[0]))
    e = max(s, min(e, hidden_seq.shape[0]))
    if e <= s:
        return hidden_seq[max(0, s-1):s].mean(dim=0) if s > 0 else hidden_seq[0].clone()
    return hidden_seq[s:e].mean(dim=0)


def slice_shard(df: pd.DataFrame, shard_id: int, num_shards: int) -> pd.DataFrame:
    idx = np.arange(len(df))
    idx = idx[idx % num_shards == shard_id]
    return df.iloc[idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--fewshots_en", type=str, required=True)
    ap.add_argument("--fewshots_ko", type=str, required=True)
    ap.add_argument("--fewshots_cot_en2ko", type=str, required=True)

    ap.add_argument("--instructions_cot", type=str, required=True)
    ap.add_argument("--instructions_plain", type=str, required=True)


    ap.add_argument("--per_category_cap", type=int, default=600)
    ap.add_argument("--total_cap", type=int, default=3600)

    ap.add_argument("--load_in_4bit", action="store_true")

    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)

    ap.add_argument("--log_every", type=int, default=20)

    args = ap.parse_args()

    seed_all(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    few_shots = {
        "en": args.fewshots_en,
        "ko": args.fewshots_ko,
        "cot_en2ko": args.fewshots_cot_en2ko,
    }


    instructions = {
        "en": args.instructions_plain,
        "ko": args.instructions_plain,
        "cot_en2ko": args.instructions_cot,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    quant_cfg = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if args.load_in_4bit:
        if not BNB_OK:
            raise RuntimeError("bitsandbytes not available. Install bitsandbytes or run without --load_in_4bit.")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch_dtype if quant_cfg is None else None,
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )
    model.eval()

    input_device = get_input_device(model)
    print(f"[info] shard {args.shard_id}/{args.num_shards}")
    print(f"[info] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[info] visible cuda device count={torch.cuda.device_count()}")
        print(f"[info] current_device={torch.cuda.current_device()}")
        print(f"[info] input_device={input_device}")
    print(f"[info] hf_device_map={getattr(model, 'hf_device_map', None)}")

    df_ko = stratified_cap_sample(load_global_mmlu("ko"), args.per_category_cap, args.total_cap, args.seed)
    df_en = stratified_cap_sample(load_global_mmlu("en"), args.per_category_cap, args.total_cap, args.seed)

    if "sample_id" in df_ko.columns and "sample_id" in df_en.columns:
        df_all = pd.merge(df_ko, df_en, on="sample_id", suffixes=("_ko", "_en"))
    else:
        n = min(len(df_ko), len(df_en))
        df_all = pd.concat(
            [df_ko.iloc[:n].add_suffix("_ko").reset_index(drop=True),
             df_en.iloc[:n].add_suffix("_en").reset_index(drop=True)],
            axis=1
        )

    all_N = len(df_all)
    df = slice_shard(df_all, args.shard_id, args.num_shards)
    N = len(df)
    print(f"[info] total merged N={all_N}, this shard N={N}")

    base_msgs = {}
    prefix_len = {}
    for fs_name, fs_csv in few_shots.items():
        instruction = instructions[fs_name]
        msgs = build_fewshot_messages(fs_csv, instruction, fs_name)
        base_msgs[fs_name] = msgs

        base_text = apply_chat(tokenizer, msgs)
        prefix_len[fs_name] = tokenize_len(tokenizer, base_text)

    conds = ["ko", "en", "cot_en2ko"]

    dry_messages = list(base_msgs["ko"])
    dry_messages.append({"role": "user", "content": "Test\n\nA) 1\nB) 2\nC) 3\nD) 4"})
    dry_text = apply_chat(tokenizer, dry_messages)
    dry_tok = tokenizer(dry_text, return_tensors="pt", add_special_tokens=False)
    dry_tok = {k: v.to(input_device) for k, v in dry_tok.items()}

    hs = forward_hidden(model, dry_tok["input_ids"], dry_tok.get("attention_mask"))
    n_layers = len(hs)
    d_model = hs[-1].shape[-1]
    print(f"[info] n_layers(incl emb)={n_layers}, d_model={d_model}")

    reps = {c: np.zeros((n_layers, N, d_model), dtype=np.float32) for c in conds}

    # ---- main loop ----
    for i in tqdm(range(N), desc=f"extract(shard={args.shard_id})"):
        row = df.iloc[i]

        # Korean fields
        q_ko = row.get("question_ko", row.get("question", ""))
        ua_ko = row.get("option_a_ko", row.get("option_a", ""))
        ub_ko = row.get("option_b_ko", row.get("option_b", ""))
        uc_ko = row.get("option_c_ko", row.get("option_c", ""))
        ud_ko = row.get("option_d_ko", row.get("option_d", ""))

        # English fields
        q_en = row.get("question_en", row.get("question", ""))
        ua_en = row.get("option_a_en", row.get("option_a", ""))
        ub_en = row.get("option_b_en", row.get("option_b", ""))
        uc_en = row.get("option_c_en", row.get("option_c", ""))
        ud_en = row.get("option_d_en", row.get("option_d", ""))

        user_ko = f"{q_ko}\n\nA) {ua_ko}\nB) {ub_ko}\nC) {uc_ko}\nD) {ud_ko}"
        user_en = f"{q_en}\n\nA) {ua_en}\nB) {ub_en}\nC) {uc_en}\nD) {ud_en}"

        cond_user = {
            "ko": user_ko,
            "en": user_en,
            "cot_en2ko": user_en,
        }

        for cond in conds:
            messages = list(base_msgs[cond])
            messages.append({"role": "user", "content": cond_user[cond]})

            full_text = apply_chat(tokenizer, messages)

            tok = tokenizer(full_text, return_tensors="pt", add_special_tokens=False, truncation=True)
            tok = {k: v.to(input_device) for k, v in tok.items()}

            hidden_states = forward_hidden(model, tok["input_ids"], tok.get("attention_mask"))

            full_len = tok["input_ids"].shape[1]
            s = min(prefix_len[cond], full_len)
            e = full_len
            span = (s, e)

            for l in range(n_layers):
                h = hidden_states[l][0]  # (seq, dim)
                v = mean_pool_span(h, span)
                reps[cond][l, i, :] = v.detach().float().cpu().numpy()

        if args.log_every > 0 and (i + 1) % args.log_every == 0:
            print(f"[info] processed {i+1}/{N} samples on shard {args.shard_id}")

    npz_path = out_dir / f"qwen3_hidden_reps.shard{args.shard_id:02d}.npz"
    meta_path = out_dir / f"meta.shard{args.shard_id:02d}.json"

    np.savez_compressed(npz_path, **{c: reps[c] for c in conds})

    meta = {
        "model": args.model,
        "n_layers_including_emb": n_layers,
        "d_model": d_model,
        "N": N,
        "conds": conds,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "note": "mean-pooled over last user message token span; span computed via cached prefix token length",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[saved] {npz_path}")
    print(f"[saved] {meta_path}")


if __name__ == "__main__":
    main()
