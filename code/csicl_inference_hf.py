from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import argparse, os, sys, time, json, logging, csv, random
from typing import List, Dict, Optional


# -------------------- Args & Logging --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--models", type=str, nargs="*", default=[
        "Qwen/Qwen3-32B",
        "deepseek-ai/DeepSeek-V3.1"
    ])
    p.add_argument("--xicl-path", type=str, required=True,
                   help="Path to X-ICL setting CSV (columns: row, content)")
    p.add_argument("--per-category-cap", type=int, default=600)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Log file: %s", log_file)


# -------------------- Utils --------------------

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_tqdm_if_not_tty():
    if not sys.stdout.isatty():
        tqdm.disable = True


def get_resume_index(resume_path: Path):
    if resume_path.exists():
        try:
            return json.loads(resume_path.read_text()).get("index", 0)
        except Exception:
            return 0
    return 0


def save_resume_index(resume_path: Path, idx: int):
    resume_path.write_text(json.dumps({"index": idx}))


def load_global_mmlu_as_df() -> Dict[str, pd.DataFrame]:
    data_en = load_dataset("CohereLabs/Global-MMLU", "en")['test'].to_pandas()
    data_ko = load_dataset("CohereLabs/Global-MMLU", "ko")['test'].to_pandas()
    data_es = load_dataset("CohereLabs/Global-MMLU", "es")['test'].to_pandas()
    return {'en': data_en, 'ko': data_ko, 'es': data_es}


def stratified_cap_sample(df: pd.DataFrame, cap_per_category: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby('subject_category', sort=False):
        n = min(len(g), cap_per_category)
        parts.append(g.sample(n=n, random_state=seed) if len(g) > n else g)
    capped = pd.concat(parts, axis=0, ignore_index=True)
    return capped


def build_base_messages(csv_path: str) -> List[Dict[str, str]]:
    msgs = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        role, content = row['row'], row['content']
        msgs.append({"role": role, "content": content})
    return msgs


# -------------------- HF Model Wrapper --------------------

class HFChatModel:
    def __init__(self, model_name: str, device: str):
        logging.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.device = device
        logging.info("Model loaded successfully.")

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024) -> str:
        # Simple chat-style concatenation
        text = ""
        for m in messages:
            prefix = "User: " if m["role"] == "user" else "System: " if m["role"] == "system" else "Assistant: "
            text += f"{prefix}{m['content']}\n"
        text += "Assistant: "

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the part after last "Assistant:"
        if "Assistant:" in decoded:
            decoded = decoded.split("Assistant:")[-1].strip()
        return decoded


# -------------------- Main --------------------

def main():
    args = parse_args()
    out_dir = Path(args.out)
    setup_logging(out_dir)
    disable_tqdm_if_not_tty()
    seed_all(args.seed)

    xicl_path = Path(args.xicl_path)
    xicl_name = xicl_path.stem
    base_messages = build_base_messages(xicl_path)

    datasets = load_global_mmlu_as_df()

    for model_name in args.models:
        logging.info(f"=== Running model: {model_name} ===")
        model = HFChatModel(model_name, args.device)

        for lang, df_full in datasets.items():
            df = stratified_cap_sample(df_full, args.per_category_cap, args.seed)
            logging.info(f">>> Evaluation language: {lang} | sampled {len(df)} / original {len(df_full)}")

            safe_model_tag = model_name.split("/")[-1].replace("/", "_")
            out_csv = out_dir / f"globalmmlu_{xicl_name}_{lang}_{safe_model_tag}.csv"
            resume_path = out_dir / f"resume_{xicl_name}_{lang}_{safe_model_tag}.json"
            out_dir.mkdir(parents=True, exist_ok=True)

            header = ['sample_id','subject','subject_category','question',
                      'option_a','option_b','option_c','option_d',
                      'answer','response']
            if not out_csv.exists():
                with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(header)

            start_idx = get_resume_index(resume_path)
            n_total = len(df)
            logging.info("Resuming from index %d / %d (csv=%s)", start_idx, n_total, out_csv.name)

            buffer_rows = []
            processed = 0

            for idx, row in tqdm(df.iterrows(), total=n_total):
                if idx < start_idx:
                    continue

                messages = list(base_messages)
                messages.append({
                    "role": "user",
                    "content": (
                        f"{row['question']}\n\n"
                        f"A) {row['option_a']}\n"
                        f"B) {row['option_b']}\n"
                        f"C) {row['option_c']}\n"
                        f"D) {row['option_d']}"
                    )
                })

                try:
                    response_text = model.generate(messages, max_new_tokens=args.max_new_tokens)
                except Exception as e:
                    logging.warning(f"Generation failed at idx={idx}: {repr(e)}")
                    response_text = ""

                buffer_rows.append([
                    row.get('sample_id', idx), row.get('subject', ''), row.get('subject_category',''),
                    row['question'], row['option_a'], row['option_b'], row['option_c'], row['option_d'],
                    row['answer'], response_text
                ])

                processed += 1
                if processed % args.save_every == 0:
                    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerows(buffer_rows)
                    buffer_rows.clear()
                    save_resume_index(resume_path, idx + 1)
                    logging.info(f"Saved progress at idx {idx} -> {out_csv}")

            if buffer_rows:
                with open(out_csv, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(buffer_rows)
            save_resume_index(resume_path, n_total)
            logging.info(f"Language={lang} done. Output: {out_csv}")

    logging.info("All done.")


if __name__ == "__main__":
    main()
