from openai import OpenAI

from datasets import load_dataset
import pandas as pd
import numpy as np
import torch 
from tqdm import tqdm
from pathlib import Path
import argparse, os, sys, time, json, logging, csv, random
from typing import List, Dict

# -------------------- Args & Logging --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--api-key', type=str, required=True, help="OpenRouter API Key")
    p.add_argument('--max-new-tokens', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--models', type=str, nargs='*', default=[
        'google/gemini-2.5-flash',
        'x-ai/grok-4-fast'
    ])
    p.add_argument('--xicl-path', type=str, help='path to X-ICL setting (i.e., xicl_setting.csv)')
    p.add_argument('--per-category-cap', type=int, default=600)
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
    try:
        if not sys.stdout.isatty():
            tqdm.__init__ = lambda *a, **k: None
    except Exception:
        pass

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

# -------------------- OpenRouter Client & Call --------------------

def get_openrouter_client(api_key: str) -> OpenAI:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client

def call_chat_completion_with_retries(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_retries: int = 5,
) -> str | None:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return resp.choices[0].message.content  # 후처리 없음, 그대로 반환
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30) + random.random()
            logging.warning("ERROR: fail to call API (attempt %d/%d): %s | retry after %.1fs",
                            attempt, max_retries, repr(e), wait)
            time.sleep(wait)
    logging.error("ERROR: finally fail to call API %s", repr(last_err))
    return None

# -------------------- Main --------------------

def main():
    args = parse_args()
    out_dir = Path(args.out)
    setup_logging(out_dir)
    disable_tqdm_if_not_tty()
    seed_all(args.seed)

    logging.info("Device: %s | CUDA: %s", "cuda" if torch.cuda.is_available() else "cpu", torch.cuda.is_available())

    xicl_path = Path(args.xicl_setting)
    xicl_name = xicl_path.stem
    with open(xicl_path, 'r') as f:
        xicl_csv = f.read().strip()

    datasets = load_global_mmlu_as_df()
    client = get_openrouter_client(args.api_key) or os.getenv("OPENROUTER_API_KEY")

    logging.info(">>> X-ICL setting: %s", xicl_name)
    base_messages = build_base_messages(xicl_csv)

    for model_name in args.models:
        logging.info(">>> Model: %s", model_name)

        for lang, df_full in datasets.items():
            df = stratified_cap_sample(
                df_full,
                cap_per_category=args.per_category_cap,
                seed=args.seed
            )
            logging.info(">>> Evaluation language: %s | sampled %d / original %d",
                         lang, len(df), len(df_full))

            safe_model_tag = model_name.split("/")[-1].replace("/", "_")
            out_csv = out_dir / f'raw_globalmmlu_{xicl_name}_{lang}_{safe_model_tag}.csv'
            resume_path = out_dir / f'resume_{xicl_name}_{lang}_{safe_model_tag}.json'
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

                raw = call_chat_completion_with_retries(
                    client=client,
                    model=model_name,
                    messages=messages,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    max_retries=5,
                )

                response_text = raw if raw is not None else ""

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
                    logging.info("Saved progress at idx %d -> %s", idx, out_csv)

            if buffer_rows:
                with open(out_csv, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(buffer_rows)
                buffer_rows.clear()
            save_resume_index(resume_path, n_total)
            logging.info("Language=%s done. Output: %s", lang, out_csv)

    logging.info("All done.")

if __name__ == "__main__":
    main()
