import argparse
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-5-2025-08-07"  # greedy decoding: temperature=0


def load_instructions(json_path: str) -> tuple[str, str]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if "stage1" not in data or "stage2" not in data:
        raise KeyError('Instruction JSON must have keys: "stage1", "stage2"')
    return data["stage1"], data["stage2"]


def generate_gradual_demo(E: str, K: str, stage1_inst: str, stage2_inst: str) -> str:
    # Stage 1: get code-switching sentence C
    s1 = client.responses.create(
        model=MODEL,
        instructions=stage1_inst,
        input=f"<English> {E}\n<Korean> {K}\n<Code-Switching>",
        temperature=0,
        top_p=1,
    )
    C = s1.output_text.strip()

    # Stage 2: final 5-step gradual outputs
    s2 = client.responses.create(
        model=MODEL,
        instructions=stage2_inst,
        input=f"<Korean> {K}\n<English> {E}\n<Code-Switching> {C}\nOutput:",
        temperature=0,
        top_p=1,
    )
    return s2.output_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--E", required=True, help="English instance")
    parser.add_argument("--K", required=True, help="Korean instance")
    parser.add_argument("--instructions_json", required=True, help='Path to JSON with keys "stage1" and "stage2"')
    args = parser.parse_args()

    stage1_inst, stage2_inst = load_instructions(args.instructions_json)
    print(generate_gradual_demo(args.E, args.K, stage1_inst, stage2_inst))
