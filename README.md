# CSICL

This is an official repository of **Code-Switching In-Context Learning for Cross-Lingual Transfer of Large Language Models**.

> We introduce code-switching in-context learning (CSICL), which gradually transitions non-English inputs to English to bridge the translation barrier and enhance multilingual reasoning in LLMs.

## Keywords
`Code-switching`, `In-context learning`, `Multilingualism`, `Cross-lingual Transfer`, `Chain-of-Thought`

## Abstract
While large language models (LLMs) exhibit strong multilingual abilities, their reliance on English as latent representations creates a *translation barrier*, where reasoning implicitly depends on internal translation into English. 
When this process fails, performance in non-English languages deteriorates sharply, limiting the inclusiveness of LLM-based applications. 
Existing cross-lingual in-context learning (X-ICL) methods primarily leverage monolingual demonstrations, often failing to mitigate this barrier and instead reinforcing it.
In this work, we introduce code-switching in-context learning (CSICL), a simple yet effective prompting strategy that progressively transitions from a target language to English within demonstrations and instruction to facilitate their latent reasoning in English.
By explicitly scaffolding the reasoning process through controlled code-switching, CSICL acts as an implicit linguistic bridge that enhances cross-lingual alignment and reduces reliance on the translation barrier.
We conduct extensive experiments across 4 LLMs, 6 tasks, and 10 languages, spanning both knowledge-intensive and reasoning-oriented domains.
Our results demonstrate that CSICL consistently outperforms X-ICL baselines, achieving gains of 3.1%p and 1.9%p in both target and unseen languages, respectively.
The improvement is even more pronounced in low-resource settings, with gains of 14.7% in target and 5.3% in unseen languages.
These findings establish code-switching as a principled and robust approach for overcoming the translation barrier during inference, moving LLMs toward more equitable and effective multilingual systems.

## Code
```bash
python csicl_inference_openrouter.py \
  --out results \
  --xicl-setting prompts/xicl_setting.csv \
  --models google/gemini-2.5-flash x-ai/grok-4-fast \
  --per-category-cap 600
```

```bash
python csicl_inference_hf.py \
  --out results \
  --xicl-path prompts/xicl_setting.csv \
  --models Qwen/Qwen3-32B deepseek-ai/DeepSeek-V3.1 \
  --per-category-cap 600

```
