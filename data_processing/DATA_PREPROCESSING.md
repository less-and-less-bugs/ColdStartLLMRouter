# Data preprocessing

How benchmark files become CSVs for routing experiments. Main scripts: **`multidata_unify.py`** (one merged table) and **`construct_router_data.py`** (per-model responses and scores).

## Pipeline (short)

```text
download_*.py  →  raw JSON/Parquet under data/<task>/
       │
       ├─► multidata_unify.py  →  single CSV (optional; no splits)
       │
       └─► generate_dataset_splits.py  →  <data_root>/data/<TASK>/{train,val,test}.csv
                    │
                    ▼
            construct_router_data.py  →  <TASK>/<model_name>.csv
```

## Steps (main path)

1. **Raw data** — Run `data_processing/download_*.py` or place files where `generate_dataset_splits.py` / `multidata_unify.py` expect them (see script configs).
2. **Splits** — Run `generate_dataset_splits.py`; set `data_dir` in `__main__` to your root. You need `train.csv`, `val.csv`, `test.csv` per task (columns include `query`, `ground_truth`, `metric`). Required for `construct_router_data.py`.
3. **API** — Set keys in `configs/models.yaml` (or env). Match `ProviderType` in `construct_router_data.py` (`DataBuilder`, currently OpenRouter-style).
4. **Env** — `export PYTHONPATH=<project_root>` and run from project root.
5. **Build model CSVs** — Edit `construct_router_data.py` `main()`: `dataset_names`, `data_dirs`, `models_to_test`. Run `python data_processing/construct_router_data.py`. Output: `{model_name}.csv` per task (`response`, tokens, `response_time`, `effect`, …). `use_saved_flag=True` skips redoing API calls when split outputs already exist.

**Optional:** `multidata_unify.py` only exports one flat CSV (no train/val/test); it does not feed `construct_router_data.py` directly.

## `multidata_unify.py`

Merges tasks into one CSV: `task_id`, `sub_task` (MMLU subjects only), `query`, `ground_truth`, `metric`, `task_description`.

| task_id | Source | Notes |
|---------|--------|--------|
| alpaca_data, GSM8K, multi_news | `data/.../*.json` | Concat `instruction`+`input`; GT from `output`/`answer` |
| SQUAD | `data/SQUAD/SQUAD.parquet` | `question`; first `answers.text` |
| MBPP | `data/mbpp/mbpp_all.json` | Templated prompt + tests; GT `code` |
| mmlu_redux | `data/mmlu_redux/*.json` | MCQ string; GT `(A)`–`(D)` |

`generate_unified_qa_dataset(output_path, sample_size, mmlu_sample_size)` — cap rows per task / per MMLU subject. Run: `python data_processing/multidata_unify.py` (tune `__main__`).

## `construct_router_data.py`

Loads three splits, calls the LLM per row, evaluates with `LLMProvider.eval(metric=...)`, writes `{model_name}.csv`. Uses `gpt2` tokenizer for usage fallback and **truncates `multi_news` to 3000 tokens**. Retries: 3× with backoff.

Depends on `llm_engine.py`, `configs/models.yaml`, `transformers`, `pandas`, `pyyaml`.

## Other files

| File | Role |
|------|------|
| `generate_dataset_splits.py` | Writes per-task `train/val/test.csv` |
| `generate_train_val_test.py` | Alternative: `unified_qa_data_{split}.csv` in one folder |
| `utils.py`, `llm_engine.py` | I/O and API + metric eval |

Keep paths and column names in sync with `DataBuilder.process_split` if you change them.
