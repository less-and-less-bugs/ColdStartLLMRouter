# AgenticRouter

Official implementation of the ACL 2026 Main paper:

**[Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](https://arxiv.org/abs/2604.09377)**.

AgenticRouter is a multi-stage LLM router that estimates task type and per-model cost / quality / latency from a **query-type knowledge graph (KG)** and **offline benchmark runs**, then selects models under constraints via a utility objective.

If you find this repository useful, please cite the paper.

---

## Paper

- **Title:** Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios
- **Status:** Accepted by ACL 2026 Main
- **Paper:** https://arxiv.org/abs/2604.09377

---

## Repository layout

| Path | Purpose |
|------|---------|
| [`data_processing/`](data_processing/) | Download scripts, train/val/test splits, LLM calls to fill per-model CSVs |
| [`data_processing/DATA_PREPROCESSING.md`](data_processing/DATA_PREPROCESSING.md) | **Step-by-step data pipeline** (raw data → splits → `{model}.csv`) |
| [`src/components/`](src/components/) | KG construction, QA / labeling, estimators, and the multistage router |
| [`src/utils/`](src/utils/) | Data loading, similarity, task hierarchy from KG, logging, finetune helpers |
| [`src/prompts/`](src/prompts/) | Prompt templates used when expanding the KG (`node_gen.py`) |
| [`configs/`](configs/) | `models.yaml` (API + pricing), optional `routing.yaml` / `tools.yaml` |

---

## 1. Benchmark data (preprocessing)

1. Follow **[`data_processing/DATA_PREPROCESSING.md`](data_processing/DATA_PREPROCESSING.md)**.
2. Short version: run `download_*.py` → `generate_dataset_splits.py` → **`construct_router_data.py`** so each task folder contains `train.csv`, `val.csv`, `test.csv` and per-model outputs such as `{model_name}.csv` (responses, tokens, latency, effectiveness).

Set `PYTHONPATH` to the **project root** when running scripts.

---

## 2. Knowledge graph (`src/components`)

The hierarchical KG (domains → subcategories → difficulty → optional preference) is built with LLM-assisted generation and stored as JSON (e.g. `kg_data.json`).

| Module | Role |
|--------|------|
| **`kg_contructor.py`** | Main builder: iteratively generates **domain**, **subcategory**, **difficulty**, and **preference** nodes; parses `<node begin>` … `<node end>` blocks; merges into nested JSON. Run stages via `__main__` (`generation_stage`, `kg_path`, `dir_path`, `skip_flag`). |
| **`kg_contructor_supp.py`** | Variant / supplementary graph construction (same pattern, different prompt set). |
| **`llm_provider.py`** | OpenAI-compatible client used by KG and annotators (configure `configs/models.yaml`). |

Prompts live under **`src/prompts/node_gen.py`** (imported by the constructors).

**Typical order:** domain nodes → subcategories per domain → difficulty per subcategory → (optional) preference. Point `kg_path` to your output directory and run one stage at a time until the JSON matches what downstream code expects.

---

## 3. Annotation & auxiliary labeling (`src/components`)

After the KG exists, you generate or collect labels that connect **queries** to **task types** and metrics.

| Module | Role |
|--------|------|
| **`task_annotator.py`** | Builds prompts from KG nodes (`annotate_data`, `generate_qa_for_task`) and calls the LLM to produce **QA pairs** per domain / subcategory / difficulty (structured markers in the reply). See also [`README_QA_Generation.md`](README_QA_Generation.md). |
| **`few_shot_t_labeler.py`** | **Hierarchical task-type classification** (multiple-choice style prompts per level) using the KG hierarchy + `DatasetGen`; can write `classification_results.json` for the router. |
| **`metric_estimator.py`** | LLM-based metric / judge utilities (uses `models.yaml`). |
| **`metric_statistics.py`** | Aggregates statistics over per-model CSVs under a data directory. |
| **`token_statistics.py`** | Token-usage style statistics (paths configurable in script). |

These artifacts (e.g. `generated_qa_*.json`, `classification_results.json`) are referenced by **`agenticrouter_multistage.py`** / **`estimator_model_multistage.py`** via constructor arguments such as `qa_data_path`, `query_task_type_path`, and `data_dir`.

---

## 4. Router & “training” logic (`src/components`)

| Module | Role |
|--------|------|
| **`estimator_model_multistage.py`** | **Estimator**: task-type distribution, similarity to KG / difficulty prompts, and per-model metric heads; optional **variational** tuning via `variational_prompt_tuner.py` / `variational_prompt_tuner_multiple.py`. |
| **`agenticrouter_multistage.py`** | **AgenticRouter**: loads benchmark CSVs (`DatasetGen`), KG + QA data, fits/evaluates the estimator (including few-shot / variational branches), routes test queries, and reports utilities and model mix. |

**Training** here means **fitting the estimator** (e.g. variational or few-shot prompt tuning on cost/effectiveness/latency signals), not full LLM pretraining. Entry point:

- **`python src/components/agenticrouter_multistage.py`** (edit `main()` for paths, `MODEL_NAMES`, `TASK_NAMES`, `few_shot_configs`, and `AgenticRouter(...)` arguments), or  
- **`./run_metricrouter.sh`** (sets `PYTHONPATH` and runs the multistage script).

You must align **local paths** (`kg_path`, `data_dir`, `model_result_dir`, `qa_data_path`, `query_task_type_path`, `prompt_tuning_save_dir`) with your machine.

---

## Installation

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

Configure **`configs/models.yaml`** with your API keys (never commit real secrets). Copy from **`configs/env.example.json`** if you use a JSON env template.

---

## License

MIT License
