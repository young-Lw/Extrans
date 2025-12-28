# EXtrans: Multi-Agent Code Translation (Custom Model)

This folder contains **EXtrans**, our own multi-agent executable code
translation model. It is evaluated on xCodeEval using the same
ExecEval-based pipeline as the baseline models, but EXtrans is *not*
part of the published baselines: it is our custom contribution.

The pipeline in this folder allows a tester to reproduce, on a fresh
machine:

- generation via EXtrans (multi-agent LLM orchestration)
- adaptation to the official xCodeEval dump format
- ExecEval-based evaluation via Docker
- pass@1 computation
- error analysis with logs

All paths in this README are relative to the folder `EXtrans` at the
repository root.

---

## 1. Folder structure

- `extrans/` – EXtrans Python package:
  - `cli.py` – command-line entry point.
  - `pipeline.py` – main orchestration logic.
  - `agents/translator.py`, `agents/critic.py`, `agents/refiner.py` – LLM
    agents used in the pipeline.
  - `openai_client.py` – OpenAI client helper (loads keys from `.env` or env).
  - `exec_eval_client.py` – client for ExecEval HTTP API.
  - `utils.py` – static scoring, normalization and analysis utilities.
- `xCodeEval-main/` – trimmed copy of xCodeEval with:
  - data under `xCodeEval-main/xCodeEval/`
  - evaluation script `evaluation/code_translation/eval_code_translation.py`
- `ExecEval/` – execution engine; includes its own `Dockerfile`.
- `scripts/` – shared evaluation helpers:
  - `xcodeeval_adapt_from_trans_jsonl.py`
  - `compute_pass1_xcodeeval.py`
  - `xcodeeval_error_analysis_v2.py`
- `dump_both/`, `logs_both/` – recommended locations for EXtrans dumps
  and logs when running the full pipeline.
- `requirements.txt` – minimal Python dependencies for EXtrans + xCodeEval.

---

## 2. Prerequisites

You need the following on a fresh machine:

- OS: Windows 10/11 with WSL2 (Ubuntu 20.04/22.04), or a native Linux
  machine with Docker installed.
- Docker Desktop (on Windows):
  - Enable `Use the WSL 2 based engine`.
  - In `Resources → WSL Integration`, enable your Ubuntu distribution.
- Python: Python 3.10+ inside WSL / Linux.
- Network access to OpenAI and a valid `OPENAI_API_KEY`.

All commands below are assumed to be run inside WSL / Linux.

---

## 3. Setup ExecEval (Docker)

You can reuse the same ExecEval image/container as for the UniTrans
baseline, or build it from this folder.

### 3.1 Build ExecEval image (first time only)

From the repository root:

```bash
cd EXtrans/ExecEval
docker build . -t exec-eval:1.0
```

If you have already built `exec-eval:1.0` elsewhere, this step can be
skipped.

### 3.2 Start ExecEval container (each time before running EXtrans)

```bash
docker run -d --name exec-eval -p 5000:5000 -e NUM_WORKERS=5 exec-eval:1.0
```

- `-p 5000:5000` exposes ExecEval on `http://127.0.0.1:5000`.
- `NUM_WORKERS=5` is a safe default for most laptops.

If the container already exists but is stopped, restart it with:

```bash
docker start exec-eval
```

### 3.3 Verify ExecEval service

```bash
curl http://127.0.0.1:5000/api/all_runtimes
```

If you see a JSON listing available runtimes, ExecEval is ready.

---

## 4. Python environment and dependencies

EXtrans runs as normal Python code (outside Docker) but depends on a few
libraries shared with xCodeEval.

From the repository root:

```bash
cd EXtrans

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains:

```text
python-dotenv>=1.0.0
openai>=1.0.0
tqdm>=4.66.0
jsonlines>=4.0.0
datasets>=2.19.0
requests>=2.31.0
```

---

## 5. Configure OpenAI API key

EXtrans uses the OpenAI API via `extrans/openai_client.py`. Configure
your key in one of the following ways.

### Option A: `.env` file (recommended)

In `EXtrans/.env`:

```bash
OPENAI_API_KEY=<your-openai-key>
```

Optional: to use a proxy/base URL, also set `OPENAI_API_BASE` or
`OPENAI_BASE_URL`.(Currently, the .env file in the document already contains a usable API. Since the API incurs costs, we hope to use it sparingly.)

### Option B: export in shell

```bash
export OPENAI_API_KEY=<your-openai-key>
```

---

## 6. Run EXtrans pipeline (end-to-end, with logs)

Make sure:

- The `exec-eval` container is running and responds on `127.0.0.1:5000`.
- The `EXtrans/.venv` environment is activated.

From the repository root:

```bash
cd EXtrans

source .venv/bin/activate

mkdir -p logs_both
export LIMIT=20              # 20 = first 20 samples; 0 = full test split
export TRANSLATOR=gpt-4o
export CRITIC=gpt-4o
export REFINER=gpt-4o
export DUMP_FOLDER="$PWD/dump_both"
mkdir -p "$DUMP_FOLDER"
```

### 6.1 Generation (EXtrans multi-agent translation)

Java → Python:

```bash
python3 -m extrans.cli \
  --xcode_dir xCodeEval-main/xCodeEval \
  --split test \
  --src_lang_cluster Java \
  --tgt_lang_cluster Python \
  --translator_model "$TRANSLATOR" \
  --critic_model "$CRITIC" \
  --refiner_model "$REFINER" \
  --nsample 1 \
  --refine_rounds 1 \
  --limit "$LIMIT" \
  --out_path outputs_j2p.jsonl \
  > logs_both/1_generate_j2p.log 2>&1
```

Python → Java:

```bash
python3 -m extrans.cli \
  --xcode_dir xCodeEval-main/xCodeEval \
  --split test \
  --src_lang_cluster Python \
  --tgt_lang_cluster Java \
  --translator_model "$TRANSLATOR" \
  --critic_model "$CRITIC" \
  --refiner_model "$REFINER" \
  --nsample 1 \
  --refine_rounds 1 \
  --limit "$LIMIT" \
  --out_path outputs_p2j.jsonl \
  > logs_both/1_generate_p2j.log 2>&1
```

Outputs:

- EXtrans translation results are written to `outputs_j2p.jsonl` and
  `outputs_p2j.jsonl`.
- Detailed generation progress is logged in `logs_both/1_generate_*.log`.

You can tail progress, for example:

```bash
tail -f logs_both/1_generate_j2p.log
```

### 6.2 Adaptation to xCodeEval format

Java → Python:

```bash
python3 scripts/xcodeeval_adapt_from_trans_jsonl.py \
  --src_jsonl outputs_j2p.jsonl \
  --xcode_dir xCodeEval-main/xCodeEval \
  --src_lang_cluster Java \
  --tgt_lang_cluster Python \
  > logs_both/2_adapt_j2p.log 2>&1
```

Python → Java:

```bash
python3 scripts/xcodeeval_adapt_from_trans_jsonl.py \
  --src_jsonl outputs_p2j.jsonl \
  --xcode_dir xCodeEval-main/xCodeEval \
  --src_lang_cluster Python \
  --tgt_lang_cluster Java \
  > logs_both/2_adapt_p2j.log 2>&1
```

Adapted data is written into the xCodeEval directory structure under
`xCodeEval-main/xCodeEval/...` for subsequent evaluation.

### 6.3 Optional: quick check of dumps

```bash
ls "$DUMP_FOLDER/oai/code_translation_n_sample_20/compact_small" | head
```

If you see `*.jsonl` files, the dumps were written correctly.

### 6.4 Evaluation with xCodeEval and ExecEval

```bash
python3 xCodeEval-main/evaluation/code_translation/eval_code_translation.py \
  > logs_both/3_eval_exec.log 2>&1
```

This step runs the official xCodeEval evaluation, which calls ExecEval to
execute translated programs against unit tests and records detailed
execution results under `$DUMP_FOLDER`.

Note: On some machines (especially macOS or low-resource systems), this
step can be unstable due to resource limits. If evaluation crashes, you
can:

1. Restart the ExecEval container in Docker Desktop or via:
   ```bash
   docker restart exec-eval
   ```
2. Re-run the evaluation command above.

Even if this step of the evaluation crashes in the middle, it doesn't matter. The subsequent steps can still run, but a complete result cannot be obtained.

### 6.5 Compute pass@1

```bash
python3 scripts/compute_pass1_xcodeeval.py | tee logs_both/4_pass1.log
```

The script prints pass@1 scores for Java/Python targets and stores them
in `logs_both/4_pass1.log`.

### 6.6 Error analysis (English taxonomy)

```bash
python3 scripts/xcodeeval_error_analysis_v2.py \
  --dump_folder "$DUMP_FOLDER" \
  --split compact_small \
  --out_dir logs_both \
  | tee logs_both/5_error_analysis_v2.log
```

Artifacts:

- `logs_both/error_analysis_v2_Java_17.json`
- `logs_both/error_analysis_v2_PyPy_3.json`

These summarize error categories for each runtime.

---

## 7. Relation to course requirements

- This directory demonstrates our **custom EXtrans model**, not a
  published baseline. It uses the evaluation pipeline while
  introducing a multi-agent design (translator/critic/refiner).
- Evaluation is performed through the official xCodeEval benchmark and
  ExecEval container, with full logs and dumps as evidence of testing.
- The Dockerised ExecEval backend ensures untrusted code is executed in
  an isolated environment, while EXtrans remains easy to inspect and
  modify as plain Python code.

---

## 8. Notes and troubleshooting

- You can reuse the same `exec-eval` container across UniTrans and
  EXtrans runs; there is no need to rebuild for each model.
- If evaluation fails with connection errors, check:
  - `docker ps` to confirm the ExecEval container is running.
  - Port mapping `-p 5000:5000`.
- If `openai` cannot be imported or API calls fail:
  - Confirm `.venv` is activated.
  - Confirm `OPENAI_API_KEY` is set (either in `.env` or via `export`).
- Start with `LIMIT=20` to validate the pipeline before running the
  full test split with `LIMIT=0`.

