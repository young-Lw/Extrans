#!/usr/bin/env bash
set -euo pipefail

# Example wrapper to run EXtrans and the downstream xCodeEval evaluation steps.
# Adjust models/paths as needed. Assumes this script lives in the EXtrans
# folder at the repository root.

ROOT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR="$ROOT_DIR/outputs"
LOG_DIR="$ROOT_DIR/logs"
DUMP_DIR="$ROOT_DIR/dump"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$DUMP_DIR"
export DUMP_FOLDER="$DUMP_DIR"

# 1) Generate translations (Java -> Python)
python3 -m extrans.cli \
  --xcode_dir "$ROOT_DIR/xCodeEval-main/xCodeEval" \
  --split test \
  --src_lang_cluster Java \
  --tgt_lang_cluster Python \
  --translator_model gpt-4o \
  --critic_model gpt-4o-mini \
  --refiner_model gpt-4-turbo \
  --nsample 1 \
  --refine_rounds 1 \
  --limit 0 \
  --out_path "$OUT_DIR/testable_java_python_extrans.jsonl" \
  > "$LOG_DIR/1_generate_j2p.log" 2>&1

# 2) Adapt to official dump
python3 "$ROOT_DIR/scripts/xcodeeval_adapt_from_trans_jsonl.py" \
  --src_jsonl "$OUT_DIR/testable_java_python_extrans.jsonl" \
  --xcode_dir "$ROOT_DIR/xCodeEval-main/xCodeEval" \
  --src_lang_cluster Java \
  --tgt_lang_cluster Python \
  > "$LOG_DIR/2_adapt_j2p.log" 2>&1

# 3) Evaluate via xCodeEval + ExecEval (server must be running at :5000)
python3 "$ROOT_DIR/xCodeEval-main/evaluation/code_translation/eval_code_translation.py" \
  > "$LOG_DIR/3_eval_exec.log" 2>&1

# 4) Compute pass@1
python3 "$ROOT_DIR/scripts/compute_pass1_xcodeeval.py" | tee "$LOG_DIR/4_pass1.log"

# 5) Error analysis v2 (English taxonomy)
python3 "$ROOT_DIR/scripts/xcodeeval_error_analysis_v2.py" \
  --dump_folder "$DUMP_FOLDER" \
  --split compact_small \
  --out_dir "$LOG_DIR" \
  | tee "$LOG_DIR/5_error_analysis_v2.log"

echo "Done. Outputs: $OUT_DIR | Dump: $DUMP_DIR | Logs: $LOG_DIR"
