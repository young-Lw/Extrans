import os
import json
import argparse
from typing import Dict, Any, Iterable, List


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_unittest_db(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_by_code_uid(path: str) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for obj in read_jsonl(path):
        cu = obj.get("code_uid")
        if cu:
            idx[cu] = obj
    return idx


def main():
    ap = argparse.ArgumentParser("Adapt our trans.jsonl into xCodeEval official dump structure")
    ap.add_argument("--src_jsonl", required=True, help="Our translation output jsonl (contains id=code_uid and target field)")
    ap.add_argument("--xcode_dir", required=True, help="Path to xCodeEval directory (contains code_translation and unittest_db.json)")
    ap.add_argument("--src_lang_cluster", required=True, choices=["Java", "Python"], help="Source language cluster (original code)")
    ap.add_argument("--tgt_lang_cluster", required=True, choices=["Java", "Python"], help="Target language cluster (translated code)")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of items (0=no cap)")
    args = ap.parse_args()

    # Prepare output dir (official evaluator expects this env var)
    dump_folder = os.getenv("DUMP_FOLDER", "").strip()
    if not dump_folder:
        raise SystemExit("DUMP_FOLDER env not set; set it to an absolute path per xCodeEval conventions")
    out_split_dir = os.path.join(dump_folder, "oai", "code_translation_n_sample_20", "compact_small")
    os.makedirs(out_split_dir, exist_ok=True)

    # Load xCodeEval test split for source language; we use it to recover source_data by code_uid
    ct_test_path = os.path.join(args.xcode_dir, "code_translation", "test", f"{args.src_lang_cluster}.jsonl")
    if not os.path.exists(ct_test_path):
        raise SystemExit(f"Not found: {ct_test_path}")
    src_index = index_by_code_uid(ct_test_path)

    # Load unit test DB to attach hidden_unit_tests by src_uid
    ut_path = os.path.join(args.xcode_dir, "unittest_db.json")
    if not os.path.exists(ut_path):
        raise SystemExit(f"Not found unittest_db.json at: {ut_path}")
    ut_db = load_unittest_db(ut_path)

    # Read our translations and adapt
    count = 0
    written = 0
    for obj in read_jsonl(args.src_jsonl):
        count += 1
        if 0 < args.limit < count:
            break
        code_uid = obj.get("id")
        if not code_uid:
            continue
        src_row = src_index.get(code_uid)
        if not src_row:
            # No matching original row; skip
            continue
        src_uid = src_row.get("src_uid")
        if not src_uid:
            continue
        uts = ut_db.get(src_uid)
        if not uts:
            continue

        # Extract candidates from our translation jsonl
        # Determine target field name by tgt_lang_cluster
        tgt_field = "python" if args.tgt_lang_cluster == "Python" else "java"
        candidates: List[str] = obj.get(tgt_field) or []
        # Build oai_response
        oai_choices = [{"message": {"content": c}} for c in candidates]
        # Compose export
        export = {
            "oai_response": {"choices": oai_choices, "prompt": ""},
            "source_data": {
                **src_row,
                "target_lang": args.tgt_lang_cluster,
                "hidden_unit_tests": json.dumps(uts),
            },
        }
        # Filename pattern similar to official dumps (index_temperature_language)
        fname = f"{written}_0.8_{args.src_lang_cluster}--{args.tgt_lang_cluster}.json"
        with open(os.path.join(out_split_dir, fname), "w", encoding="utf-8") as fw:
            json.dump(export, fw, ensure_ascii=False, indent=2)
        written += 1

    print(f"Adapted {written} samples -> {out_split_dir}")


if __name__ == "__main__":
    main()

