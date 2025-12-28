import os
import json
import sys


def compute_pass1_for(path: str, label: str):
    t = c = 0
    if not os.path.exists(path):
        print(f"{label} file not found: {path}")
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            uts = o.get("unit_test_results", [])
            if not uts:
                continue
            t += 1
            c += 1 if all(x.get("exec_outcome") == "PASSED" for x in uts[0]) else 0
    score = (c / t) if t else float("nan")
    print(f"{label} pass@1 = {score:.4f} ({c}/{t})")


def main():
    base = os.path.join(
        os.environ.get("DUMP_FOLDER", ""),
        "oai",
        "code_translation_n_sample_20",
        "compact_small",
        "eval_code_translation_compact_small_execeval",
    )
    compute_pass1_for(os.path.join(base, "Java 17.jsonl"), "Java target")
    compute_pass1_for(os.path.join(base, "PyPy 3.jsonl"), "Python target")


if __name__ == "__main__":
    main()

