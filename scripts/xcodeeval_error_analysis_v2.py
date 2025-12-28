import os
import json
import argparse
from collections import Counter
from typing import Dict, Any, List
import re


def _extract_text(ut: Dict[str, Any]) -> str:
    """Best-effort extract any diagnostic text from a unit-test result entry."""
    parts: List[str] = []
    for k in ("result", "message", "stderr", "stdout", "error", "detail"):
        v = ut.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
        elif isinstance(v, list):
            # some fields might be list[str]
            parts.extend([str(x) for x in v if x])
    # also try 'output' list
    out_list = ut.get("output")
    if isinstance(out_list, list):
        parts.extend([str(x) for x in out_list if x])
    return "\n".join(parts)[:2000]  # cap


NUM_PATTERN = re.compile(r"^[+-]?(?:\d+)(?:\.\d+)?$")


def _is_numeric_only(text: str) -> bool:
    if text is None:
        return False
    s = str(text).strip()
    if not s:
        return False
    toks = re.split(r"\s+", s)
    return all(NUM_PATTERN.match(t) is not None for t in toks)


def _io_format_mismatch(exp: str, got: str) -> bool:
    """Heuristic: if expected is numeric-only but got contains non-numeric tokens, treat as format mismatch."""
    if exp is None or got is None:
        return False
    exp_is_num = _is_numeric_only(exp)
    if not exp_is_num:
        return False
    # allow digits, sign, dots and whitespace in numeric output; extra letters hint format/prose
    cleaned = re.sub(r"[\s\d+\-\.]+", "", str(got))
    return len(cleaned) > 0


def _classify(sample: Dict[str, Any]) -> str:
    """Map ExecEval outcomes to teacher-specified categories.

    Categories (corresponding to the original Chinese labels):
      - Compilation Error
      - Semantic / Logic Error (merged)
      - I/O Format Error
      - Type Error
      - Index Error
      - Performance Error
      - Syntax Error
      - Logic Error (merged into Semantic / Logic Error)
    We only return an error label; PASSED samples return "PASSED" and will be filtered out upstream.
    """
    ch_list = sample.get("unit_test_results", [])
    if not ch_list:
        return "OTHER"
    # We analyze the first choice only (nsample=1 in our pipeline)
    uts = ch_list[0] if isinstance(ch_list[0], list) else []
    if not uts:
        return "OTHER"

    # Gather outcomes and any diagnostic text
    outcomes = [str(ut.get("exec_outcome", "")).upper() for ut in uts]
    texts = "\n".join(_extract_text(ut) for ut in uts)
    t = texts.lower()

    # 0) All passed -> mark PASSED so caller can exclude
    if all(o == "PASSED" for o in outcomes if o):
        return "PASSED"

    # 1) Compilation vs syntax
    if any(o == "COMPILATION_ERROR" for o in outcomes):
        # rough syntax patterns
        syntax_markers = (
            "syntax error",
            "illegal start of",
            "unclosed string",
            "not a statement",
            "expected ';'",
            "mismatched input",
        )
        if any(m in t for m in syntax_markers):
            return "Syntax Error"
        return "Compilation Error"

    # 2) Performance
    if any(o in ("TIME_LIMIT_EXCEEDED", "MEMORY_LIMIT_EXCEEDED") for o in outcomes):
        return "Performance Error"

    # 3) Runtime -> finer buckets
    if any(o == "RUNTIME_ERROR" for o in outcomes):
        # Type-related
        type_markers = (
            "typeerror",
            "valueerror",
            "numberformatexception",
            "classcastexception",
            "illegalargumentexception",
            "cannot convert",
            "incompatible types",
            "invalid literal for int",
        )
        if any(m in t for m in type_markers):
            return "Type Error"
        # Index-related
        index_markers = (
            "indexerror",
            "arrayindexoutofboundsexception",
            "stringindexoutofboundsexception",
            "index out of range",
            "out of bounds",
        )
        if any(m in t for m in index_markers):
            return "Index Error"
        # I/O format / parsing
        io_markers = (
            "inputmismatchexception",
            "scanf",
            "format",
            "parseint",
            "parsedouble",
            "no such element exception",
            "nosuchelementexception",
            "for input string",
            "eoferror",
        )
        if any(m in t for m in io_markers):
            return "I/O Format Error"
        # runtime but no specific pattern: likely implementation-level
        logic_markers = (
            "nullpointerexception",
            "arithmeticexception",
            "/ by zero",
            "zerodivisionerror",
            "nameerror",
            "attributeerror",
            "keyerror",
            "filenotfounderror",
            "modulenotfounderror",
            "illegalstateexception",
        )
        if any(m in t for m in logic_markers):
            return "Logic Error"
        # default for runtime errors: implementation-level
        return "Logic Error"

    # 4) Wrong answer -> split into Semantic Understanding Error vs Logic Error vs I/O Format Error
    if all(o in ("PASSED", "WRONG_ANSWER") for o in outcomes) and any(o == "WRONG_ANSWER" for o in outcomes):
        # Inspect the first WRONG_ANSWER case
        for ut in uts:
            if str(ut.get("exec_outcome", "")).upper() == "WRONG_ANSWER":
                exp_list = ut.get("output") or []
                exp0 = exp_list[0] if exp_list else None
                got = ut.get("result")
                if _io_format_mismatch(exp0, got):
                    return "I/O Format Error"
                # If both expected and actual look numeric, treat as Logic Error; else Semantic Understanding Error
                if _is_numeric_only(exp0) and _is_numeric_only(str(got or "").strip()):
                    return "Logic Error"
                return "Semantic Understanding Error"

    # Fallback
    return "OTHER"


def analyze(eval_path: str) -> Counter:
    cnt = Counter()
    if not os.path.exists(eval_path):
        return cnt
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = _classify(obj)
            if label == "PASSED":
                continue
            # Only count errors (exclude PASSED explicitly)
            if label != "OTHER" or True:
                cnt[label] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser("Teacher-specified error analysis for xCodeEval ExecEval outputs")
    ap.add_argument("--dump_folder", default=os.environ.get("DUMP_FOLDER", ""))
    ap.add_argument("--split", default="compact_small")
    ap.add_argument("--out_dir", default="log_details")
    args = ap.parse_args()

    if not args.dump_folder:
        raise SystemExit("DUMP_FOLDER not set; provide --dump_folder or export env var")

    eval_dir = os.path.join(
        args.dump_folder,
        "oai",
        "code_translation_n_sample_20",
        args.split,
        f"eval_code_translation_{args.split}_execeval",
    )
    os.makedirs(args.out_dir, exist_ok=True)

    targets = ["Java 17", "PyPy 3"]
    for target in targets:
        path = os.path.join(eval_dir, f"{target}.jsonl")
        cnt = analyze(path)
        # Remove PASSED entirely; only report errors
        if "PASSED" in cnt:
            del cnt["PASSED"]
        # Merge OTHER into Semantic Understanding Error (higher-level misinterpretation)
        if "OTHER" in cnt and cnt["OTHER"] > 0:
            cnt["Semantic Understanding Error"] = cnt.get("Semantic Understanding Error", 0) + cnt["OTHER"]
            del cnt["OTHER"]
        total_err = sum(cnt.values())
        print(f"== {target} ==")
        if total_err == 0:
            print("No errors found or file missing.")
            continue
        for k, v in cnt.most_common():
            pct = 100.0 * v / total_err
            print(f"{k:18s} : {v:5d} ({pct:6.2f}%)")
        print(f"TOTAL_ERRORS: {total_err}")

        summary = {
            "target": target,
            "total_errors": total_err,
            "counts": dict(cnt),
            "percentages": {k: (100.0 * v / total_err) for k, v in cnt.items()},
            "eval_path": path,
        }
        out_name = f"error_analysis_v2_{target.replace(' ', '_')}.json"
        with open(os.path.join(args.out_dir, out_name), "w", encoding="utf-8") as fw:
            json.dump(summary, fw, ensure_ascii=False, indent=2)
        print(f"Summary written -> {os.path.join(args.out_dir, out_name)}")


if __name__ == "__main__":
    main()
