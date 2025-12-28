import re
from typing import List, Iterable, Dict
import re


_FENCE_START = re.compile(r"^```\w*\s*$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    lines: List[str] = []
    for ln in str(text).splitlines():
        s = ln.strip()
        # hard stop markers (align with our executable prompts)
        if s.endswith("END_OF_PROGRAM") or s.endswith("END_OF_CODE") or s.endswith("END_OF_CASE"):
            break
        if _FENCE_START.match(s):
            continue
        if s in ("java", "python", "cpp", "c++"):
            continue
        if s == "```":
            continue
        lines.append(ln)
    out = "\n".join(lines).strip()
    if out.endswith("````"):
        out = out[: -len("````")]
    if out.endswith("```"):
        out = out[: -len("```")]
    return out.strip()


POLICY = (
    "Rules:\n"
    "1) Preserve the program semantics exactly; do not change input/output behavior.\n"
    "2) Strict stdin/stdout: read all input from STDIN, print only required answers to STDOUT, no extra text.\n"
    "3) Provide an executable entrypoint. For Java: public class Main { public static void main(String[] args){...} }. \n"
    "   For Python: wrap logic under if __name__ == \"__main__\": and read from sys.stdin.\n"
    "4) Prefer primitive types and iterative solutions; avoid unnecessary heavy libraries.\n"
    "5) For Java, prefer BufferedReader + StringTokenizer for input and StringBuilder for output; avoid repeated println in loops and BigInteger unless necessary.\n"
    "6) For Python, prefer sys.stdin.readline()/sys.stdout.write(); avoid extra prints and heavy per-iteration operations.\n"
    "7) Keep control-flow and update logic strictly equivalent to the source (no algorithm rewrites).\n"
    "8) For integer math, preserve integer division semantics (e.g., Python // -> integer division in the target) and avoid implicit float conversions.\n"
    "9) Carefully preserve loop bounds (start/end and < vs <=), avoiding off-by-one indexing bugs.\n"
    "10) Output only the code (no comments/markdown). End your answer with END_OF_PROGRAM.\n"
)


def build_minimal_prompt(src_lang: str, dst_lang: str, source_code: str, augmentation: str = "") -> str:
    base = (
        f"Translate the following {src_lang} code to {dst_lang}.\n"
        f"{POLICY}\n\n"
        f"<SourceCode>\n{source_code}\n</SourceCode>\n"
    )
    if augmentation:
        base += f"\n<ExtraTestInfo>\n{augmentation}\n</ExtraTestInfo>\n"
    # emphasize ending marker
    base += "\nReturn only the target code and finish with END_OF_PROGRAM.\n"
    return base


def _format_testcase_io(t: Dict, tgt_lang: str) -> str:
    """Format a single test case as Inputs/Outputs blocks."""
    inp = t.get("input", "").rstrip("\n")
    outs = t.get("output", []) or []
    out_text = "\n".join(str(x) for x in outs)
    lang_hint = "Java" if tgt_lang == "Java" else "Python"
    return (
        f"Inputs ({lang_hint}):\n{inp}\n"
        f"Outputs (STDOUT):\n{out_text}\n"
    )


def build_testcase_aug(tgt_lang: str, tests: Iterable[Dict], k: int = 2) -> str:
    acc: List[str] = []
    for i, t in enumerate(tests):
        if i >= max(0, k):
            break
        acc.append(_format_testcase_io(t, tgt_lang))
    if not acc:
        return ""
    return (
        "Given the following hidden test cases, ensure the program strictly matches I/O: \n"
        + "\n---\n".join(acc)
        + "\nUse only STDIN/STDOUT with no extra text."
    )


def build_exec_prompt_executable(src_lang: str, dst_lang: str, source_code: str, augmentation: str = "") -> str:
    """Prompt template focused on producing a complete executable program."""
    base = (
        f"Translate the following {src_lang} code into a COMPLETE, directly EXECUTABLE {dst_lang} program.\n"
        f"Requirements:\n"
        f"- Include the necessary entry point (e.g., main method for {dst_lang}).\n"
        f"- Read input and write output strictly via STDIN/STDOUT (no extra text).\n"
        f"- Do NOT add explanations/comments. Output ONLY the code.\n"
        f"- Do NOT wrap with triple backticks.\n\n"
        f"SOURCE CODE START\n{source_code}\nSOURCE CODE END\n"
    )
    # Target-specific guardrails (lightweight but effective)
    if dst_lang == "Java":
        base += (
            "\nNotes for Java target:\n"
            "- Keep algorithm and control flow identical to the source (no high-level rewrites).\n"
            "- Prefer primitive types (int/long/double); avoid BigInteger unless absolutely needed.\n"
            "- Use BufferedReader + StringTokenizer for input; aggregate output via StringBuilder and print once.\n"
            "- Respect integer division semantics (Python // -> Java integer division).\n"
            "- Arrays over collections if possible; avoid creating Streams inside loops.\n"
        )
    elif dst_lang == "Python":
        base += (
            "\nNotes for Python target:\n"
            "- Use sys.stdin.readline() and sys.stdout.write(); avoid extra prints.\n"
            "- Preserve integer division and modulo semantics; avoid float when not necessary.\n"
            "- Keep loops and boundary conditions identical to the source.\n"
        )
    if augmentation:
        base += f"\nGIVEN TEST CASE HINTS\n{augmentation}\nEND_OF_CASE\n"
    # Encourage the model to stop cleanly when possible
    base += "\nFinish your answer with END_OF_PROGRAM.\n"
    return base


# ---------- Lightweight analyses and guards (programmatic, no LLM) ----------

_LOOP_PATTERN_JAVA = re.compile(r"(for|while)\s*\(.*?\)\s*\{", re.DOTALL)
_LOOP_PATTERN_PY = re.compile(r"\bfor\b[\s\S]*?:")


def analyze_code(lang: str, code: str) -> Dict[str, int]:
    """Return a few counts of risky patterns for scoring/guarding.
    This is intentionally lightweight and fast.
    """
    s = code or ""
    out = {
        "len": len(s),
        "println": s.count("println("),
        "concat_plus": s.count("+ \""),
        "biginteger": s.count("BigInteger"),
        "scanner": s.count("Scanner"),
        "split": s.count("split("),
        "streams": s.count(".stream("),
        "arrays_sort": s.count("Arrays.sort("),
        "collections_sort": s.count("Collections.sort("),
        "print_py": s.count("print("),
        "stdin_py": s.count("sys.stdin.readline"),
    }
    # crude nesting estimator
    if lang == "Java":
        out["est_loops"] = len(_LOOP_PATTERN_JAVA.findall(s))
    else:
        out["est_loops"] = len(_LOOP_PATTERN_PY.findall(s))
    return out


def is_worse_than(lang: str, before: str, after: str) -> bool:
    """Heuristic guard: reject an after-version that clearly regresses.
    Criteria: explosive growth, more BigInteger, more println/concat, deeper loops.
    """
    a0, a1 = analyze_code(lang, before), analyze_code(lang, after)
    if a1["len"] > 1.5 * max(1, a0["len"]):
        return True
    if lang == "Java":
        if a1["biginteger"] > a0["biginteger"]:
            return True
        if a1["println"] > a0["println"] + 1:
            return True
        if a1["concat_plus"] > a0["concat_plus"] + 1:
            return True
        if a1["est_loops"] > a0["est_loops"] + 1:
            return True
    else:  # Python
        if a1["print_py"] > a0["print_py"] + 1:
            return True
        if a1["est_loops"] > a0["est_loops"] + 1:
            return True
    return False


# ---------- Strict prompt (only used on failures; low extra cost) ----------

def build_exec_prompt_strict(src_lang: str, dst_lang: str, source_code: str, augmentation: str = "") -> str:
    """A stricter variant to rescue borderline cases with stronger constraints.
    Used only when pre-exec gate fails, to avoid heavy API overhead.
    """
    rules = [
        "Output only code; no comments/markdown; end with END_OF_PROGRAM.",
        "Keep algorithm and control flow identical to the source (no rewrites).",
        "Read strictly from STDIN and write strictly to STDOUT (no extra text).",
    ]
    if dst_lang == "Java":
        rules += [
            "Provide: public class Main { public static void main(String[] args){...} }.",
            "Use BufferedReader + StringTokenizer for input; avoid Scanner.",
            "Aggregate outputs with StringBuilder and print once.",
            "Prefer primitive types (int/long); avoid BigInteger unless required by huge ranges.",
            "Do not create Streams inside loops; avoid printing inside loops.",
            "Respect integer division semantics.",
        ]
    else:
        rules += [
            "Wrap logic under if __name__ == \"__main__\":.",
            "Use sys.stdin.readline() / sys.stdout.write(); avoid extra prints.",
            "Prefer integer arithmetic (// when intended); avoid float unless necessary.",
        ]
    head = (
        f"Strictly translate {src_lang} to executable {dst_lang} program.\n"
        + "\n".join(f"- {x}" for x in rules)
        + "\n\n<Source>\n"
        + source_code
        + "\n</Source>\n"
    )
    if augmentation:
        head += f"\n<TestHints>\n{augmentation}\n</TestHints>\n"
    return head


# ---------- Conservative normalizers (safe, no-LLM) ----------

def _ensure_import(src: str, imp: str) -> str:
    if imp in src:
        return src
    # add after package if any, else at top
    lines = src.splitlines()
    insert_at = 0
    for i, ln in enumerate(lines[:5]):
        if ln.strip().startswith("package "):
            insert_at = i + 1
    lines.insert(insert_at, imp)
    return "\n".join(lines)


def normalize_java(code: str) -> str:
    s = code or ""
    out = s
    # If main exists but no class declaration at all, wrap with class Main
    if "public static void main" in out and "class Main" not in out and "class " not in out:
        out = "public class Main {\n" + out.strip() + "\n}\n"
    # If the code uses BufferedReader/StringTokenizer but lacks imports, add them
    if "BufferedReader" in out or "InputStreamReader" in out:
        out = _ensure_import(out, "import java.io.*;")
    if "StringTokenizer" in out:
        out = _ensure_import(out, "import java.util.*;")
    # If Arrays or Collections are used, ensure java.util import
    if "Arrays." in out or "Collections." in out:
        out = _ensure_import(out, "import java.util.*;")
    return out


_RX_INPUT_CALL = re.compile(r"\binput\(\)\b")


def normalize_python(code: str) -> str:
    s = code or ""
    out = s
    used_sys = False
    # Replace bare input() with sys.stdin.readline().strip()
    if _RX_INPUT_CALL.search(out):
        out = _RX_INPUT_CALL.sub("sys.stdin.readline().strip()", out)
        used_sys = True
    # Ensure import sys at top if used
    if used_sys and "import sys" not in out:
        out = "import sys\n" + out
    return out
