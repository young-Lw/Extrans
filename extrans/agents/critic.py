from typing import Optional
import sys
from ..openai_client import get_openai_client


CRITIC_PROMPT = (
    "You are a code reviewer for executable competitive-programming style solutions. "
    "Given a target-language program, list concrete issues in bullet points: I/O usage, entrypoint, edge-cases, off-by-one, integer overflow, performance (O(...)), library availability, and style that breaks execution. "
    "Be terse; do not rewrite code. Also flag typical performance anti-patterns: loop-internal printing/sorting/string concatenation; heavy BigInteger usage in Java unless necessary; repeated Scanner.nextLine()+split chains; Streams created inside large loops; deep recursion without pruning."
)

def _heuristic_critique(lang: str, code: str) -> str:
    bullets = []
    L = (code or "")
    if lang == "Java":
        if "public static void main" not in L:
            bullets.append("Add executable entrypoint: public static void main(String[] args)")
        if "class Main" not in L:
            bullets.append("Ensure class name Main with public access")
        if "Scanner" in L and "BufferedReader" not in L:
            bullets.append("Prefer BufferedReader + StringTokenizer over Scanner for input speed")
        if "println(" in L:
            bullets.append("Avoid frequent System.out.println in loops; buffer with StringBuilder and print once")
        if "/" in L and "//" in L:
            bullets.append("Check all divisions: ensure integer division semantics where required (no unintended double)")
        if "for (" in L and ("<=" in L or "< " in L):
            bullets.append("Review loop bounds (< vs <=) to avoid off-by-one and index out of range")
        if "BigInteger" in L:
            bullets.append("Avoid BigInteger unless necessary; use primitive types where possible")
        if L.count("for (") + L.count("while (") >= 3:
            bullets.append("Reduce deep nested loops; avoid per‑iteration sorting or scans")
        if "+ \"" in L:
            bullets.append("Avoid string concatenation in loops; use StringBuilder")
    else:  # Python
        if "if __name__ == \"__main__\"" not in L:
            bullets.append("Wrap execution under if __name__ == \"__main__\":")
        if "import sys" not in L:
            bullets.append("Use sys.stdin.readline() for input; avoid extra prints")
        if L.count("print(") > 3:
            bullets.append("Avoid printing inside large loops; aggregate output")
        if "//" in L:
            bullets.append("Use integer division '//' where needed; avoid float imprecision")
    if not bullets:
        bullets.append("Ensure strict STDIN/STDOUT only; no extra text; handle edge cases and indices correctly")
    return "\n- " + "\n- ".join(bullets)


class CriticAgent:
    def __init__(self, model: str, timeout_s: int = 0, retries: int = 0) -> None:
        self.model = model
        self.client = get_openai_client()
        self.timeout_s = timeout_s
        self.retries = max(0, retries)

    def critique(self, code: str, lang: str) -> str:
        msg = (
            f"{CRITIC_PROMPT}\n\n"
            f"Language: {lang}\n"
            f"<Program>\n{code}\n</Program>\n"
        )
        attempt = 0
        while True:
            try:
                if self.timeout_s and self.timeout_s > 0:
                    try:
                        ret = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": msg}],
                            temperature=0.2,
                            max_tokens=500,
                            timeout=self.timeout_s,
                        )
                    except TypeError:
                        ret = self.client.with_options(timeout=self.timeout_s).chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": msg}],
                            temperature=0.2,
                            max_tokens=500,
                        )
                else:
                    ret = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": msg}],
                        temperature=0.2,
                        max_tokens=500,
                    )
                return (ret.choices[0].message.content or "").strip()
            except Exception as e:
                attempt += 1
                print(f"[CriticAgent] API error (attempt {attempt}/{self.retries+1}): {e}", file=sys.stderr)
                if attempt > self.retries:
                    # Fall back to deterministic heuristic critique to keep refine effective
                    return _heuristic_critique(lang, code)
