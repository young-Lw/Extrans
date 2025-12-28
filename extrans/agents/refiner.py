from typing import Optional
import sys

from ..openai_client import get_openai_client
from ..utils import strip_code_fences


REFINER_PROMPT = (
    "Improve the following program according to the bullet-point critique (and optional test hints). "
    "Keep the same algorithmic intent, fix correctness and I/O issues, ensure an executable entrypoint, and avoid adding comments or explanations. "
    "Output only the fixed code without markdown fences."
)


class RefinerAgent:
    def __init__(self, model: str, timeout_s: int = 0, retries: int = 0) -> None:
        self.model = model
        self.client = get_openai_client()
        self.timeout_s = timeout_s
        self.retries = max(0, retries)

    def refine(self, code: str, lang: str, critique: str, augmentation: Optional[str] = None) -> str:
        aug = f"\n\nTest hints:\n{augmentation}\n" if augmentation else "\n"
        msg = (
            f"{REFINER_PROMPT}\n\n"
            f"Language: {lang}{aug}"
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
                            temperature=0.5,
                            max_tokens=1400,
                            timeout=self.timeout_s,
                        )
                    except TypeError:
                        ret = self.client.with_options(timeout=self.timeout_s).chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": msg}],
                            temperature=0.5,
                            max_tokens=1400,
                        )
                else:
                    ret = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": msg}],
                        temperature=0.5,
                        max_tokens=1400,
                    )
                return strip_code_fences((ret.choices[0].message.content or "").strip())
            except Exception as e:
                attempt += 1
                print(f"[RefinerAgent] API error (attempt {attempt}/{self.retries+1}): {e}", file=sys.stderr)
                if attempt > self.retries:
                    return code
