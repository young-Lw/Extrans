from typing import List, Optional
import sys

from ..openai_client import get_openai_client
from ..utils import (
    strip_code_fences,
    build_minimal_prompt,
    build_exec_prompt_executable,
    build_exec_prompt_strict,
)


class TranslatorAgent:
    def __init__(self, model: str, timeout_s: int = 0, retries: int = 0) -> None:
        self.model = model
        self.client = get_openai_client()
        self.timeout_s = timeout_s
        self.retries = max(0, retries)

    def translate(
        self,
        src_lang: str,
        dst_lang: str,
        source_code: str,
        nsample: int = 1,
        augmentation: Optional[str] = None,
    ) -> List[str]:
        # Prompt choice:
        # - For Python->Java, fall back to minimal EXtrans-style prompt (your original strong baseline).
        # - Otherwise, use executable-focused prompt.
        if src_lang == "Python" and dst_lang == "Java":
            prompt = build_minimal_prompt(src_lang, dst_lang, source_code, augmentation or "")
        else:
            prompt = build_exec_prompt_executable(src_lang, dst_lang, source_code, augmentation or "")
        # direction-aware temperature: prefer steadier outputs
        # - Python target: 0.6  (reduce variance)
        # - Java target:   0.5  (stricter structure)
        temp = 0.4 if (dst_lang == "Java") else 0.6
        attempt = 0
        while True:
            try:
                # prefer per-call timeout if supported; fallback to with_options
                if self.timeout_s and self.timeout_s > 0:
                    try:
                        ret = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            n=nsample,
                            temperature=temp,
                            top_p=0.95,
                            max_tokens=1200,
                            timeout=self.timeout_s,
                        )
                    except TypeError:
                        ret = self.client.with_options(timeout=self.timeout_s).chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            n=nsample,
                            temperature=temp,
                            top_p=0.95,
                            max_tokens=1200,
                        )
                else:
                    ret = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        n=nsample,
                        temperature=temp,
                        top_p=0.95,
                        max_tokens=1200,
                    )
                outs: List[str] = []
                for ch in ret.choices:
                    code = strip_code_fences((ch.message.content or "").strip())
                    if code:
                        outs.append(code)
                return outs
            except Exception as e:
                attempt += 1
                print(f"[TranslatorAgent] API error (attempt {attempt}/{self.retries+1}): {e}", file=sys.stderr)
                if attempt > self.retries:
                    return []

    def translate_strict(
        self,
        src_lang: str,
        dst_lang: str,
        source_code: str,
        augmentation: Optional[str] = None,
    ) -> List[str]:
        prompt = build_exec_prompt_strict(src_lang, dst_lang, source_code, augmentation or "")
        # keep n=1 for strict retry to limit cost
        temp = 0.4 if (dst_lang == "Java") else 0.6
        attempt = 0
        while True:
            try:
                ret = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    temperature=temp,
                    top_p=0.95,
                    max_tokens=1200,
                )
                code = strip_code_fences((ret.choices[0].message.content or "").strip())
                return [code] if code else []
            except Exception as e:
                attempt += 1
                print(f"[TranslatorAgent] API error (attempt {attempt}/{self.retries+1}): {e}", file=sys.stderr)
                if attempt > self.retries:
                    return []
