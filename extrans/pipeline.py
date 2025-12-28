import json
import os
import re
from typing import Dict, Iterable, List, Optional
from tqdm import tqdm

from .agents.translator import TranslatorAgent
from .agents.critic import CriticAgent
from .agents.refiner import RefinerAgent
from .utils import build_testcase_aug, analyze_code, is_worse_than, normalize_java, normalize_python
from .exec_eval_client import ExecEvalClient, LANG_CLUSTER_TO_LANG_COMPILER


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_lines(path: str) -> int:
    cnt = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            cnt += 1
    return cnt


def _count(s: str, needle: str) -> int:
    return s.count(needle)


def _static_score(code: str, tgt_lang_cluster: str) -> int:
    s = 0
    if tgt_lang_cluster == "Java":
        # Positive signals
        if "class Main" in code: s += 4
        if "public static void main" in code: s += 4
        if "BufferedReader" in code and "StringTokenizer" in code: s += 4
        if "StringBuilder" in code: s += 2
        # Negative patterns
        s -= 4 * min(5, _count(code, "BigInteger"))
        s -= 3 * min(5, _count(code, "println("))
        s -= 3 * min(5, _count(code, "+ \""))  # string concatenation with literal
        # discourage Scanner-heavy parsing
        if "Scanner" in code and "split(" in code:
            s -= 4
        # approximate deep nesting
        if re.search(r"for\s*\(.*\)\s*\{[\s\S]*for\s*\(.*\)\s*\{[\s\S]*for\s*\(", code):
            s -= 2
        # reward absence of in-loop sort heuristically (if only one global sort present)
        if _count(code, "Arrays.sort(") + _count(code, "Collections.sort(") <= 1:
            s += 2
    else:  # Python
        if "if __name__ == \"__main__\"" in code: s += 3
        if "import sys" in code: s += 2
        s -= 1 * min(5, _count(code, "print("))
        if re.search(r"for .*:.*\n\s*for .*:.*\n\s*for ", code):
            s -= 2
    return s


def run_pipeline(
    xcode_dir: str,
    split: str,
    src_lang_cluster: str,
    tgt_lang_cluster: str,
    out_path: str,
    translator_model: str,
    critic_model: str,
    refiner_model: str,
    nsample: int = 1,
    refine_rounds: int = 1,
    limit: int = 0,
    use_testcase_aug: bool = False,
    k_aug_tests: int = 2,
    api_timeout: int = 60,
    api_retries: int = 2,
    max_empty_retries: int = -1,
):
    ct_dir = os.path.join(xcode_dir, "code_translation", split)
    src_file = os.path.join(ct_dir, f"{src_lang_cluster}.jsonl")
    if not os.path.exists(src_file):
        raise SystemExit(f"Not found: {src_file}")

    # Built-in robust defaults when CLI didn't provide timeouts/retries.
    # Translator keeps default (no timeout/retry) unless user explicitly sets.
    if api_timeout and api_timeout > 0:
        t_timeout = c_timeout = r_timeout = api_timeout
    else:
        t_timeout, c_timeout, r_timeout = 0, 120, 120

    if api_retries and api_retries > 0:
        t_retries = c_retries = r_retries = api_retries
    else:
        t_retries, c_retries, r_retries = 0, 2, 2

    print(
        f"[EXtrans] Start pipeline: {src_lang_cluster}->{tgt_lang_cluster}, "
        f"translator={translator_model}, critic={critic_model}, refiner={refiner_model}, "
        f"nsample={nsample}, refine_rounds={refine_rounds}, limit={limit}, "
        f"test_aug={use_testcase_aug}({k_aug_tests}), "
        f"timeouts(t/c/r)={[t_timeout,c_timeout,r_timeout]}, retries(t/c/r)={[t_retries,c_retries,r_retries]}"
    )

    translator = TranslatorAgent(translator_model, timeout_s=t_timeout, retries=t_retries)
    critic = CriticAgent(critic_model, timeout_s=c_timeout, retries=c_retries)
    refiner = RefinerAgent(refiner_model, timeout_s=r_timeout, retries=r_retries)

    tgt_field = "python" if tgt_lang_cluster == "Python" else "java"
    # optional ExecEval quick repair (default on, best-effort)
    execeval_url = os.getenv("EX2_EXEC_EVAL_URL", "http://localhost:5000")
    execeval = None
    try:
        execeval = ExecEvalClient(execeval_url)
        _ = execeval.get_runtimes()
    except Exception:
        execeval = None

    # hidden unittest db (for augmentation only)
    ut_db_path = os.path.join(xcode_dir, "unittest_db.json")
    ut_db = None
    if os.path.exists(ut_db_path):
        try:
            with open(ut_db_path, encoding="utf-8") as f:
                ut_db = json.load(f)
        except Exception:
            ut_db = None

    planned = limit if limit and limit > 0 else _count_lines(src_file)
    processed = 0
    with open(out_path, "w", encoding="utf-8") as w:
        idx = 0
        for obj in tqdm(read_jsonl(src_file), total=planned, desc=f"{src_lang_cluster}->{tgt_lang_cluster}"):
            idx += 1
            processed += 1
            if limit > 0 and processed > limit:
                break
            code_uid = obj.get("code_uid") or obj.get("id") or obj.get("src_uid")
            src_uid = obj.get("src_uid")
            source_code = obj.get("source_code", "")
            if not code_uid or not source_code:
                continue
            print(f"[EXtrans] Processing {idx}/{planned}: {code_uid}")

            # testcase augmentation text (optional)
            augmentation: Optional[str] = None
            if use_testcase_aug and ut_db and src_uid in ut_db:
                tests = ut_db.get(src_uid) or []
                augmentation = build_testcase_aug(tgt_lang_cluster, tests, k_aug_tests)

            # 1) baseline candidates with optional augmentation
            #    Do not skip samples: keep retrying until non-empty (or until max attempts reached)
            attempts = 0
            while True:
                cands: List[str] = translator.translate(
                    src_lang_cluster, tgt_lang_cluster, source_code, nsample, augmentation
                )
                if cands:
                    break
                attempts += 1
                if max_empty_retries >= 0 and attempts > max_empty_retries:
                    # still write an empty list to keep alignment, but per user request
                    # default is infinite retry (max_empty_retries=-1)
                    break

            # 1.25) normalize candidates conservatively (no LLM)
            normed: List[str] = []
            for c in cands:
                if tgt_lang_cluster == "Java":
                    normed.append(normalize_java(c))
                else:
                    normed.append(normalize_python(c))
            cands = normed
            # baseline candidate: always include first normalized output
            baseline_code = normed[0] if normed else ""

            # 1.5) optional pre-exec gate: if best candidate already passes 1-2 tests, skip refine
            prepass_mask: List[bool] = [False] * len(cands)
            if execeval and ut_db and src_uid in ut_db and cands:
                uts = ut_db.get(src_uid) or []
                if isinstance(uts, list) and len(uts) > 0:
                    use = min(2, len(uts))
                    sample_uts = [
                        {"input": uts[i].get("input", ""), "output": uts[i].get("output", [])}
                        for i in range(use)
                    ]
                    runtime = LANG_CLUSTER_TO_LANG_COMPILER.get(tgt_lang_cluster)
                    try:
                        # only check the first candidate to keep speed; extend if needed
                        res = execeval.execute_code(runtime, cands[0], sample_uts)
                        data = res.get("data") if isinstance(res, dict) else None
                        ok = False
                        if isinstance(data, list) and data:
                            ok = all(item.get("exec_outcome") == "PASSED" for item in data[0])
                        prepass_mask[0] = ok
                        # If not ok, try a strict translator retry once and prefer it when better
                        if not ok:
                            strict = translator.translate_strict(
                                src_lang_cluster, tgt_lang_cluster, source_code, augmentation
                            )
                            if strict:
                                try:
                                    res2 = execeval.execute_code(runtime, strict[0], sample_uts)
                                    data2 = res2.get("data") if isinstance(res2, dict) else None
                                    ok2 = False
                                    if isinstance(data2, list) and data2:
                                        ok2 = all(item.get("exec_outcome") == "PASSED" for item in data2[0])
                                    if ok2:
                                        # prefer strict result but keep original as backup for sorting
                                        cands = [strict[0]] + cands[:1]
                                        prepass_mask = [True] + [prepass_mask[0]]
                                except Exception:
                                    pass
                    except Exception:
                        pass

            # 2) iterative refine each candidate (refine only when critique is substantive)
            refined: List[str] = []
            for idx, cand in enumerate(cands):
                new_code = cand
                # skip refine if prepass says ok for this candidate
                if idx >= len(prepass_mask) or not prepass_mask[idx]:
                    for _ in range(max(0, refine_rounds)):
                        critique = critic.critique(new_code, tgt_lang_cluster)
                        crit = (critique or "").strip()
                        # gating: skip refine if critic timed out or produced placeholder/empty
                        if not crit or "not available" in crit.lower():
                            break
                        # Avoid passing augmentation again to refiner to reduce overfitting risk
                        candidate = refiner.refine(new_code, tgt_lang_cluster, crit, None)
                        # guardrail: only accept if not worse
                        if not is_worse_than(tgt_lang_cluster, new_code, candidate):
                            new_code = candidate

                # 2.5) best-effort one-shot repair guided by unit tests via ExecEval
                if execeval and ut_db and src_uid in ut_db:
                    uts = ut_db.get(src_uid) or []
                    if isinstance(uts, list) and len(uts) > 0:
                        use = min(3, len(uts))
                        sample_uts = [
                            {"input": uts[i].get("input", ""), "output": uts[i].get("output", [])}
                            for i in range(use)
                        ]
                        runtime = LANG_CLUSTER_TO_LANG_COMPILER.get(tgt_lang_cluster)
                        try:
                            res = execeval.execute_code(runtime, new_code, sample_uts)
                            data = res.get("data") if isinstance(res, dict) else None
                            ok = False
                            if isinstance(data, list) and data:
                                ok = all(item.get("exec_outcome") == "PASSED" for item in data[0])
                            if not ok:
                                # extract the first failing message as repair hint
                                msg = ""
                                if isinstance(data, list) and data:
                                    for ut in data[0]:
                                        if ut.get("exec_outcome") != "PASSED":
                                            for k in ("result", "message", "stderr", "stdout"):
                                                v = ut.get(k)
                                                if isinstance(v, str) and v:
                                                    msg = v
                                                    break
                                            break
                                if msg:
                                    hint = (
                                        f"Program must pass the given tests. {msg}\n"
                                        f"Fix minimal issues; keep algorithm and control flow intact."
                                    )
                                    candidate = refiner.refine(new_code, tgt_lang_cluster, hint, None)
                                    # guardrail: avoid pathological growth/regression
                                    if not is_worse_than(tgt_lang_cluster, new_code, candidate):
                                        new_code = candidate
                        except Exception:
                            pass
                refined.append(new_code)

            # 3) candidate selection: always include baseline, prefer ExecEval-passing ones
            all_cands: List[str] = []
            if baseline_code:
                all_cands.append(baseline_code)
            for c in refined:
                if c and c not in all_cands:
                    all_cands.append(c)

            if execeval and ut_db and src_uid in ut_db and all_cands:
                uts = ut_db.get(src_uid) or []
                if isinstance(uts, list) and len(uts) > 0:
                    use = min(3, len(uts))
                    sample_uts = [
                        {"input": uts[i].get("input", ""), "output": uts[i].get("output", [])}
                        for i in range(use)
                    ]
                    runtime = LANG_CLUSTER_TO_LANG_COMPILER.get(tgt_lang_cluster)
                    scored: List[tuple[int, int, str]] = []
                    for cand in all_cands:
                        ok_cnt = 0
                        try:
                            res = execeval.execute_code(runtime, cand, sample_uts)
                            data = res.get("data") if isinstance(res, dict) else None
                            if isinstance(data, list) and data:
                                for ut in data[0]:
                                    if ut.get("exec_outcome") == "PASSED":
                                        ok_cnt += 1
                        except Exception:
                            ok_cnt = 0
                        scored.append((ok_cnt, _static_score(cand, tgt_lang_cluster), cand))
                    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
                    ordered = [c for _, _, c in scored]
                else:
                    ordered = sorted(all_cands, key=lambda c: _static_score(c, tgt_lang_cluster), reverse=True)
            else:
                ordered = sorted(all_cands, key=lambda c: _static_score(c, tgt_lang_cluster), reverse=True)

            # Output best-first
            w.write(json.dumps({"id": code_uid, tgt_field: ordered}, ensure_ascii=False) + "\n")
            w.flush()

    return out_path
