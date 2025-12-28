import argparse
from .pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser("EXtrans multi-agent translation runner (test-aug + stronger sorting)")
    ap.add_argument("--xcode_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--src_lang_cluster", required=True, choices=["Java", "Python"])
    ap.add_argument("--tgt_lang_cluster", required=True, choices=["Java", "Python"])
    ap.add_argument("--translator_model", default="gpt-4o")
    ap.add_argument("--critic_model", default="gpt-4o-mini")
    ap.add_argument("--refiner_model", default="gpt-4-turbo")
    ap.add_argument("--nsample", type=int, default=1)
    ap.add_argument("--refine_rounds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_path", required=True)
    # API robustness
    ap.add_argument("--api_timeout", type=int, default=0, help="0 disables per-call timeout")
    ap.add_argument("--api_retries", type=int, default=0)
    ap.add_argument("--max_empty_retries", type=int, default=-1, help="-1 for infinite retries on empty output")

    # new flags (all optional; defaults keep parity with original)
    ap.add_argument("--use_testcase_aug", action="store_true")
    ap.add_argument("--k_aug_tests", type=int, default=2)

    args = ap.parse_args()
    run_pipeline(
        xcode_dir=args.xcode_dir,
        split=args.split,
        src_lang_cluster=args.src_lang_cluster,
        tgt_lang_cluster=args.tgt_lang_cluster,
        out_path=args.out_path,
        translator_model=args.translator_model,
        critic_model=args.critic_model,
        refiner_model=args.refiner_model,
        nsample=args.nsample,
        refine_rounds=args.refine_rounds,
        limit=args.limit,
        use_testcase_aug=args.use_testcase_aug,
        k_aug_tests=args.k_aug_tests,
        api_timeout=args.api_timeout,
        api_retries=args.api_retries,
        max_empty_retries=args.max_empty_retries,
    )


if __name__ == "__main__":
    main()
