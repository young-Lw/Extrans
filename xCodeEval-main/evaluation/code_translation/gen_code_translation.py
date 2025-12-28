import os
import time
import tqdm
import json
import openai
import argparse
import datasets
import concurrent
import numpy as np
from promptsource.templates import Template

SHORT_LANG_MAP = {
    "GNU C++": "C++",
    "GNU C++17": "C++",
    "MS C++ 2017": "C++",
    "MS C++": "C++",
    "Java 8": "Java",
    "Java 6": "Java",
    "GNU C++11": "C++",
    "Java 11": "Java",
    "GNU C++14": "C++",
    "Mono C#": "C#",
    "GNU C": "C",
    "Python 3": "Python",
    "PyPy 3": "Python",
    "GNU C11": "C",
    "Go": "Go",
    "Rust": "Rust",
    "PyPy 2": "Python",
    "Python 2": "Python",
    "MS C#": "C#",
    "Kotlin": "Kotlin",
    "GNU C++0x": "C++",
    "Java 7": "Java",
    "Node.js": "Javascript",
    ".NET Core C#": "C#",
    "PHP": "PHP",
    "GNU C++17 Diagnostics": "C++",
    "Clang++17 Diagnostics": "C++",
    "JavaScript": "Javascript",
    "Ruby": "Ruby",
    "C# 10": "C#",
    "C# 8": "C#",
    "Clang++20 Diagnostics": "C++",
    "GNU C++17 (64)": "C++",
    "GNU C++20 (64)": "C++",
    "Java 17": "Java",
    "Kotlin 1.4": "Kotlin",
    "Kotlin 1.5": "Kotlin",
    "Kotlin 1.6": "Kotlin",
    "Kotlin 1.7": "Kotlin",
    "PyPy 3-64": "Python",
    "Python 3 + libs": "Python",
    "Ruby 3": "Ruby",
    "Rust 2021": "Rust",
}

LANGS = sorted(set([v for k, v in SHORT_LANG_MAP.items()]))


openai.api_key = os.environ["OPENAI_API_KEY"]


def gen(prompt, temperature, nsample):
    cnt = 0
    while True:
        if cnt == 999:
            return None
        try:
            c = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"{prompt}"},
                ],
                temperature=temperature,
                top_p=1,
                n=nsample,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            break
        except Exception as e:
            cnt += 1
            time.sleep(5)
            print(f"{e}")
    c["prompt"] = prompt
    return c


xcodeeval_prompt_template = {
    "code_translation": [
        "Here is code in {{source_lang}} programming lanaguge. Translate the following code from {{source_lang}} to {{target_lang}} programming lanaguge. Do not output any extra description or tokens other than the translated code. \n\n{{source_code}}||END-of-SRC|| "
    ]
}


def process_prompt(
    dt, temperature, template, language, nsample, output_dir, index, dry_run=0
):
    dt["source_lang"] = dt["lang"]
    dt["target_lang"] = language
    language = f"{dt['source_lang']}--{dt['target_lang']}"
    file_path = os.path.join(output_dir, f"{index}_{temperature}_{language}.json")
    if not os.path.exists(file_path):
        dt["prob_desc_sample_inputs"] = json.loads(dt["prob_desc_sample_inputs"])
        dt["prob_desc_sample_outputs"] = json.loads(dt["prob_desc_sample_outputs"])
        lm_io = template.apply(dt)
        assert len(lm_io) == 2, f"{json.dumps(lm_io, indent=4)}"
        if dry_run:
            open(file_path, "w").write(f"{json.dumps(lm_io[0], indent=4)}")
        else:
            out = gen(lm_io[0], temperature, nsample)
            export_data = {"oai_response": out, "source_data": dt}
            open(file_path, "w").write(f"{json.dumps(export_data, indent=4)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="dumped/oai/code_translation_n_sample_20",
        help="Output Folder to save the API request.",
    )
    parser.add_argument(
        "--num-proc",
        default=1,
        help="Number of parallel API request.",
    )
    parser.add_argument(
        "--dry-run",
        default=0,
        help="Number of parallel API request.",
    )
    parser.add_argument(
        "--nsample",
        default=20,
        type=int,
        help="Number of parallel API request.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    templates = [
        Template(
            f"code_translation_{idx}", template, "xCodeEval", delimeter="||END-of-SRC||"
        )
        for idx, template in enumerate(xcodeeval_prompt_template["code_translation"])
    ]
    template = templates[0]

    code_translation_dataset_small = datasets.load_dataset(
        "NTU-NLP-sg/xCodeEval", "code_translation", num_proc=16, trust_remote_code=True
    )[
        "compact_small"
    ]
    code_translation_dataset = datasets.load_dataset(
        "NTU-NLP-sg/xCodeEval", "code_translation", num_proc=16
    )[
        "compact"
    ]
    temperature_list = [0.3157894736842105]

    out_dir = args.output_dir + "/compact_small"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.num_proc)
    ) as executor:
        futures = []
        for idx, dt in tqdm.tqdm(
            enumerate(code_translation_dataset_small),
            total=len(code_translation_dataset_small),
            desc=f"Preparing samples",
        ):
            for language in LANGS:
                if SHORT_LANG_MAP[dt["lang"]] == language:
                    continue
                for temperature in temperature_list:
                    future = executor.submit(
                        process_prompt,
                        dt,
                        temperature,
                        template,
                        language,
                        args.nsample,
                        out_dir,
                        idx,
                        args.dry_run,
                    )
                    futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Calling OpenAI API",
        ):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

    out_dir = args.output_dir + "/compact"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.num_proc)
    ) as executor:
        futures = []
        for idx, dt in tqdm.tqdm(
            enumerate(code_translation_dataset),
            total=len(code_translation_dataset),
            desc=f"Preparing samples",
        ):
            for language in ["Python"]:
                if SHORT_LANG_MAP[dt["lang"]] == language:
                    continue
                for temperature in temperature_list:
                    future = executor.submit(
                        process_prompt,
                        dt,
                        temperature,
                        template,
                        language,
                        args.nsample,
                        out_dir,
                        idx,
                        args.dry_run,
                    )
                    futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Calling OpenAI API",
        ):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
