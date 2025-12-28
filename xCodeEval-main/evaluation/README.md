# How to perform evaluation using ExecEval

## Configure ExecEval

Follow the instrauction [here](https://github.com/ntunlp/execeval).

> :warning: ❌ Do not run ExecEval without Docker image

`Java` and `Kotlin 1.5` has a lots of Memory related issue. If you don't have large amount of memory please reduce the multiple worker. If you are running in a laptop or Desktop with limited amount of RAM (~ 32 GB), please use num `NUM_WORKERS`=1. Performance holds well upto 1/3rd of the CPU available at max.

## Setup Environment
Install python packages in your own environment.
```
pip install -r requirement.txt
```

Install ExecEval.
```
git clone https://github.com/ntunlp/ExecEval
cd ExecEval
docker build . -t exec-eval:1.0
docker run -it -p 5000:5000 -e NUM_WORKERS=37 exec-eval:1.0
```

## Generate samples

Generate samples using OpenAI api.

```
python evaluation/program_synthesis/gen_program_synthesis.py
python evaluation/code_translation/gen_code_translation.py
python evaluation/apr/gen_apr.py
```

## Eval Samples using ExecEval

Keep ExecEval server/endpoint running and then run the following code, 

```
python evaluation/program_synthesis/eval_program_synthesis.py
python evaluation/code_translation/eval_code_translation.py
python evaluation/apr/eval_apr.py
```

## Calculate pass@k

Calculate pass@k by the following script,

```
python evaluation/program_synthesis/get_result.py
python evaluation/code_translation/get_result.py
python evaluation/apr/get_result.py
```