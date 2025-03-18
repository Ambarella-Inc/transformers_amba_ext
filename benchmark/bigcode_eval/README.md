<h1 align="center">Instruct HumanEval benchmark with Codellama 13B on bigcode-evaluation</h1>

## Background
It is a more recommended solution to do this benchmark on the native Lychee environment because current benchmark support different sample configurations and the `model.generate` function with do_sample might be used in `bigcode_eval/utils.py`, and the output logits size from N1 is relatively large, this will cost more time with the data transmission for RPC mode.

## Clone the bigcode-evaluation-harness project from github
```
(Lychee)# git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
(Lychee)# cd bigcode-evaluation-harness
(Lychee)# git reset --hard 6116c6a9
```

## Apply patch with transformers_amba_ext support for project
```
(Lychee)# git apply ../bigcode_eval_001_add_transformers_amba_ext_support_for_hf_models.patch


## Optional: apply RPC patch if users need to run a model on a remote PC. Users must first check and change the IP address and port number in the patch file below.
(ubuntu)# git apply ../bigcode_eval_002_add_transformers_amba_ext_support_with_rpc_for_hf_models.patch
```
Note:
* Users can setup RPC environment with [Enable RPC Mode](../../README.md#Section_Enable_RPC)

## Use local datasets(`Optional`)
Users can download the dataset from [huggingface](https://huggingface.co/datasets/codeparrot/instructhumaneval) on the GPU server as the command below if current board can not download it directly, then copy the instructhumaneval folder to the current workspace.

```
// download dataset on the GPU server
(ubuntu)# mkdir -p codeparrot; cd codeparrot
(ubuntu)# git clone https://huggingface.co/datasets/codeparrot/instructhumaneval

// copy the codeparrot folder to the current workspace
```

## Install dependencies for bigcode_eval
```
(Lychee)# pip3 install -e .
```

## Evaluation
```
(Lychee)# python3 main.py \
	--model /home/lychee/cooper_max_demos/llm_demo/codellama_13B/ \
	--tasks instruct-humaneval \
	--instruction_tokens [INST],[/INST],'' \
	--limit 164 --no_do_sample \
	--n_samples 1 --batch_size 1 \
	--allow_code_execution --save_generations
```
Note:
* Users can set different sample configurations with temperature, top_p, and top_k while do_sample is set to true, and for example: `--do_sample true --temperature 0.2 --top_p 0.95 --top_k 0`.
