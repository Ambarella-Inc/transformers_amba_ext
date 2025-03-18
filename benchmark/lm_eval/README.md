<h1 align="center">MMLU benchmark with Llama3_8B on lm-evaluation-harness</h1>

## Background
It is a more recommended solution to do this benchmark on the native Lychee environment because the function with `model(inps).logits` was used in `lm_eval/models/huggingface.py` and the output logits size(`(input_token_num + 1) * vocab_size * sizeof(fp16)`) from N1 is relatively large, this will cost more time with the data transmission for RPC mode.

## Clone the lm-evaluation-harness project from github
```
(Lychee)# git clone https://github.com/EleutherAI/lm-evaluation-harness.git
(Lychee)# cd lm-evaluation-harness
(Lychee)# git reset --hard d693dcd
```
## Apply patch with transformers_amba_ext support for project
```
(Lychee)# git apply ../lm_eval_001_add_transformers_amba_ext_support_for_hf_models.patch


## Optional: apply RPC patch if users need to run a model on a remote PC. Users must first check and change the IP address and port number in the patch file below.
(ubuntu)# git apply ../lm_eval_002_add_transformers_amba_ext_support_with_rpc_for_hf_models.patch
```
Note:
* Users can setup RPC environment with [Enable RPC Mode](../../README.md#Section_Enable_RPC)

## Apply patch with local datasets support for MMLU benckmark(`Optional`)
Users can use python script (`dataset_hails_mmlu_save_to_disk.py`) to generate offline datasets from [huggingface](https://huggingface.co/datasets/hails/mmlu_no_train) on the GPU server if current board can not download it directly, then copy the hails_mmlu_no_train folder to the current workspace and apply below patch to enable using local datasets.
```
(Lychee)# git apply ../lm_eval_003_add_local_datasets_support_for_mmlu_benchmark.patch
```

## Install dependencies for lm_eval
```
(Lychee)# pip3 install -e .
```

## Evaluation
```
(Lychee)# lm_eval \
	--model hf \
	--model_args pretrained=/home/lychee/cooper_max_demos/llm_demo/llama3_8B/ \
	--tasks mmlu \
	--device cpu \
	--batch_size 1 \
	--output_path output/ \
	--system_instruction "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
	--log_samples --num_fewshot 0
```

Note:
* Users can set the system_instruction config or not, the used value (`<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n`) comes from the prompt_config.json of the model path.
* Users can get the reported result with 5-shot from [Meta](https://ai.meta.com/blog/meta-llama-3/).
* Users need to change the model path(`--model_args pretrained=`) and `system_instruction` if they want to do MMLU benchmark with other LLM models.
