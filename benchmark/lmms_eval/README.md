<h1 align="center">MMBench/VQAV2 with Llava13B/LlavaOV on lmms-eval</h1>

## Clone the lmms-eval project from github
```
(Lychee)# git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
(Lychee)# cd lmms-eval
(Lychee)# git reset --hard b5a3050
```

## Apply patch with transformers_amba_ext support for project
```
(Lychee)# git apply ../lmms_eval_001_add_transformers_amba_ext_support_for_llava_and_llava-onevisin.patch


## Optional: apply RPC patch if users need to run a model on a remote PC. Users must first check and change the IP address and port number in the patch file below.
(ubuntu)# git apply ../lmms_eval_002_add_transformers_amba_ext_support_with_rpc_for_llava_and_llava-onevision.patch
```
Note:
* Users can setup RPC environment with [Enable RPC Mode](../../README.md#Section_Enable_RPC)

## Use local datasets(`Optional`)
Users can download the lmms-lab/MMBench dataset from [huggingface](https://huggingface.co/datasets/lmms-lab/MMBench) on the GPU server as the command below if current board can not download it directly, then copy the lmms-lab folder to the current workspace.

```
// download dataset on the GPU server
(ubuntu)# mkdir -p lmms-lab; cd lmms-lab
(ubuntu)# git clone https://huggingface.co/datasets/lmms-lab/MMBench

// copy the lmms-lab folder to the current workspace
```

## Install dependencies for lmms_eval
```
(Lychee)# pip3 install -e .
```

## Evaluation
It is a more recommended solution to do the VQAV2 benchmark on the remote PC because the size of VQAV2 is too large(214K) and the CPU memory on Lychee maybe not enough to load it. Then users can switch to RPC mode if they also meet the memory issue on native Lychee.

#### Evaluation with Llava13B
Llava13B only support the single image datasets like MMBench and VQAV2.
```
(Lychee)# lmms-eval \
		--model llava \
		--model_args pretrained=${model_path},device_map="cpu" \
		--tasks mmbench_en_dev --batch_size 1 --device cpu \
		--output_path output
```
Note:
* Users need to downlaod `preprocessor_config.json` from [huggingface](https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/preprocessor_config.json) if encounter an error related to this file
* Users can change tasks name for different dataset like VQAV2.
* Users can get the reported result from [LMMs-Eval Shared Results](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?gid=0#gid=0).

#### Evaluation with Llava-OneVision
Llava-OneVision supports single-image datasets like MMBench and VQAV2 with high-resolution mode on GPU by default. However, our transformers_amba_ext package only supports multi-image mode for now, so we have to make a small change with `image_aspect_ratio=pad` to enable multi-image mode for those single-image datasets. Our patch has applied this change by default.

Llava-OneVision also supports multi-image datasets like LLaVA-NeXT-Interleave-Bench and chooses multi-image mode by default.
```
(Lychee)# lmms-eval \
		--model llava_onevision \
		--model_args pretrained="${model_path},conv_template=qwen_1_5,model_name=llava_qwen" \
		--tasks mmbench_en_dev --verbosity=DEBUG \
		--batch_size 1 --log_samples --log_samples_suffix llava_onevision \
		--output_path output_next
```
Note:
* Users can change tasks name for different dataset like VQAV2 or LLaVA-NeXT-Interleave-Bench.
* Users can get the reported result from [Paper](https://arxiv.org/pdf/2408.03326).
