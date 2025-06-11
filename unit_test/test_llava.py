import argparse
import numpy as np
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers_amba_ext import LlavaLlamaForCausalLM

def vllm_chat(args):
	model_path = args.model_path
	arm_sample = args.arm_sample
	image = args.image

	model = LlavaLlamaForCausalLM.from_pretrained(
		model_path, device_ip=args.ip, device_port=args.port, log_level=args.log_level)
	tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)
	streamer = TextIteratorStreamer(tokenizer)

	image_tensor = np.fromfile(image, dtype=np.uint8).reshape(-1, 3, 336, 336)
	model.tokenizer_image_token(image_tensor)

	pos = [0]
	while (1):
		print("\n\n")
		print(f"(pos {pos[0]})")
		print("[user]:\n")
		input_string = input()
		if input_string == "":
			continue

		if (input_string == "EOT") and (len(input_string) == 3):
			break

		if (input_string == "RESET") and (len(input_string) == 5):
			pos[0] = model.reset()
			continue

		print("\n[amba]\n")

		prompt_ids = model.encode(input_string)
		generation_kwargs = dict(
			input_ids=prompt_ids,
			past_key_values=True,
			position=pos,
			streamer=streamer,
			do_sample = False if arm_sample == True else None)
		thread = Thread(target=model.generate, kwargs=generation_kwargs)
		thread.start()
		for piece in streamer:
			print(piece, end="", flush=True)

		thread.join()

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Vision LLM')
	parser.add_argument('-m', '--model-path', type=str,
		default="/home/lychee/cooper_max_demos/llm_demo/llava_13B",
		help='Specify the artifacts path of LLM model.')
	parser.add_argument('-i', '--image', type=str,
		default="image.bin",
		help='Specify the input image binrary.')
	parser.add_argument('--ip', type=str,
		default=None,
		help='the ip of remote board with RPC mode.')
	parser.add_argument('--port', type=int,
		default=None,
		help='the port of remote board with RPC mode.')
	parser.add_argument('--arm-sample', action='store_const', const=True,
		default=False,
		help='Enable arm sample or not. Default is disabled.')
	parser.add_argument('-v', '--log-level', type=int,
		default=0,
		help='Log level for Shepherd. 0: error; 1: warn; 2: info; 3: debug; 4: verbose.')
	args = parser.parse_args()

	vllm_chat(args)

