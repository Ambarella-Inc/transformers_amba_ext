import argparse
import time, copy
import numpy as np
from threading import Thread
from transformers import AutoTokenizer, AutoConfig, TextIteratorStreamer
from transformers_amba_ext import LlavaOnevisionForConditionalGeneration

global Thread_State
Thread_State = True

def generate_thread(streamer):
	while Thread_State:
		if streamer.text_queue.empty():
			time.sleep(0.01)
		else:
			for piece in streamer:
				print(piece, end="", flush=True)

def vllm_chat(args):
	model_path = args.model_path
	arm_sample = args.arm_sample
	image = args.image
	vit_mode = args.vit_type

	model = LlavaOnevisionForConditionalGeneration.from_pretrained(
		model_path, device_ip=args.ip, device_port=args.port, log_level=args.log_level)
	tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)
	streamer = TextIteratorStreamer(tokenizer)
	generation_kwargs = dict(streamer=streamer)
	thread = Thread(target=generate_thread, kwargs=generation_kwargs)
	thread.start()

	if vit_mode == 0:
		image_tensor = np.fromfile(image, dtype=np.uint8).reshape(-1, 3, 720, 1280)
	else:
		image_tensor = np.fromfile(image, dtype=np.uint8).reshape(-1, 3, 384, 384)
	model.tokenizer_image_token(image_tensor, vit_mode=vit_mode)

	pos = [0]
	while (1):
		print("\n\n")
		print(f"(pos {pos[0]})")
		print("[user]:\n")
		input_string = input()
		if input_string == "":
			continue

		if (input_string == "EOT") and (len(input_string) == 3):
			global Thread_State
			Thread_State = False
			break

		if (input_string == "RESET") and (len(input_string) == 5):
			pos[0] = model.reset()
			continue

		print("\n[amba]\n")

		prompt_ids = model.encode(input_string)
		model.generate(
			input_ids=prompt_ids,
			past_key_values=True,
			position=pos,
			do_sample = False if arm_sample == True else None,
			streamer=streamer)

	thread.join()

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Vision LLM')
	parser.add_argument('-m', '--model-path', type=str,
		default="/home/lychee/cooper_max_demos/llm_demo/llava_onevision",
		help='Specify the artifacts path of LLM model.')
	parser.add_argument('-i', '--image', type=str,
		default="auto_cat.bin",
		help='Specify the input image binrary.')
	parser.add_argument('-t', '--vit-type', type=int,
		default=1,
		help='Specify the vit mode for vision tower. 0: single image; 1: multi image; 2: video.')
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

