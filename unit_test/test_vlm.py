import argparse
import signal
import numpy as np
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers_amba_ext import VLMForCausalLM, vit_mode

run_flag = True

def get_user_input():
    print("[user]: (Press Enter twice to finish)")
    lines = []
    while True:
        line = input()
        if line == "":
            if not lines: continue
            break
        lines.append(line)
    return "\n".join(lines)

def sigstop(sig, frame):
	print(f"sigstop msg sig: {sig}")
	global run_flag
	run_flag = False

def vllm_chat(args):
	model_path = args.model_path
	arm_sample = args.arm_sample
	image = args.image
	vmode = args.vit_type

	model = VLMForCausalLM.from_pretrained(
		model_path, device_ip=args.ip, device_port=args.port, log_level=args.log_level)
	tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)
	streamer = TextIteratorStreamer(tokenizer)

	# VLM usually expects images in (N, 3, H, W) format.
	# Here we load from binary file. Users need to ensure the binary format matches.
	# For generic VLM, we assume a reasonable default or let the user provide pre-formatted binary.
	image_tensor = np.fromfile(image, dtype=np.uint8)

	# Note: The exact reshape depends on the model configuration.
	# For demonstration, we use a placeholder or try to infer.
	# In actual use, this should match the vision tower input requirements.
	try:
		# Example for VLM-Chat-V1.5 or similar often uses 448x448
		image_tensor = image_tensor.reshape(-1, 3, 448, 448)
	except:
		print("Warning: Could not reshape image to (-1, 3, 448, 448). Using raw input if possible.")

	model.tokenizer_image_token(image_tensor, vit_mode=vmode)
	pending_image = image_tensor
	pos = [0]
	while run_flag:
		print("\n\n")
		print(f"(pos {pos[0]})")
		print("[user]:\n")
		input_string = get_user_input()
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
	parser = argparse.ArgumentParser(description='VLM Chat')
	parser.add_argument('-m', '--model-path', type=str,
		default="/home/lychee/cooper_max_demos/llm_demo/vlm",
		help='Specify the artifacts path of VLM model.')
	parser.add_argument('-i', '--image', type=str,
		default="test_image.bin",
		help='Specify the input image binary.')
	parser.add_argument('-t', '--vit-type', type=int,
		default=0,
		help='Specify the unified vit mode. 0: SINGLE; 1: MULTI; 2: VIDEO; 3: AUDIO.')
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

	signal.signal(signal.SIGINT, sigstop)
	signal.signal(signal.SIGQUIT, sigstop)
	signal.signal(signal.SIGTERM, sigstop)

	vllm_chat(args)
