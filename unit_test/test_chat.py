import argparse
import signal
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers_amba_ext import AutoModelForCausalLM

run_flag = True

def sigstop(sig, frame):
	print(f"sigstop msg sig: {sig}")
	global run_flag
	run_flag = False

def llm_chat(args):
	model_path = args.model_path
	arm_sample = args.arm_sample

	model = AutoModelForCausalLM.from_pretrained(
		model_path, device_ip=args.ip, device_port=args.port, log_level=args.log_level)
	tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)
	streamer = TextIteratorStreamer(tokenizer)

	pos = [0]
	while run_flag:
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
		# Users can also use tokenizer from transformers here to do encode
		input_ids = model.encode(input_string)
		generation_kwargs = dict(
			input_ids=input_ids,
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
	parser = argparse.ArgumentParser(description='Text LLM')
	parser.add_argument('-m', '--model-path', type=str,
		default="/home/lychee/cooper_max_demos/llm_demo/llama3_8B",
		help='Specify the artifacts path of LLM model.')
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

	llm_chat(args)

