
import os

LLAVA_MODEL_TYPE_NAME = "llava"
LLAVA_OV_MODEL_TYPE_NAME = "llava_qwen"

class ov_vit_mode(int):
	VIT_SINGLE_IMG_MODE = 0
	VIT_MULTI_IMG_MODE = 1
	VIT_VIDEO_MODE = 2

def get_lib_name_for_platform():
	lib_name = "shepherd"

	if "aarch64" == os.uname().machine and "Lychee" in os.uname().nodename:
		ambarella_arch = os.getenv("AMBARELLA_ARCH")
		lib_name = f"amba-{lib_name}-{ambarella_arch}"

	return lib_name


class infer_config():
	def __init__(
		self,
		model_path = "codellama7B",
		batch_size = 64,
		max_user_num = 1,
		device_type = 0,
		device_ip = "192.168.1.3",
		device_port = 8890,
		lib_log_level = 0,
	):
		self.model_path = model_path
		self.batch_size = batch_size
		self.max_user_num = max_user_num
		self.device_type = device_type
		self.device_ip = device_ip
		self.device_port = device_port
		self.lib_log_level = lib_log_level
		self.pos_margin = 768

		self.lib_name = get_lib_name_for_platform()

		self.max_sequence_length = None
		self.bos_token_id = None
		self.eos_token_id = None
