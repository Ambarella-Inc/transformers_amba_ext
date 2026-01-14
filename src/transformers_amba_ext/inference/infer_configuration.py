
import os
from ..inference.libshepd_inc import vlm_vit_mode

LLAVA_MODEL_TYPE_NAME = "llava"
LLAVA_OV_MODEL_TYPE_NAME = "llava_qwen"
VLM_MODEL_TYPE_NAME = "vlm"

# Unified external VIT mode for all VLM models
class vit_mode(int):
	SINGLE = 0   # Single image mode
	MULTI = 1    # Multi image mode
	VIDEO = 2    # Video mode
	AUDIO = 3    # Audio mode


class ov_vit_mode(int):
	VIT_SINGLE_IMG_MODE = 0
	VIT_MULTI_IMG_MODE = 1
	VIT_VIDEO_MODE = 2

def map_vit_mode_for_llava_ov(mode):
	if mode == vit_mode.SINGLE:
		return ov_vit_mode.VIT_SINGLE_IMG_MODE
	elif mode == vit_mode.MULTI:
		return ov_vit_mode.VIT_MULTI_IMG_MODE
	elif mode == vit_mode.VIDEO:
		return ov_vit_mode.VIT_VIDEO_MODE
	else:
		raise ValueError(f"Unsupported vit_mode: {mode} for llava_ov")

def map_vit_mode_for_vlm(mode):
	if mode == vit_mode.SINGLE:
		return vlm_vit_mode.VLM_VIT_IMAGE
	elif (mode == vit_mode.MULTI) or (mode == vit_mode.VIDEO):
		return vlm_vit_mode.VLM_VIT_VIDEO
	elif mode == vit_mode.AUDIO:
		return vlm_vit_mode.VLM_VIT_AUDIO
	else:
		raise ValueError(f"Unsupported vit_mode: {mode} for vlm")


def get_lib_name_for_platform():
	arch = os.getenv("AMBARELLA_ARCH")
	ambarella_arch = arch if arch is not None else "n1"

	lib_name = "shepherd"
	if "aarch64" == os.uname().machine and "Lychee" in os.uname().nodename:
		lib_name = f"amba-{lib_name}-{ambarella_arch}"

	return ambarella_arch, lib_name


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

		self.arch, self.lib_name = get_lib_name_for_platform()

		self.max_sequence_length = None
		self.bos_token_id = None
		self.eos_token_id = None
