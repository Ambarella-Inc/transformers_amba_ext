
import os
from typing import Optional

AMBA_EXT_MAX_BATCH_SIZE:int = 0
AMBA_EXT_VLM_VIT_NAME:str = ""

LLAVA_MODEL_TYPE_NAME = "llava"
LLAVA_OV_MODEL_TYPE_NAME = "llava_qwen"
VLM_MODEL_TYPE_NAME = "vlm"

# Unified external VIT mode for all VLM models
class vit_mode(int):
	SINGLE = 0   # Single image mode
	MULTI = 1    # Multi image mode
	VIDEO = 2    # Video mode
	AUDIO = 3    # Audio mode

class device_type(int):
	LOCAL = 0
	REMOTE = 1

class backend_type(str):
	SHEPD = "shepherd"
	CRT = "crt"

def get_lib_name_for_platform(backend: backend_type):
	arch = os.getenv("AMBARELLA_ARCH")
	ambarella_arch = arch if arch is not None else "n1"

	if backend == backend_type.CRT:
		lib_name = "cavalryrt"

	elif backend == backend_type.SHEPD:
		lib_name = backend
		if "aarch64" == os.uname().machine and ("Lychee" in os.uname().nodename or "Systech" in os.uname().nodename):
			lib_name = f"amba-{backend}-{ambarella_arch}"

	return ambarella_arch, lib_name

class infer_config():
	def __init__(
		self,
		model_path = "codellama7B",
		# batch_size = 64,
		max_user_num = 1,
		device_ip = "192.168.1.3",
		device_port = 8890,
		device_id = 0,
		lib_log_level = 0,
		backend: Optional[backend_type] = None,
		max_img_num = None,
	):
		self.batch_size = \
			int(os.getenv("AMBA_EXT_MAX_BATCH_SIZE", AMBA_EXT_MAX_BATCH_SIZE))
		self.vlm_vit_name = \
			os.getenv("AMBA_EXT_VLM_VIT_NAME", AMBA_EXT_VLM_VIT_NAME)

		self.model_path = model_path
		self.max_user_num = max_user_num
		self.device_type = device_type.REMOTE \
			if device_ip is not None or device_port is not None \
			else device_type.LOCAL
		self.device_ip = device_ip
		self.device_port = device_port
		self.device_id = int(device_id) if device_id is not None else 0
		self.lib_log_level = lib_log_level
		self.backend = backend if backend is not None else backend_type.SHEPD
		if self.backend not in (
				backend_type.SHEPD, backend_type.CRT):
			raise ValueError(f"Unsupported backend: {self.backend}")

		# test_crt_shpd_vlm / test_shepherd_vlm --in N.
		if max_img_num is None:
			max_img_num = os.getenv("AMBA_EXT_VLM_MAX_IMG_NUM", "1")
		self.max_img_num = max(1, int(max_img_num))

		self.arch, self.lib_name = get_lib_name_for_platform(self.backend)

		self.pos_margin = 768
		self.max_sequence_length = None
		self.bos_token_id = None
		self.eos_token_id = None
