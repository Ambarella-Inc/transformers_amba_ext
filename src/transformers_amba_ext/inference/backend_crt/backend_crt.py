import ctypes
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from ...inference.backend_crt import __version__
from ...inference.backend_crt import libcrt as libcrt
from ...inference.backend_crt import libcrt_inc as crt_inc
from ...inference.backend_crt import libcrt_shpd as libcrt_shpd
from ...inference.backend_shepd import libshepd_inc as inc
from ...inference.backend_shepd.libshepd_inc import vlm_vit_mode
from ...inference import infer_configuration as config
from ...inference.infer_configuration import (
	LLAVA_OV_MODEL_TYPE_NAME, LLAVA_MODEL_TYPE_NAME, VLM_MODEL_TYPE_NAME, vit_mode)
from ...utils import logging

logger = logging.get_logger(__name__)

def _resolve_vit_net_path(
	model_path: str,
	keyword: str,
	suffix: str = ".bin",
) -> str:
	model_dir = Path(model_path)
	if not model_dir.is_dir():
		raise FileNotFoundError(f"Model path not found: {model_path}")

	matches = sorted(
		f.name
		for f in model_dir.iterdir()
		if f.is_file() and keyword in f.name and f.name.endswith(suffix)
	)

	if not matches:
		raise FileNotFoundError(
			f"No file matching keyword '{keyword}' under {model_path}")

	if len(matches) > 1:
		logger.warning(
			"Multiple VIT net files match keyword '%s': %s, using '%s'",
			keyword, matches, matches[0],
		)

	return os.path.join(model_path, matches[0])


def _discover_vlm_encoder_nets(model_path, vlm_vit_name="", arch="n1"):
	"""Discover image / video / audio encoder nets under a VLM pack.

	Returns a list of (vlm_vit_mode, net_path). A pack may ship one or more of
	audio_model.bin, vision_model.bin, video_model.bin.
	"""
	nets = []
	if vlm_vit_name:
		vit_net_path = os.path.join(model_path, vlm_vit_name)
		if not os.path.exists(vit_net_path):
			raise FileNotFoundError(
				f"vit_name: {vlm_vit_name} not found in model_path: {model_path}.\n"
				f"Users can also use 'export AMBA_EXT_VLM_VIT_NAME=USER_VIT_NAME' "
				f"for a different VLM model")
		name = vlm_vit_name.lower()
		if "audio" in name:
			mode = vlm_vit_mode.VLM_VIT_AUDIO
		elif "video" in name:
			mode = vlm_vit_mode.VLM_VIT_VIDEO
		else:
			mode = vlm_vit_mode.VLM_VIT_IMAGE
		return [(mode, vit_net_path)]

	candidates = [
		(vlm_vit_mode.VLM_VIT_AUDIO, "audio_model.bin"),
		(vlm_vit_mode.VLM_VIT_IMAGE, "vision_model.bin"),
		(vlm_vit_mode.VLM_VIT_IMAGE, f"vit_{arch}_cavalry.bin"),
		(vlm_vit_mode.VLM_VIT_VIDEO, "video_model.bin"),
	]
	seen_modes = set()
	for mode, name in candidates:
		vit_net_path = os.path.join(model_path, name)
		if os.path.exists(vit_net_path) and mode not in seen_modes:
			nets.append((mode, vit_net_path))
			seen_modes.add(mode)
	return nets

class ov_vit_mode(int):
	VIT_SINGLE_IMG_MODE = 0
	VIT_MULTI_IMG_MODE = 1
	VIT_VIDEO_MODE = 2

class infer_crt(libcrt.libcrt_api):
	def __init__(self, config: config.infer_config):
		self.config = config
		self.dev_ctx = None
		self._shpd = None

		""" The shepd_cfg can't be defined as a local variable in function
			since the pointer of model_path was used by libcrt/shepherd
		"""
		self.shepd_cfg = inc.shepd_config()
		super().__init__(self.config.lib_name)
		self._shpd = libcrt_shpd.libcrt_shpd_api(self.crt_api)

	def __load_vit_img_ctype(self, img_data):
		# Preserve original bytes (uint8 images or packed fp16 audio).
		blob = np.ascontiguousarray(img_data).reshape(-1)
		if blob.dtype != np.uint8:
			blob = blob.view(np.uint8)
		vit_img_size = int(blob.size)
		vit_img_ctype = (ctypes.c_uint8 * vit_img_size).from_buffer_copy(blob)
		return vit_img_ctype, vit_img_size

	def infer_get_version(self):
		shepd_ver = inc.shepherd_version()
		rval = self._shpd.crt_shpd_get_version(self.dev_ctx, ctypes.byref(shepd_ver))
		logger.debug("[crt shepd ver] version: %u.%u.%u, mod_time: 0x%x, retval: %u" % (
			shepd_ver.major, shepd_ver.minor, shepd_ver.patch, shepd_ver.mod_time, rval))
		version = "%u.%u.%u" % (shepd_ver.major, shepd_ver.minor, shepd_ver.patch)
		return version


	def _infer_version_check(self):
		ver_infer = __version__
		ver_shepd = self.infer_get_version()
		ver_infer_list = ver_infer.split('.')
		ver_shepd_list = ver_shepd.split('.')

		if (ver_infer_list[0] != ver_shepd_list[0]) or (ver_infer_list[1] != ver_shepd_list[1]):
			raise ValueError(
				f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				f"Wrapper API on the transformers_amba_ext package. Version: {ver_infer}\n"
				f"Backend API of remote shepherd via libcrt. Version: {ver_shepd}\n"
				f"This mismatched version might trigger some unknown issues with incompatibility API.\n"
				f"Please contact the Ambarella team if you are unsure how to handle it.\n"
				f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")


	def infer_init(self):
		init_cfg = crt_inc.crt_config()
		init_cfg.log_level = self.config.lib_log_level
		rval = super().crt_init(ctypes.byref(init_cfg))
		if rval < 0:
			raise ValueError(f"[crt_init] fail, rval: {rval}")

		device_list = crt_inc.crt_device()
		rval = super().crt_get_device_list(ctypes.byref(device_list))
		if rval < 0:
			raise ValueError(f"[crt_get_device_list] fail, rval: {rval}")

		device_id = getattr(self.config, "device_id", 0) or 0
		if device_id >= device_list.device_num:
			raise ValueError(
				f"device_id {device_id} >= device_num {device_list.device_num}")

		self.dev_ctx = super().crt_create_device_context(device_id)
		if not self.dev_ctx:
			raise ValueError(f"[crt_create_device_context] fail, device_id: {device_id}")

		logger.info(f"[crt] device_id={device_id}, device_num={device_list.device_num}")
		self._infer_version_check()


	def infer_exit(self):
		if self.dev_ctx:
			rval = super().crt_destroy_device_context(self.dev_ctx)
			if rval < 0:
				raise ValueError(f"[crt_destroy_device_context] fail, rval: {rval}")
			self.dev_ctx = None

		rval = super().crt_exit()
		if rval < 0:
			raise ValueError(f"[crt_exit] fail, rval: {rval}")


	def __infer_model_llava_init(self):
		vit_net_path = f"{self.config.model_path}/vision_model.bin"
		self.shepd_cfg.shepd_ex.llava_ex.vit_net_fn = ctypes.c_char_p(vit_net_path.encode('utf-8'))
		self.shepd_cfg.shepd_ex.llava_ex.img_start_token_id = 29871
		self.shepd_cfg.shepd_ex.llava_ex.img_end_token_id = 29871
		self.shepd_cfg.shepd_ex.llava_ex.vit_in.internal_cavalry_mem = 1
		logger.info(f"llava_13B: \n"
			f"  vit_net_fn: {self.shepd_cfg.shepd_ex.llava_ex.vit_net_fn}\n"
			f"  img_start_token_id: {self.shepd_cfg.shepd_ex.llava_ex.img_start_token_id}\n"
			f"  img_end_token_id: {self.shepd_cfg.shepd_ex.llava_ex.img_end_token_id}")


	def __infer_model_vlm_init(self):
		nets = _discover_vlm_encoder_nets(
			self.config.model_path, self.config.vlm_vit_name, self.config.arch)
		if not nets:
			raise FileNotFoundError(
				f"No VLM encoder net found in model_path: {self.config.model_path}.\n"
				f"Expected audio_model.bin, vision_model.bin, or vit_{self.config.arch}_cavalry.bin.\n"
				f"Users can also use 'export AMBA_EXT_VLM_VIT_NAME=USER_VIT_NAME' "
				f"for a different VLM model")

		vit_net_num = len(nets)
		self.vlm_vit = (inc.vlm_vit * vit_net_num)()
		self.shepd_cfg.shepd_ex.vlm_ex.size = vit_net_num
		self.shepd_cfg.shepd_ex.vlm_ex.vit = ctypes.cast(self.vlm_vit, ctypes.c_void_p)

		for i, (mode, vit_net_path) in enumerate(nets):
			self.vlm_vit[i].vit_mode = mode
			self.vlm_vit[i].max_img_num = getattr(self.config, "max_img_num", 1)
			self.vlm_vit[i].vit_net_fn = ctypes.c_char_p(vit_net_path.encode('utf-8'))
			logger.info(
				f"VLM[{i}]: vit_net_fn: {vit_net_path}, "
				f"vit_mode: {mode}, max_img_num: {self.vlm_vit[i].max_img_num}")

	def _vlm_net_index_for_mode(self, vit_mode):
		size = int(self.shepd_cfg.shepd_ex.vlm_ex.size)
		for i in range(size):
			if self.vlm_vit[i].vit_mode == vit_mode:
				return i
		available = [int(self.vlm_vit[i].vit_mode) for i in range(size)]
		raise ValueError(
			f"No VLM encoder net registered for vit_mode={vit_mode}, "
			f"available={available}")

	def __infer_model_llava_ov_init(self):
		vit_net_num = 3
		self.llava_ov_vit = (inc.llava_onevision_vit * vit_net_num)()
		self.shepd_cfg.shepd_ex.llava_onevision_ex.size = vit_net_num
		self.shepd_cfg.shepd_ex.llava_onevision_ex.vit = ctypes.cast(self.llava_ov_vit, ctypes.c_void_p)

		vit_net_path = _resolve_vit_net_path(self.config.model_path, "single_image_preproc_1280x720")
		self.llava_ov_vit[0].vit_mode = ov_vit_mode.VIT_SINGLE_IMG_MODE
		self.llava_ov_vit[0].max_img_num = 1
		self.llava_ov_vit[0].vit_net_fn = ctypes.c_char_p(vit_net_path.encode('utf-8'))

		vit_net_path = _resolve_vit_net_path(self.config.model_path, "multi_image_self_contained")
		self.llava_ov_vit[1].vit_mode = ov_vit_mode.VIT_MULTI_IMG_MODE
		self.llava_ov_vit[1].max_img_num = 8
		self.llava_ov_vit[1].vit_net_fn = ctypes.c_char_p(vit_net_path.encode('utf-8'))

		vit_net_path = _resolve_vit_net_path(self.config.model_path, "video_mode_self_contained")
		self.llava_ov_vit[2].vit_mode = ov_vit_mode.VIT_VIDEO_MODE
		self.llava_ov_vit[2].max_img_num = 64
		self.llava_ov_vit[2].vit_net_fn = ctypes.c_char_p(vit_net_path.encode('utf-8'))

		logger.debug("llava_onevision config:\n")
		for i in range(vit_net_num):
			logger.debug(f"  [{i}]: mode: {self.llava_ov_vit[i].vit_mode}, "
				f"net_fn: {self.llava_ov_vit[i].vit_net_fn}, "
				f"max_img_num: {self.llava_ov_vit[i].max_img_num}")

	def infer_model_init(self,
		model_type: Optional[str] = None,
		is_embed_model :Optional[bool] = None
	):
		shepd_cfg = self.shepd_cfg
		shepd_cfg.model_path = ctypes.c_char_p(self.config.model_path.encode('utf-8'))
		shepd_cfg.batch_size = self.config.batch_size \
			if self.config.batch_size != 0 else inc.SHEPD_MAX_BATCH_SIZE
		shepd_cfg.max_user_num = self.config.max_user_num

		# Remote transport is owned by libcrt device context (device_id).
		shepd_cfg.device.device_type = inc.shepd_device_type_t.SHEPD_DEVICE_LOCAL

		# Match test_crt_shpd_vlm --no-weight-pack. Host falls back to
		# per-file upload when weights.gpack is absent.
		no_weight_pack = os.getenv("AMBA_EXT_NO_WEIGHT_PACK", "0") not in (
			"0", "", "false", "False")
		if no_weight_pack:
			shepd_cfg.packed_model = 0
			logger.info("Weight pack disabled; using per-file weight upload")
		else:
			shepd_cfg.packed_model = 1

		if model_type is not None:
			if model_type == LLAVA_OV_MODEL_TYPE_NAME:
				shepd_cfg.extra_type = inc.shepd_extra_type_t.EXTRA_TYPE_LLAVA_ONEVISION
				self.__infer_model_llava_ov_init()
			elif model_type == LLAVA_MODEL_TYPE_NAME:
				shepd_cfg.extra_type = inc.shepd_extra_type_t.EXTRA_TYPE_LLAVA
				self.__infer_model_llava_init()
			elif model_type == VLM_MODEL_TYPE_NAME:
				shepd_cfg.extra_type = inc.shepd_extra_type_t.EXTRA_TYPE_VLM
				self.__infer_model_vlm_init()
			else:
				logger.error(f"Unsupported model type: {model_type}")
		else:
			shepd_cfg.extra_type = inc.shepd_extra_type_t.EXTRA_TYPE_NONE
			logger.info("No extra config")

		if is_embed_model is not None and is_embed_model == True:
			shepd_cfg.query_last_hidden = 1

		model_handle = self._shpd.crt_shpd_model_create(self.dev_ctx, ctypes.byref(shepd_cfg))
		if model_handle is None:
			raise ValueError("[crt_shpd_model_create] fail")

		self.config.max_sequence_length = shepd_cfg.max_seq_length
		self.config.eos_token_id = shepd_cfg.eos_token_id
		self.config.vocab_size = shepd_cfg.vocab_size

		logger.info(f"[crt_shpd_model_create]:\n"
			f"  model_path: {self.config.model_path}\n"
			f"  batch_size: {shepd_cfg.batch_size}\n"
			f"  max_user_num: {self.config.max_user_num}\n"
			f"  device_type: {self.config.device_type}\n"
			f"  max_sequence_length: {self.config.max_sequence_length}\n"
			f"  vocab_size: {self.config.vocab_size}\n"
			f"  eos_token_id: {self.config.eos_token_id}\n")

		return model_handle

	def infer_model_release(self, model_handle):
		rval = self._shpd.crt_shpd_model_release(model_handle)
		if rval < 0:
			raise ValueError(f"[crt_shpd_model_release] fail, rval: {rval}")

	def infer_user_init(self, model_handle):
		user_handle = self._shpd.crt_shpd_user_create(model_handle)
		if user_handle is None:
			raise ValueError(f"[crt_shpd_user_create]: "
				f"model: {model_handle}, user_ctx: {user_handle}")
		return user_handle

	def infer_user_encode(self, model_handle, user_handle, input_text, no_sys_prompt):
		enc_cfg = inc.tokenizer_enc_cfg()
		id_list = inc.token_id_list()
		enc_cfg.no_sys_prompt = no_sys_prompt
		in_text_ctype = ctypes.c_char_p(input_text.encode('utf-8'))
		rval = self._shpd.crt_shpd_user_tokenizer_encode(
			model_handle, user_handle, enc_cfg, in_text_ctype, id_list)
		if rval < 0:
			raise ValueError(f"[crt_shpd_user_preprocess] fail, rval: {rval}")

		id_list_u32_arr = None
		if id_list.num:
			id_list_u32_p = ctypes.cast(id_list.ids, ctypes.POINTER(ctypes.c_uint32))
			id_list_u32_arr = np.ctypeslib.as_array(id_list_u32_p, shape = (id_list.num,))

		return id_list_u32_arr

	def infer_user_decode(self, model_handle, user_handle, input_ids, ids_num):
		dec_cfg = inc.tokenizer_dec_cfg()
		dec_res = inc.tokenizer_dec_res()
		dec_cfg.no_piece_mode = 1
		rval = self._shpd.crt_shpd_user_tokenizer_decode(
			model_handle, user_handle, dec_cfg, input_ids, ids_num, ctypes.byref(dec_res))
		if rval < 0:
			raise ValueError(f"[infer_user_decode] fail, rval: {rval}")

		b_text = dec_res.text[:dec_res.len]
		text = b_text.decode("utf-8", errors="ignore")
		return text

	def infer_user_preprocess(self, model_handle, user_handle, model_type, vit_mode,
		input_text, img_data, img_num
	):
		vit_img_data_ctype, vit_img_size = self.__load_vit_img_ctype(img_data)

		if model_type == LLAVA_OV_MODEL_TYPE_NAME:
			if vit_mode > ov_vit_mode.VIT_VIDEO_MODE:
				raise ValueError(f"unsupported vit mode ({vit_mode}), should be "
					f"single_image: {ov_vit_mode.VIT_SINGLE_IMG_MODE}, "
					f"multi_image: {ov_vit_mode.VIT_MULTI_IMG_MODE}, "
					f"video: {ov_vit_mode.VIT_VIDEO_MODE}")
			self.shepd_cfg.shepd_ex.llava_onevision_ex.index = vit_mode
			self.llava_ov_vit[vit_mode].vit_in.img_num = img_num
			self.llava_ov_vit[vit_mode].vit_in.internal_cavalry_mem = 1
			self.llava_ov_vit[vit_mode].vit_in.img_mem.size = vit_img_size
			self.llava_ov_vit[vit_mode].vit_in.img_mem.virt = ctypes.cast(vit_img_data_ctype, ctypes.c_void_p)
		elif model_type == LLAVA_MODEL_TYPE_NAME:
			self.shepd_cfg.shepd_ex.llava_ex.vit_in.img_num = img_num
			self.shepd_cfg.shepd_ex.llava_ex.vit_in.img_mem.size = vit_img_size
			self.shepd_cfg.shepd_ex.llava_ex.vit_in.img_mem.virt = ctypes.cast(vit_img_data_ctype, ctypes.c_void_p)
		elif model_type == VLM_MODEL_TYPE_NAME:
			if vit_mode > vlm_vit_mode.VLM_VIT_AUDIO:
				raise ValueError(f"unsupported vlm vit mode ({vit_mode})")
			net_index = self._vlm_net_index_for_mode(vit_mode)
			self.shepd_cfg.shepd_ex.vlm_ex.index = net_index
			self.vlm_vit[net_index].vit_in.img_num = img_num
			self.vlm_vit[net_index].vit_in.internal_cavalry_mem = 1
			self.vlm_vit[net_index].vit_in.img_mem.size = vit_img_size
			self.vlm_vit[net_index].vit_in.img_mem.virt = ctypes.cast(vit_img_data_ctype, ctypes.c_void_p)
		else:
			logger.error(f"infer_user_preprocess: unsupported model_type: {model_type}")

		preproc_res = inc.preproc_res()
		in_text_ctype = ctypes.c_char_p(input_text.encode('utf-8')) if input_text is not None else None
		rval = self._shpd.crt_shpd_user_preprocess(
			model_handle, user_handle, in_text_ctype, self.shepd_cfg.shepd_ex, preproc_res)
		logger.debug(f"infer_user_preprocess: "
			f"vit_mode: {vit_mode}, img_num: {img_num}, vit_img_size: {vit_img_size}, "
			f"out_ids_num: {preproc_res.id_list.num}")
		if rval < 0:
			raise ValueError(f"[crt_shpd_user_preprocess] fail, rval: {rval}")

		input_ids_u32_arr = None
		if preproc_res.id_list.num:
			input_ids_u32_p = ctypes.cast(preproc_res.id_list.ids, ctypes.POINTER(ctypes.c_uint32))
			input_ids_u32_arr = np.ctypeslib.as_array(input_ids_u32_p, shape = (preproc_res.id_list.num,))

		return input_ids_u32_arr

	def infer_user_run_ids(self, model_handle, user_handle, input_ids, ids_num):
		run_cfg = inc.shepd_run_cfg()
		run_cfg.sample_hw_type = inc.shepd_sample_hw_t.SAMPLER_HW_TYPE_ARM
		output = inc.shepd_output()
		rval = self._shpd.crt_shpd_user_run_ids(
			model_handle, user_handle, ctypes.byref(run_cfg), input_ids, ids_num, ctypes.byref(output))
		if rval < 0:
			raise ValueError(f"[crt_shpd_user_run_ids] fail, rval: {rval}")

		return output.token_id, output.pos

	def infer_user_run_logits(self, model_handle, user_handle, input_ids, ids_num, token_id):
		run_cfg = inc.shepd_run_cfg()
		run_cfg.sample_hw_type = inc.shepd_sample_hw_t.SAMPLER_HW_TYPE_NONE
		run_cfg.query_logits_en = 1
		run_cfg.force_token_id = token_id
		output = inc.shepd_output()
		rval = self._shpd.crt_shpd_user_run_ids(
			model_handle, user_handle, ctypes.byref(run_cfg), input_ids, ids_num, ctypes.byref(output))
		if rval < 0:
			raise ValueError(f"[crt_shpd_user_run_ids] fail, rval: {rval}")

		logits_u8_p = ctypes.cast(output.logits_virt, ctypes.POINTER(ctypes.c_uint8))
		logits_u8_arr = np.ctypeslib.as_array(logits_u8_p, shape = (output.logits_mem_size,))
		logits_fp16_arr = np.frombuffer(logits_u8_arr.data, np.float16)
		return logits_fp16_arr.reshape(1, -1).astype(np.float32), output.pos

	def infer_user_run_embeddings(self, model_handle, user_handle, input_ids, ids_num):
		run_cfg = inc.shepd_run_cfg()
		output = inc.shepd_output()
		rval = self._shpd.crt_shpd_user_run_ids(
			model_handle, user_handle, ctypes.byref(run_cfg), input_ids, ids_num, ctypes.byref(output))
		if rval < 0:
			raise ValueError(f"[infer_user_run_embeddings] fail, rval: {rval}")

		embed_u8_p = ctypes.cast(output.last_hidden_virt, ctypes.POINTER(ctypes.c_uint8))
		embed_u8_arr = np.ctypeslib.as_array(embed_u8_p, shape = (output.last_hidden_mem_size,))
		embed_fp16_arr = np.frombuffer(embed_u8_arr.data, np.float16)
		hidden_size_fp16 = output.last_hidden_elem_size // np.float16().itemsize
		return embed_fp16_arr.reshape(-1, hidden_size_fp16)

	def infer_user_release(self, model_handle, user_handle):
		rval = self._shpd.crt_shpd_user_release(model_handle, user_handle)
		if rval < 0:
			raise ValueError(f"[crt_shpd_user_release] fail, rval: {rval}")

	def infer_input_text_cvt(self, input_text):
		return ctypes.c_char_p(input_text.encode('utf-8'))

	def infer_input_token_cvt(self, input_token):
		return ctypes.c_void_p(input_token)

	def infer_user_reset(self, model_handle, user_handle):
		reset_cfg = inc.shepd_reset_cfg()
		reset_cfg.reset_type = inc.shepd_reset_type_t.RESET_TYPE_HARD
		rval = self._shpd.crt_shpd_user_reset(
			model_handle, user_handle, ctypes.byref(reset_cfg))
		if rval < 0:
			raise ValueError(f"[crt_shpd_user_reset] fail, type: {reset_cfg.reset_type}, rval: {rval}")

		return reset_cfg.reset_pos

	def infer_at_eos(
		self,
		token_id: np.uint32,
	):
		return (token_id == self.config.eos_token_id)

	def infer_at_end(
		self,
		pos: np.uint32,
		max_length: np.uint32
	):
		return (pos >= max_length)

	def map_vit_mode_for_llava_ov(self, mode):
		if mode == vit_mode.SINGLE:
			return ov_vit_mode.VIT_SINGLE_IMG_MODE
		elif mode == vit_mode.MULTI:
			return ov_vit_mode.VIT_MULTI_IMG_MODE
		elif mode == vit_mode.VIDEO:
			return ov_vit_mode.VIT_VIDEO_MODE
		else:
			raise ValueError(f"Unsupported vit_mode: {mode} for llava_ov")

	def map_vit_mode_for_vlm(self, mode):
		if mode == vit_mode.SINGLE:
			return vlm_vit_mode.VLM_VIT_IMAGE
		elif (mode == vit_mode.MULTI) or (mode == vit_mode.VIDEO):
			return vlm_vit_mode.VLM_VIT_VIDEO
		elif mode == vit_mode.AUDIO:
			return vlm_vit_mode.VLM_VIT_AUDIO
		else:
			raise ValueError(f"Unsupported vit_mode: {mode} for vlm")
