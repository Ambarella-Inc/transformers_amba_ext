import ctypes
import numpy as np
from typing import Dict, Any, Optional

from ..inference import libshepd as libshepd
from ..inference import infer_configuration as config
from ..inference import libshepd_inc as inc
from ..inference.infer_configuration import ov_vit_mode
from ..inference.infer_configuration import LLAVA_OV_MODEL_TYPE_NAME, LLAVA_MODEL_TYPE_NAME
from ..utils import logging

logger = logging.get_logger(__name__)

class inference_runtime(libshepd.libshepd_api):
	def __init__(self, config: config.infer_config):
		self.config = config

		""" The shepd_cfg can't be defined as a local variable in function
			since the pointer of model_path was used by libshepd
		"""
		self.shepd_cfg = inc.shepd_config()
		super().__init__(self.config.lib_name)

	def __load_vit_img_ctype(self, img_data):
		image = img_data.astype(np.uint8).reshape(-1)
		vit_img_size = image.shape[-1]
		vit_img_ctype = (ctypes.c_uint8 * vit_img_size)(*image)
		return vit_img_ctype, vit_img_size

	def infer_get_version(self):
		shepd_ver = inc.shepherd_version()
		rval = super().shepherd_get_version(ctypes.byref(shepd_ver))
		logger.debug("[shepd ver] version: %u.%u.%u, mod_time: 0x%x, rval: %u" % (
			shepd_ver.major, shepd_ver.minor, shepd_ver.patch, shepd_ver.mod_time, rval))
		version = "%u.%u.%u" % (shepd_ver.major, shepd_ver.minor, shepd_ver.patch)
		return version

	def infer_init(self):
		init_cfg = inc.shepd_init_cfg()
		init_cfg.log_level = self.config.lib_log_level
		rval = super().shepherd_init(ctypes.byref(init_cfg))
		if rval < 0:
			raise ValueError(f"[shepherd_init] fail, rval: {rval}")

	def infer_exit(self):
		rval = super().shepherd_exit()
		if rval < 0:
			raise ValueError(f"[shepherd_exit] fail, rval: {rval}")

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

	def __infer_model_llava_ov_init(self):
		vit_net_num = 3
		self.llava_ov_vit = (inc.llava_onevision_vit * vit_net_num)()
		self.shepd_cfg.shepd_ex.llava_onevision_ex.size = vit_net_num
		self.shepd_cfg.shepd_ex.llava_onevision_ex.vit = ctypes.cast(self.llava_ov_vit, ctypes.c_void_p)

		vit_net_path = f"{self.config.model_path}/single_image_preproc_1280x720_n1_cavalry.bin"
		self.llava_ov_vit[0].vit_mode = ov_vit_mode.VIT_SINGLE_IMG_MODE
		self.llava_ov_vit[0].max_img_num = 1
		self.llava_ov_vit[0].vit_net_fn = ctypes.c_char_p(vit_net_path.encode('utf-8'))

		vit_net_path = f"{self.config.model_path}/multi_image_self_contained_fp16_n1_cavalry.bin"
		self.llava_ov_vit[1].vit_mode = ov_vit_mode.VIT_MULTI_IMG_MODE
		self.llava_ov_vit[1].max_img_num = 8
		self.llava_ov_vit[1].vit_net_fn = ctypes.c_char_p(vit_net_path.encode('utf-8'))

		vit_net_path = f"{self.config.model_path}/video_mode_self_contained_fp16_n1_cavalry.bin"
		self.llava_ov_vit[2].vit_mode = ov_vit_mode.VIT_VIDEO_MODE
		self.llava_ov_vit[2].max_img_num = 16
		self.llava_ov_vit[2].vit_net_fn = ctypes.c_char_p(vit_net_path.encode('utf-8'))

		logger.debug("llava_onevision config:\n")
		for i in range(vit_net_num):
			logger.debug(f"  [{i}]: mode: {self.llava_ov_vit[i].vit_mode}, "
				f"net_fn: {self.llava_ov_vit[i].vit_net_fn}, "
				f"max_img_num: {self.llava_ov_vit[i].max_img_num}")

	def infer_model_init(self, model_type: Optional[str] = None):
		shepd_cfg = self.shepd_cfg
		shepd_cfg.model_path = ctypes.c_char_p(self.config.model_path.encode('utf-8'))
		shepd_cfg.batch_size = self.config.batch_size
		shepd_cfg.max_user_num = self.config.max_user_num

		shepd_cfg.device.device_type = self.config.device_type
		if self.config.device_type == inc.shepd_device_type_t.SHEPD_DEVICE_REMOTE:
			shepd_cfg.device.device_ip = ctypes.c_char_p(self.config.device_ip.encode('utf-8'))
			shepd_cfg.device.device_port = self.config.device_port

		if model_type is not None:
			if model_type == LLAVA_OV_MODEL_TYPE_NAME:
				shepd_cfg.extra_type = inc.shepd_extra_type_t.EXTRA_TYPE_LLAVA_ONEVISION
				self.__infer_model_llava_ov_init()
			elif model_type == LLAVA_MODEL_TYPE_NAME:
				shepd_cfg.extra_type = inc.shepd_extra_type_t.EXTRA_TYPE_LLAVA
				self.__infer_model_llava_init()
			else:
				logger.error(f"Unsupported model type: {model_type}")
		else:
			shepd_cfg.extra_type = inc.shepd_extra_type_t.EXTRA_TYPE_NONE
			logger.info("No extra config")

		model_handle = super().shepherd_model_create(ctypes.byref(shepd_cfg))
		if model_handle is None:
			raise ValueError("[shepherd_model_create] fail")

		self.config.max_sequence_length = shepd_cfg.max_seq_length
		self.config.eos_token_id = shepd_cfg.eos_token_id
		self.config.vocab_size = shepd_cfg.vocab_size

		logger.info(f"[shepherd_model_create]:\n"
			f"  model_path: {self.config.model_path}\n"
			f"  batch_size: {self.config.batch_size}\n"
			f"  max_user_num: {self.config.max_user_num}\n"
			f"  device_type: {self.config.device_type}\n"
			f"  max_sequence_length: {self.config.max_sequence_length}\n"
			f"  vocab_size: {self.config.vocab_size}\n"
			f"  eos_token_id: {self.config.eos_token_id}\n")

		return model_handle

	def infer_model_release(self, model_handle):
		rval = super().shepherd_model_release(model_handle)
		if rval < 0:
			raise ValueError(f"[shepherd_model_release] fail, rval: {rval}")

	def infer_user_init(self, model_handle):
		user_handle = super().shepherd_user_create(model_handle)
		if user_handle is None:
			raise ValueError(f"[shepherd_user_create]: "
				f"model: {model_handle}, user_ctx: {user_handle}")
		return user_handle

	def infer_user_encode(self, model_handle, user_handle, input_text):
		enc_cfg = inc.tokenizer_enc_cfg()
		id_list = inc.token_id_list()
		in_text_ctype = ctypes.c_char_p(input_text.encode('utf-8'))
		rval = super().shepherd_user_tokenizer_encode(
			model_handle, user_handle, enc_cfg, in_text_ctype, id_list)
		if rval < 0:
			raise ValueError(f"[shepherd_user_preprocess] fail, rval: {rval}")

		id_list_u32_arr = None
		if id_list.num:
			id_list_u32_p = ctypes.cast(id_list.ids, ctypes.POINTER(ctypes.c_uint32))
			id_list_u32_arr = np.ctypeslib.as_array(id_list_u32_p, shape = (id_list.num,))

		return id_list_u32_arr

	def infer_user_decode(self):
		raise NotImplementedError("Unsupported for now")

	def infer_user_preprocess(self, model_handle, user_handle, model_type, vit_mode, input_text, img_data, img_num):
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
		else:
			logger.error(f"infer_user_preprocess: unsupported model_type: {model_type}")

		preproc_res = inc.preproc_res()
		in_text_ctype = ctypes.c_char_p(input_text.encode('utf-8')) if input_text is not None else None
		rval = super().shepherd_user_preprocess(
			model_handle, user_handle, in_text_ctype, self.shepd_cfg.shepd_ex, preproc_res)
		logger.debug(f"infer_user_preprocess: "
			f"vit_mode: {vit_mode}, img_num: {img_num}, vit_img_size: {vit_img_size}, "
			f"out_ids_num: {preproc_res.id_list.num}")
		if rval < 0:
			raise ValueError(f"[shepherd_user_preprocess] fail, rval: {rval}")

		input_ids_u32_arr = None
		if preproc_res.id_list.num:
			input_ids_u32_p = ctypes.cast(preproc_res.id_list.ids, ctypes.POINTER(ctypes.c_uint32))
			input_ids_u32_arr = np.ctypeslib.as_array(input_ids_u32_p, shape = (preproc_res.id_list.num,))

		return input_ids_u32_arr

	def infer_user_run_ids(self, model_handle, user_handle, input_ids, ids_num):
		run_cfg = inc.shepd_run_cfg()
		run_cfg.sample_hw_type = inc.shepd_sample_hw_t.SAMPLER_HW_TYPE_ARM
		output = inc.shepd_output()
		rval = super().shepherd_user_run_ids(
			model_handle, user_handle, ctypes.byref(run_cfg), input_ids, ids_num, ctypes.byref(output))
		if rval < 0:
			raise ValueError(f"[shepherd_user_context_run_ids] fail, rval: {rval}")

		return output.token_id, output.pos

	def infer_user_run_logits(self, model_handle, user_handle, input_ids, ids_num, token_id):
		run_cfg = inc.shepd_run_cfg()
		run_cfg.sample_hw_type = inc.shepd_sample_hw_t.SAMPLER_HW_TYPE_NONE
		run_cfg.query_logits_en = 1
		run_cfg.force_token_id = token_id
		output = inc.shepd_output()
		rval = super().shepherd_user_run_ids(
			model_handle, user_handle, ctypes.byref(run_cfg), input_ids, ids_num, ctypes.byref(output))
		if rval < 0:
			raise ValueError(f"[shepherd_user_context_run_ids] fail, rval: {rval}")

		logits_u8_p = ctypes.cast(output.logits_virt, ctypes.POINTER(ctypes.c_uint8))
		logits_u8_arr = np.ctypeslib.as_array(logits_u8_p, shape = (output.logits_mem_size,))
		logits_fp16_arr = np.frombuffer(logits_u8_arr.data, np.float16)
		return logits_fp16_arr.reshape(1, -1).astype(np.float32), output.pos

	def infer_user_release(self, model_handle, user_handle):
		rval = super().shepherd_user_release(model_handle, user_handle)
		if rval < 0:
			raise ValueError(f"[shepherd_user_release] fail, rval: {rval}")

	def infer_input_text_cvt(self, input_text):
		return ctypes.c_char_p(input_text.encode('utf-8'))

	def infer_input_token_cvt(self, input_token):
		return ctypes.c_void_p(input_token)

	def infer_user_reset(self, model_handle, user_handle):
		reset_cfg = inc.shepd_reset_cfg()
		reset_cfg.reset_type = inc.shepd_reset_type_t.RESET_TYPE_HARD
		rval = super().shepherd_user_reset(
			model_handle, user_handle, ctypes.byref(reset_cfg))
		if rval < 0:
			raise ValueError(f"[shepherd_user_reset] fail, type: {reset_cfg.reset_type}, rval: {rval}")

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

