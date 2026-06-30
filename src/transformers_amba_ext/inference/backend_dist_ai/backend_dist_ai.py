import ctypes
import numpy as np
from typing import Dict, Any, Optional

from ...inference.backend_dist_ai import __version__
from ...inference.backend_dist_ai import libdist_ai as libdist_ai
from ...inference.backend_dist_ai import libdist_ai_inc as inc
from ...inference import infer_configuration as config
from ...utils import logging

logger = logging.get_logger(__name__)

class infer_dist_ai(libdist_ai.libdist_ai_api):
	def __init__(self, config: config.infer_config):
		self.config = config
		self.dist_ai_econfig = inc.dist_ai_ext_config_t()
		self.is_reset = 0

		super().__init__(self.config.lib_name)

	def infer_get_version(self):
		dist_ai_ver = inc.dist_ai_version()
		rval = super().dist_ai_get_version(ctypes.byref(dist_ai_ver))
		logger.debug("[dist_ai ver] version: %u.%u.%u, mod_time: 0x%x, rval: %u" % (
			dist_ai_ver.major, dist_ai_ver.minor, dist_ai_ver.patch, dist_ai_ver.mod_time, rval))
		version = "%u.%u.%u" % (dist_ai_ver.major, dist_ai_ver.minor, dist_ai_ver.patch)
		return version

	def _infer_version_check(self):
		ver_infer = __version__
		ver_dist_ai = self.infer_get_version()
		ver_infer_list = ver_infer.split('.')
		ver_dist_ai_list = ver_dist_ai.split('.')

		if (ver_infer_list[0] != ver_dist_ai_list[0]) or (ver_infer_list[1] != ver_dist_ai_list[1]):
			raise ValueError(
				f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				f"Wrapper API on the transformers_amba_ext package. Version: {ver_infer}\n"
				f"Backend API of dist_ai library. Version: {ver_dist_ai}\n"
				f"This mismatched version might trigger some unknown issues with incompatibility API.\n"
				f"Please contact the Ambarella team if you are unsure how to handle it.\n"
				f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

	def infer_init(self):
		self.dist_ai_econfig.log_level = self.config.lib_log_level
		self.dist_ai_econfig.rank_num = 8           #fixme with hardcode
		self.dist_ai_econfig.rank_id = 0            #fixme with hardcode
		self.dist_ai_econfig.rank_root_id = 0       #fixme with hardcode
		self.dist_ai_econfig.submod_en_bitmap = 0xf #fixme with hardcode
		self.dist_ai_econfig.model_path = self.config.model_path.encode('utf-8')
		for i in range(self.dist_ai_econfig.rank_num):
			rank_addr = f"10.4.240.{31 + i}".encode('utf-8') #fixme with hardcode
			self.dist_ai_econfig.rank_addr_table[i].value = rank_addr

		rval = super().dist_ai_init(ctypes.byref(self.dist_ai_econfig))
		if rval < 0:
			raise ValueError(f"[dist_ai_init] fail, rval: {rval}")

		self._infer_version_check()

	def infer_exit(self):
		rval = super().dist_ai_exit()
		if rval < 0:
			raise ValueError(f"[dist_ai_exit] fail, rval: {rval}")

	def infer_model_init(self,
		model_type: Optional[str] = None,
		is_embed_model :Optional[bool] = None
	):
		_, _ = model_type, is_embed_model

		mhandle = super().dist_ai_model_create(ctypes.byref(self.dist_ai_econfig))
		if mhandle is None:
			raise ValueError("[dist_ai_model_create] fail")

		self.config.max_sequence_length = self.dist_ai_econfig.max_context_len
		self.config.eos_token_id = self.dist_ai_econfig.eos_token_id

		logger.info(f"[dist_ai_model_create]:\n"
			f"  model_path: {self.config.model_path}\n"
			f"  max_sequence_length: {self.config.max_sequence_length}\n"
			f"  eos_token_id: {self.config.eos_token_id}\n")

		return mhandle

	def infer_model_release(self, mhandle):
		rval = super().dist_ai_model_release(mhandle)
		if rval < 0:
			raise ValueError(f"[dist_ai_model_release] fail, rval: {rval}")

	def infer_user_init(self, mhandle):
		uhandle = super().dist_ai_user_create(ctypes.byref(self.dist_ai_econfig))
		if uhandle is None:
			raise ValueError(f"[dist_ai_user_create]: "
				f"model: {mhandle}, user_ctx: {uhandle}")
		return uhandle

	def infer_user_encode(self, mhandle, uhandle, input_text, no_sys_prompt):
		_, _, _, _ = mhandle, uhandle, input_text, no_sys_prompt
		raise NotImplementedError("infer_user_encode is not implemented yet.")
		# enc_cfg = inc.tokenizer_enc_cfg()
		# id_list = inc.token_id_list()
		# enc_cfg.no_sys_prompt = no_sys_prompt
		# in_text_ctype = ctypes.c_char_p(input_text.encode('utf-8'))
		# rval = super().shepherd_user_tokenizer_encode(
		# 	mhandle, uhandle, enc_cfg, in_text_ctype, id_list)
		# if rval < 0:
		# 	raise ValueError(f"[shepherd_user_preprocess] fail, rval: {rval}")

		# id_list_u32_arr = None
		# if id_list.num:
		# 	id_list_u32_p = ctypes.cast(id_list.ids, ctypes.POINTER(ctypes.c_uint32))
		# 	id_list_u32_arr = np.ctypeslib.as_array(id_list_u32_p, shape = (id_list.num,))

		# return id_list_u32_arr

	def infer_user_decode(self, mhandle, uhandle, input_ids, ids_num):
		_, _, _, _ = mhandle, uhandle, input_ids, ids_num
		raise NotImplementedError("infer_user_decode is not implemented yet.")
		# dec_cfg = inc.tokenizer_dec_cfg()
		# dec_res = inc.tokenizer_dec_res()
		# dec_cfg.no_piece_mode = 1
		# rval = super().shepherd_user_tokenizer_decode(
		# 	mhandle, uhandle, dec_cfg, input_ids, ids_num, ctypes.byref(dec_res))
		# if rval < 0:
		# 	raise ValueError(f"[infer_user_decode] fail, rval: {rval}")

		# b_text = dec_res.text[:dec_res.len]
		# text = b_text.decode("utf-8", errors="ignore")
		# return text

	def infer_user_preprocess(self, mhandle, uhandle, model_type, vit_mode,
		input_text, img_data, img_num
	):
		_, _, _, _, _, _, _ = mhandle, uhandle, model_type, vit_mode, \
			input_text, img_data, img_num
		raise NotImplementedError("infer_user_preprocess is not implemented yet.")

	def infer_user_bcast_ids(self, mhandle, uhandle, input_ids, ids_num):
		bcast_data = inc.dist_ai_bcast_t()
		bcast_data.ids_num = ids_num
		bcast_data.is_reset = 0
		if self.is_reset == 1:
			bcast_data.is_reset = 1
			bcast_data.reset_type = inc.dist_ai_reset_type_t.RESET_TYPE_HARD
			bcast_data.reset_pos = 0
			self.is_reset = 0

		rval = super().dist_ai_user_bcast_ids(
			mhandle, uhandle, ctypes.byref(bcast_data), input_ids)
		if rval < 0:
			raise ValueError(f"[dist_ai_user_bcast_ids] fail, rval: {rval}")

	def infer_user_run_ids(self, mhandle, uhandle, input_ids, ids_num):
		run_cfg = inc.dist_ai_run_config_t()
		run_cfg.infer_mode = inc.dist_ai_infer_mode_t.INF_MODE_NORMAL
		run_cfg.debug_bitmap = 0
		output = inc.dist_ai_out_t()
		rval = super().dist_ai_user_run_ids(
			mhandle, uhandle, input_ids, ids_num,
			ctypes.byref(run_cfg), ctypes.byref(output))
		if rval < 0:
			raise ValueError(f"[dist_ai_user_run_ids] fail, rval: {rval}")

		return output.token, output.pos

	def infer_user_run_logits(self, mhandle, uhandle, input_ids, ids_num, token_id):
		_, _, _, _, _ = mhandle, uhandle, input_ids, ids_num, token_id
		raise NotImplementedError("infer_user_run_logits is not implemented yet.")

	def infer_user_run_embeddings(self, mhandle, uhandle, input_ids, ids_num):
		_, _, _, _ = mhandle, uhandle, input_ids, ids_num
		raise NotImplementedError("infer_user_run_embeddings is not implemented yet.")

	def infer_user_release(self, mhandle, uhandle):
		rval = super().dist_ai_user_release(mhandle, uhandle)
		if rval < 0:
			raise ValueError(f"[dist_ai_user_release] fail, rval: {rval}")

	def infer_input_text_cvt(self, input_text):
		return ctypes.c_char_p(input_text.encode('utf-8'))

	def infer_input_token_cvt(self, input_token):
		return ctypes.c_void_p(input_token)

	def infer_user_reset(self, mhandle, uhandle):
		_, _ = mhandle, uhandle

		self.is_reset = 1
		print(f"[infer_user_reset] is_reset: {self.is_reset}")
		return 0

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
