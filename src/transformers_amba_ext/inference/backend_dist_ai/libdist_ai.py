

import ctypes
import ctypes.util
from ...inference.backend_dist_ai import libdist_ai_inc as inc

class libdist_ai_api():
	def __init__(self, lib_name):
		lib_path = ctypes.util.find_library(lib_name)
		if lib_path is None:
			raise ValueError(f"lib_path is None with lib_name: {lib_name}, please export it.")
		self.dist_ai_api = ctypes.cdll.LoadLibrary(lib_path)

	def dist_ai_init(self, econfig):
		self.dist_ai_api.dist_ai_init.argtypes = [ctypes.POINTER(inc.dist_ai_ext_config_t)]
		self.dist_ai_api.dist_ai_init.restype = ctypes.c_int
		return self.dist_ai_api.dist_ai_init(econfig)

	def dist_ai_exit(self):
		self.dist_ai_api.dist_ai_exit.argtypes = []
		self.dist_ai_api.dist_ai_exit.restype = ctypes.c_int
		return self.dist_ai_api.dist_ai_exit()

	def dist_ai_model_create(self, econfig):
		self.dist_ai_api.dist_ai_model_creat.argtypes = [ctypes.POINTER(inc.dist_ai_ext_config_t)]
		self.dist_ai_api.dist_ai_model_creat.restype = ctypes.c_void_p
		return self.dist_ai_api.dist_ai_model_creat(econfig)

	def dist_ai_model_release(self, mhandle):
		self.dist_ai_api.dist_ai_model_release.argtypes = [ctypes.c_void_p]
		self.dist_ai_api.dist_ai_model_release.restype = ctypes.c_int
		return self.dist_ai_api.dist_ai_model_release(mhandle)

	def dist_ai_user_create(self, mhandle):
		self.dist_ai_api.dist_ai_user_creat.argtypes = [ctypes.c_void_p]
		self.dist_ai_api.dist_ai_user_creat.restype = ctypes.c_void_p
		return self.dist_ai_api.dist_ai_user_creat(mhandle)

	def dist_ai_user_release(self, mhandle, uhandle):
		self.dist_ai_api.dist_ai_user_release.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
		self.dist_ai_api.dist_ai_user_release.restype = ctypes.c_int
		return self.dist_ai_api.dist_ai_user_release(mhandle, uhandle)

	def dist_ai_user_tokenizer_encode(self, mhandle, uhandle, enc_cfg, text, id_list):
		raise NotImplementedError("dist_ai_user_tokenizer_encode is not implemented yet.")
		# self.dist_ai_api.dist_ai_user_tokenizer_encode.argtypes = [
		# 	ctypes.c_void_p,
		# 	ctypes.c_void_p,
		# 	ctypes.POINTER(inc.tokenizer_enc_cfg),
		# 	ctypes.c_char_p,
		# 	ctypes.POINTER(inc.token_id_list)
		# ]
		# self.dist_ai_api.dist_ai_user_tokenizer_encode.restype = ctypes.c_int
		# return self.dist_ai_api.dist_ai_user_tokenizer_encode(
		# 	handle, user_ctx, enc_cfg, text, id_list)

	def dist_ai_user_tokenizer_decode(self, mhandle, uhandle, dec_cfg, ids, ids_num, dec_res):
		raise NotImplementedError("dist_ai_user_tokenizer_decode is not implemented yet.")
		# self.dist_ai_api.dist_ai_user_tokenizer_decode.argtypes = [
		# 	ctypes.c_void_p,
		# 	ctypes.c_void_p,
		# 	ctypes.POINTER(inc.tokenizer_dec_cfg),
		# 	ctypes.c_void_p,
		# 	ctypes.c_uint32,
		# 	ctypes.POINTER(inc.tokenizer_dec_res)
		# ]
		# self.dist_ai_api.dist_ai_user_tokenizer_decode.restype = ctypes.c_int
		# return self.dist_ai_api.dist_ai_user_tokenizer_decode(
		# 	handle, user_ctx, dec_cfg, ids, ids_num, dec_res)

	def dist_ai_user_preprocess(self, handle, user_ctx, input_text, shep_ex, preproc_res):
		raise NotImplementedError("dist_ai_user_preprocess is not implemented yet.")
		# self.dist_ai_api.dist_ai_user_preprocess.argtypes = [
		# 	ctypes.c_void_p,
		# 	ctypes.c_void_p,
		# 	ctypes.c_char_p,
		# 	ctypes.POINTER(inc.shepd_extra),
		# 	ctypes.POINTER(inc.preproc_res)
		# ]
		# self.dist_ai_api.dist_ai_user_preprocess.restype = ctypes.c_int
		# return self.dist_ai_api.dist_ai_user_preprocess(
		# 	handle, user_ctx, input_text, shep_ex, preproc_res)

	def dist_ai_user_run_ids(
			self, mhandle, uhandle, input_ids, ids_num, run_cfg, out):
		self.dist_ai_api.dist_ai_user_run_ids.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_uint32,
			ctypes.POINTER(inc.dist_ai_run_config_t),
			ctypes.POINTER(inc.dist_ai_out_t)
		]
		self.dist_ai_api.dist_ai_user_run_ids.restype = ctypes.c_int
		return self.dist_ai_api.dist_ai_user_run_ids(
			mhandle, uhandle, input_ids, ids_num, run_cfg, out)

	def dist_ai_user_reset(self, uhandle, reset_cfg):
		self.dist_ai_api.dist_ai_user_reset.argtypes = [
			ctypes.c_void_p,
			ctypes.POINTER(inc.dist_ai_reset_cfg_t),
		]
		self.dist_ai_api.dist_ai_user_reset.restype = ctypes.c_int
		return self.dist_ai_api.dist_ai_user_reset(uhandle, reset_cfg)

	def dist_ai_get_version(self, ver):
		self.dist_ai_api.dist_ai_get_version.argtypes = [ctypes.POINTER(inc.dist_ai_version)]
		self.dist_ai_api.dist_ai_get_version.restype = ctypes.c_int
		return self.dist_ai_api.dist_ai_get_version(ver)

	def dist_ai_user_bcast_ids(self, mhandle, uhandle, bcast_data, ids_list):
		self.dist_ai_api.dist_ai_user_bcast_ids.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(inc.dist_ai_bcast_t),
			ctypes.c_void_p
		]
		self.dist_ai_api.dist_ai_user_bcast_ids.restype = ctypes.c_int
		return self.dist_ai_api.dist_ai_user_bcast_ids(
			mhandle, uhandle, bcast_data, ids_list)

