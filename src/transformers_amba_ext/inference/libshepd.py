

import ctypes
import ctypes.util
from ..inference import libshepd_inc as inc

class libshepd_api():
	def __init__(self, lib_name):
		lib_path = ctypes.util.find_library(lib_name)
		if lib_path is None:
			raise ValueError(f"lib_path is None with lib_name: {lib_name}, please export it.")
		self.shepd_api = ctypes.cdll.LoadLibrary(lib_path)

	def shepherd_init(self, init_cfg):
		self.shepd_api.shepherd_init.argtypes = [ctypes.POINTER(inc.shepd_init_cfg)]
		self.shepd_api.shepherd_init.restype = ctypes.c_int
		return self.shepd_api.shepherd_init(init_cfg)

	def shepherd_model_create(self, shepd_cfg):
		self.shepd_api.shepherd_model_create.argtypes = [ctypes.POINTER(inc.shepd_config)]
		self.shepd_api.shepherd_model_create.restype = ctypes.c_void_p
		return self.shepd_api.shepherd_model_create(shepd_cfg)

	def shepherd_model_release(self, handle):
		self.shepd_api.shepherd_model_release.argtypes = [ctypes.c_void_p]
		self.shepd_api.shepherd_model_release.restype = ctypes.c_int
		return self.shepd_api.shepherd_model_release(handle)

	def shepherd_user_create(self, handle):
		self.shepd_api.shepherd_user_create.argtypes = [ctypes.c_void_p]
		self.shepd_api.shepherd_user_create.restype = ctypes.c_void_p
		return self.shepd_api.shepherd_user_create(handle)

	def shepherd_user_tokenizer_encode(self, handle, user_ctx, enc_cfg, text, id_list):
		self.shepd_api.shepherd_user_tokenizer_encode.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(inc.tokenizer_enc_cfg),
			ctypes.c_char_p,
			ctypes.POINTER(inc.token_id_list)
		]
		self.shepd_api.shepherd_user_tokenizer_encode.restype = ctypes.c_int
		return self.shepd_api.shepherd_user_tokenizer_encode(
			handle, user_ctx, enc_cfg, text, id_list)

	def shepherd_user_tokenizer_decode(self, handle, user_ctx, dec_cfg, ids, ids_num, dec_res):
		self.shepd_api.shepherd_user_tokenizer_decode.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(inc.tokenizer_dec_cfg),
			ctypes.c_void_p,
			ctypes.c_uint32,
			ctypes.POINTER(inc.tokenizer_dec_res)
		]
		self.shepd_api.shepherd_user_tokenizer_decode.restype = ctypes.c_int
		return self.shepd_api.shepherd_user_tokenizer_decode(
			handle, user_ctx, dec_cfg, ids, ids_num, dec_res)

	def shepherd_user_preprocess(self, handle, user_ctx, input_text, shep_ex, preproc_res):
		self.shepd_api.shepherd_user_preprocess.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_char_p,
			ctypes.POINTER(inc.shepd_extra),
			ctypes.POINTER(inc.preproc_res)
		]
		self.shepd_api.shepherd_user_preprocess.restype = ctypes.c_int
		return self.shepd_api.shepherd_user_preprocess(
			handle, user_ctx, input_text, shep_ex, preproc_res)

	def shepherd_user_run_ids(
			self, handle, user_ctx, run_cfg, input_ids, ids_num, shepd_out):
		self.shepd_api.shepherd_user_run_ids.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(inc.shepd_run_cfg),
			ctypes.c_void_p,
			ctypes.c_uint32,
			ctypes.POINTER(inc.shepd_output)
		]
		self.shepd_api.shepherd_user_run_ids.restype = ctypes.c_int
		return self.shepd_api.shepherd_user_run_ids(
			handle, user_ctx, run_cfg, input_ids, ids_num, shepd_out)

	def shepherd_user_reset(self, handle, user_ctx, reset_cfg):
		self.shepd_api.shepherd_user_reset.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(inc.shepd_reset_cfg),
		]
		self.shepd_api.shepherd_user_reset.restype = ctypes.c_int
		return self.shepd_api.shepherd_user_reset(
			handle, user_ctx, reset_cfg)

	def shepherd_user_release(self, handle, user_ctx):
		self.shepd_api.shepherd_user_release.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
		self.shepd_api.shepherd_user_release.restype = ctypes.c_int
		return self.shepd_api.shepherd_user_release(handle, user_ctx)

	def shepherd_user_dump(self, handle, user_ctx, path, dump_cfg):
		self.shepd_api.shepherd_user_dump.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_char_p,
			ctypes.POINTER(inc.shepd_dump_cfg),
		]
		self.shepd_api.shepherd_user_dump.restype = ctypes.c_int
		return self.shepd_api.shepherd_user_dump(handle, user_ctx, path, dump_cfg)

	def shepherd_user_perf(self, handle, user_ctx, perf_cfg):
		self.shepd_api.shepherd_user_perf.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(inc.shepd_perf_cfg),
		]
		self.shepd_api.shepherd_user_perf.restype = ctypes.c_int
		return self.shepd_api.shepherd_user_perf(handle, user_ctx, perf_cfg)

	def shepherd_get_version(self, ver):
		self.shepd_api.shepherd_get_version.argtypes = [ctypes.POINTER(inc.shepherd_version)]
		self.shepd_api.shepherd_get_version.restype = ctypes.c_int
		return self.shepd_api.shepherd_get_version(ver)

	def shepherd_exit(self):
		self.shepd_api.shepherd_exit.argtypes = []
		self.shepd_api.shepherd_exit.restype = ctypes.c_int
		return self.shepd_api.shepherd_exit()
