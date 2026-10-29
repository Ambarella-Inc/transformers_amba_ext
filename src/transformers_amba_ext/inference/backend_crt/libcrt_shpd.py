
import ctypes
from ...inference.backend_shepd import libshepd_inc as shepd_inc


class libcrt_shpd_api():
	"""ctypes bindings for crt_shpd_* APIs from libcrt."""

	def __init__(self, crt_cdll):
		self.crt_api = crt_cdll
		self._crt_shpd_api_bind()

	def _crt_shpd_api_bind(self):
		self.crt_api.crt_shpd_model_create.argtypes = [
			ctypes.c_void_p, ctypes.POINTER(shepd_inc.shepd_config)]
		self.crt_api.crt_shpd_model_create.restype = ctypes.c_void_p

		self.crt_api.crt_shpd_model_release.argtypes = [ctypes.c_void_p]
		self.crt_api.crt_shpd_model_release.restype = ctypes.c_int

		self.crt_api.crt_shpd_user_create.argtypes = [ctypes.c_void_p]
		self.crt_api.crt_shpd_user_create.restype = ctypes.c_void_p

		self.crt_api.crt_shpd_user_tokenizer_encode.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(shepd_inc.tokenizer_enc_cfg),
			ctypes.c_char_p,
			ctypes.POINTER(shepd_inc.token_id_list),
		]
		self.crt_api.crt_shpd_user_tokenizer_encode.restype = ctypes.c_int

		self.crt_api.crt_shpd_user_tokenizer_decode.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(shepd_inc.tokenizer_dec_cfg),
			ctypes.c_void_p,
			ctypes.c_uint32,
			ctypes.POINTER(shepd_inc.tokenizer_dec_res),
		]
		self.crt_api.crt_shpd_user_tokenizer_decode.restype = ctypes.c_int

		self.crt_api.crt_shpd_user_preprocess.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_char_p,
			ctypes.POINTER(shepd_inc.shepd_extra),
			ctypes.POINTER(shepd_inc.preproc_res),
		]
		self.crt_api.crt_shpd_user_preprocess.restype = ctypes.c_int

		self.crt_api.crt_shpd_user_run_ids.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(shepd_inc.shepd_run_cfg),
			ctypes.c_void_p,
			ctypes.c_uint32,
			ctypes.POINTER(shepd_inc.shepd_output),
		]
		self.crt_api.crt_shpd_user_run_ids.restype = ctypes.c_int

		self.crt_api.crt_shpd_user_reset.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.POINTER(shepd_inc.shepd_reset_cfg),
		]
		self.crt_api.crt_shpd_user_reset.restype = ctypes.c_int

		self.crt_api.crt_shpd_user_release.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
		self.crt_api.crt_shpd_user_release.restype = ctypes.c_int

		self.crt_api.crt_shpd_get_version.argtypes = [
			ctypes.c_void_p, ctypes.POINTER(shepd_inc.shepherd_version)]
		self.crt_api.crt_shpd_get_version.restype = ctypes.c_int

	def crt_shpd_model_create(self, dev_ctx, shepd_cfg):
		return self.crt_api.crt_shpd_model_create(dev_ctx, shepd_cfg)

	def crt_shpd_model_release(self, mhandle):
		return self.crt_api.crt_shpd_model_release(mhandle)

	def crt_shpd_user_create(self, mhandle):
		return self.crt_api.crt_shpd_user_create(mhandle)

	def crt_shpd_user_tokenizer_encode(self, mhandle, user_ctx, enc_cfg, text, id_list):
		return self.crt_api.crt_shpd_user_tokenizer_encode(
			mhandle, user_ctx, enc_cfg, text, id_list)

	def crt_shpd_user_tokenizer_decode(
			self, mhandle, user_ctx, dec_cfg, ids, ids_num, dec_res):
		return self.crt_api.crt_shpd_user_tokenizer_decode(
			mhandle, user_ctx, dec_cfg, ids, ids_num, dec_res)

	def crt_shpd_user_preprocess(self, mhandle, user_ctx, input_text, shepd_ex, res):
		return self.crt_api.crt_shpd_user_preprocess(
			mhandle, user_ctx, input_text, shepd_ex, res)

	def crt_shpd_user_run_ids(
			self, mhandle, user_ctx, run_cfg, input_ids, ids_num, shepd_output):
		return self.crt_api.crt_shpd_user_run_ids(
			mhandle, user_ctx, run_cfg, input_ids, ids_num, shepd_output)

	def crt_shpd_user_reset(self, mhandle, user_ctx, reset_cfg):
		return self.crt_api.crt_shpd_user_reset(mhandle, user_ctx, reset_cfg)

	def crt_shpd_user_release(self, mhandle, user_ctx):
		return self.crt_api.crt_shpd_user_release(mhandle, user_ctx)

	def crt_shpd_get_version(self, dev_ctx, ver):
		return self.crt_api.crt_shpd_get_version(dev_ctx, ver)
