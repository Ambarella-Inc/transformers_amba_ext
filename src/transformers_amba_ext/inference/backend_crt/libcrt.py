
import ctypes
import ctypes.util
from ...inference.backend_crt import libcrt_inc as inc


class libcrt_api():
	def __init__(self, lib_name):
		lib_path = ctypes.util.find_library(lib_name)
		if lib_path is None:
			raise ValueError(
				f"lib_path is None with lib_name: {lib_name}, please export LD_LIBRARY_PATH.")
		self.crt_api = ctypes.cdll.LoadLibrary(lib_path)
		self._crt_api_bind()

	def _crt_api_bind(self):
		self.crt_api.crt_init.argtypes = [ctypes.POINTER(inc.crt_config)]
		self.crt_api.crt_init.restype = ctypes.c_int

		self.crt_api.crt_exit.argtypes = []
		self.crt_api.crt_exit.restype = ctypes.c_int

		self.crt_api.crt_get_version.argtypes = [ctypes.POINTER(inc.crt_version)]
		self.crt_api.crt_get_version.restype = ctypes.c_int

		self.crt_api.crt_get_device_list.argtypes = [ctypes.POINTER(inc.crt_device)]
		self.crt_api.crt_get_device_list.restype = ctypes.c_int

		self.crt_api.crt_create_device_context.argtypes = [ctypes.c_uint32]
		self.crt_api.crt_create_device_context.restype = ctypes.c_void_p

		self.crt_api.crt_destroy_device_context.argtypes = [ctypes.c_void_p]
		self.crt_api.crt_destroy_device_context.restype = ctypes.c_int

		self.crt_api.crt_get_device_ip.argtypes = [
			ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
		self.crt_api.crt_get_device_ip.restype = ctypes.c_int

	def crt_init(self, config):
		return self.crt_api.crt_init(config)

	def crt_exit(self):
		return self.crt_api.crt_exit()

	def crt_get_version(self, ver):
		return self.crt_api.crt_get_version(ver)

	def crt_get_device_list(self, device_list):
		return self.crt_api.crt_get_device_list(device_list)

	def crt_create_device_context(self, device_id):
		return self.crt_api.crt_create_device_context(device_id)

	def crt_destroy_device_context(self, device_context):
		return self.crt_api.crt_destroy_device_context(device_context)

	def crt_get_device_ip(self, device_context, ip, size):
		return self.crt_api.crt_get_device_ip(device_context, ip, size)
