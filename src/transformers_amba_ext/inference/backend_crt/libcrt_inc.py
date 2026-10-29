import ctypes
import enum

CRT_MAX_DEVICE_NUM = 16


class crt_log(enum.IntEnum):
	CRT_LOG_ERR = 0
	CRT_LOG_WARN = 1
	CRT_LOG_INFO = 2
	CRT_LOG_DEBUG = 3
	CRT_LOG_VERBOSE = 4
	CRT_LOG_DEFAULT = CRT_LOG_INFO


class crt_version(ctypes.Structure):
	_fields_ = [
		("major", ctypes.c_uint32),
		("minor", ctypes.c_uint32),
		("patch", ctypes.c_uint32),
		("mod_time", ctypes.c_uint),
		("description", ctypes.c_char * 64),
	]


class crt_config(ctypes.Structure):
	_fields_ = [
		("log_level", ctypes.c_uint8),
		("reserved_0", ctypes.c_uint8 * 3),
		("reserved", ctypes.c_uint32 * 63),
	]


class CRT_DEVICE_MODE(enum.IntEnum):
	CRT_DEVICE_MODE_LOCAL = 0
	CRT_DEVICE_MODE_REMOTE = 1
	CRT_DEVICE_MODE_NUM = 2


class crt_dev_info(ctypes.Structure):
	_fields_ = [
		("dev_id", ctypes.c_uint32),
		("dev_mode", ctypes.c_int),
		("reserved", ctypes.c_uint32 * 6),
	]


class crt_device(ctypes.Structure):
	_fields_ = [
		("device_num", ctypes.c_uint32),
		("dev_info", crt_dev_info * CRT_MAX_DEVICE_NUM),
	]
