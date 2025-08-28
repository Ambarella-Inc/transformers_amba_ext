
import numpy as np
import torch
from typing import Optional, Any, Dict, Union
from transformers import TextIteratorStreamer

from ...utils import logging
from ...generate import utils
from ...inference import libshepd_inc as inc
from ...inference import inference as infer
from ...inference import infer_configuration as inf_config
from ...inference import infer_multi_user as inf_multi_user
from ... import __version__

logger = logging.get_logger(__name__)

class model_base():
	def __init__(
		self,
		pretrained_model_path: Optional[str] = None,
		device_ip: Optional[str] = None,
		device_port: Optional[int] = None,
		log_level: Optional[int] = None,
		model_type: Optional[str] = None,
	):
		config = inf_config.infer_config(
			model_path=pretrained_model_path,
			batch_size = 64,
			max_user_num = 8,
			device_ip = device_ip if device_ip is not None else None,
			device_port = device_port if device_port is not None else None,
			device_type = inc.shepd_device_type_t.SHEPD_DEVICE_REMOTE \
				if device_ip is not None or device_port is not None \
				else inc.shepd_device_type_t.SHEPD_DEVICE_LOCAL,
			lib_log_level = log_level if log_level is not None else 0)
		self.ext_config = config

		self.infer = infer.inference_runtime(config)
		self.infer.infer_init()
		self.__infer_version_check(__version__, self.infer.infer_get_version())
		self.model_handle = self.infer.infer_model_init(model_type)

		self.multi_user_ctx = inf_multi_user.infer_multi_user_ctx(config.max_user_num)

	def __del__(self):
		for i in range(self.multi_user_ctx.get_user_cnt()):
			user_ctx = self.multi_user_ctx.get_user_ctx_with_index(i)
			self.infer.infer_user_release(self.model_handle, user_ctx.handle)
		self.multi_user_ctx.release_all_user()

		self.infer.infer_model_release(self.model_handle)
		self.infer.infer_exit()

	def __infer_version_check(self, ver_infer, ver_shepd):
		ver_infer_list = ver_infer.split('.')
		ver_shepd_list = ver_shepd.split('.')

		if (ver_infer_list[0] != ver_shepd_list[0]) or (ver_infer_list[1] != ver_shepd_list[1]):
			raise ValueError(
				f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				f"Wrapper API on the transformers_amba_ext package. Version: {ver_infer}\n"
				f"Shepherd library. Version: {ver_shepd}\n"
				f"This mismatched version might trigger some unknown issues with incompatibility API.\n"
				f"Please contact the Ambarella team if you are unsure how to handle it.\n"
				f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

	def __multi_user_creat(self):
		user_handle = self.infer.infer_user_init(self.model_handle)
		user_ctx = self.multi_user_ctx.creat_user(user_handle)
		return user_ctx

	def multi_user_get(
		self,
		user_id: Optional[list] = None
	):
		user_ctx = None

		if user_id is None:
			user_ctx = self.multi_user_ctx.get_first_user_ctx() \
				if self.multi_user_ctx.get_user_cnt() > 0 else self.__multi_user_creat()
		else:
			user_ctx = self.multi_user_ctx.get_user_ctx(user_id[0]) \
				if (user_id and user_id[0] != 0) else self.__multi_user_creat()
			if user_id:
				user_id[0] = user_ctx.id
			else:
				user_id.append(user_ctx.id)

		return user_ctx

	def input_ids_cvt(self, input_ids: torch.Tensor):
		return input_ids.numpy().astype(np.uint32).reshape(-1)

	def output_ids_cvt(self, output_ids: np.uint32):
		return torch.tensor(output_ids.astype(np.int32).reshape(1, -1), dtype=torch.int32)

	def encode(self,
		prompt : str = None,
		user_id: Optional[list] = None,
	):
		r"""This API will apply default system prompt for input text by Shepherd library and then
		return the encoded token list. Users can also use the tokenizer from Transformers to do
		apply_chat_template and encode.

		Args:
			prompt (`str`):
				Indices the input text data.
			user_id (`list`, *optional*):
				It's an extended configuration for Ambarella chips to index the user ID for current inference.
				Users need specify this parameters if enable multi user.
		Returns (`torch.Tensor`):
			return the encoded token list for input prompt
		"""
		user_ctx = self.multi_user_get(user_id)
		id_list = self.infer.infer_user_encode(
			self.model_handle, user_ctx.handle, prompt)

		return self.output_ids_cvt(id_list)

	def generate_id(self,
		input_ids: Union[np.ndarray, torch.Tensor] = None,
		user_id: Optional[list] = None,
	):
		r"""Generate next token for current inputs.

		Args:
			input_ids (`numpy.ndarray`, `torch.Tensor` of shape `(1, sequence_length)`):
				Indices of input sequence tokens in the vocabulary.
			user_id (`list`, *optional*):
				It's an extended configuration for Ambarella chips to index the user ID for current inference.
				Users need specify this parameters if enable multi user.
		Returns:
			token: the output token id
			pos: the output position
		"""
		ids_num = input_ids.shape[-1]
		if isinstance(input_ids, torch.Tensor):
			input_ids = self.input_ids_cvt(input_ids)

		input_ids_ctype = self.infer.infer_input_token_cvt(input_ids.ctypes.data)
		user_ctx = self.multi_user_get(user_id)

		token, pos = self.infer.infer_user_run_ids(
			self.model_handle, user_ctx.handle, input_ids_ctype, ids_num)

		return token, pos

	def generate_ids_until(self,
		input_ids,
		position: Optional[list] = None,
		user_id: Optional[list] = None,
		past_key_values: Optional[bool] = None,
		streamer: Optional[TextIteratorStreamer] = None,
		kwargs: Dict = None,
	):
		r"""
		Args:
			input_ids (`torch.LongTensor` of shape `(1, sequence_length)`):
				Indices of input sequence tokens in the vocabulary.
			position (`list`, *optional*):
				Indices to get of position of current conversation, like position=[], default is None.
			user_id (`list`, *optional*):
				It's an extended configuration for Ambarella chips to index the user ID for current inference.
				Users need specify this parameters if enable multi user.
			past_key_values (`bool`, *optional*):
				Indices the past conversation can be used for current inference.
			streamer (`TextIteratorStreamer`, *optional*):
				Streamer object that will be used to stream the generated sequences. Generated tokens are passed
				through `streamer.put(token_ids)` and the streamer is responsible for any further processing.

		Returns:
			`numpy.array`: the array of output token id
		"""
		ids_num = input_ids.shape[-1]
		use_past_key_values = past_key_values if past_key_values is not None else False
		user_ctx = self.multi_user_get(user_id)
		max_length = 0
		last_token_id = 0
		response = []

		kwargs_supported = self.__get_supported_args(**kwargs)

		while True:
			token, pos = self.generate_id(input_ids, user_id)

			max_length = self.__valid_max_length(ids_num, **kwargs_supported)
			if self.infer.infer_at_eos(last_token_id) or self.infer.infer_at_end(pos, max_length):
				break

			if not self.infer.infer_at_eos(token):
				response = np.append(response, token)
				if streamer is not None:
					streamer.put(np.array([[token]]))

			last_token_id = token

		if streamer is not None:
			streamer.end()

		if (pos > self.ext_config.max_sequence_length - self.ext_config.pos_margin):
			logger.warning(f"auto reset since hit pos margin, "
				f"pos ({pos}) > max_len ({self.ext_config.max_sequence_length}) - margin ({self.ext_config.pos_margin})")
			pos = self.reset(user_id)

		if use_past_key_values == False:
			pos = self.reset(user_id)

		self.multi_user_ctx.update_user_pos(user_ctx.id, pos)
		if position is not None:
			if position:
				position[0] = pos
			else:
				position.append(pos)

		logger.debug(f"[generate_ids_until]: "
			f"user_ctx: user_id: {user_ctx.id}, handle: {user_ctx.handle}, pos: {pos}")
		return response.astype(np.uint32).reshape(-1)

	def generate_logits(self,
		input_ids: Union[np.ndarray, torch.Tensor] = None,
		user_id: Optional[list] = None,
	):
		r"""Generate the logits of next token for current inputs.

		Args:
			input_ids (`numpy.ndarray`, `torch.Tensor` of shape `(1, sequence_length)`):
				Indices of input sequence tokens in the vocabulary.
			user_id (`list`, *optional*):
				It's an extended configuration for Ambarella chips to index the user ID for current inference.
				Users need specify this parameters if enable multi user.
		Returns:
			logits (`numpy.ndarray`): the output logits
			pos: the output position
		"""
		ids_num = input_ids.shape[-1]
		if isinstance(input_ids, torch.Tensor):
			input_ids = self.input_ids_cvt(input_ids)

		input_ids_ctype = self.infer.infer_input_token_cvt(input_ids.ctypes.data)
		user_ctx = self.multi_user_get(user_id)
		token = input_ids[0] if ids_num == 1 else 0

		logits, pos = self.infer.infer_user_run_logits(
			self.model_handle, user_ctx.handle, input_ids_ctype, ids_num, token)

		return logits, pos

	def generate_logits_until(self,
		input_ids,
		position: Optional[list] = None,
		user_id: Optional[list] = None,
		past_key_values: Optional[bool] = None,
		streamer: Optional[TextIteratorStreamer] = None,
		kwargs: Dict = None,
	):
		ids_num = input_ids.shape[-1]
		use_past_key_values = past_key_values if past_key_values is not None else False
		user_ctx = self.multi_user_get(user_id)
		token = 0
		last_token_id = 0
		max_length = 0
		response = []

		logits_process = utils.GenerationMixin()
		kwargs_supported = self.__get_supported_args(**kwargs)

		while True:
			logits, pos = self.generate_logits(input_ids, user_id)
			token = logits_process.sample(input_ids, logits, **kwargs_supported)
			input_ids = token

			max_length = self.__valid_max_length(ids_num, **kwargs_supported)
			if self.infer.infer_at_eos(last_token_id) or self.infer.infer_at_end(pos, max_length):
				break

			if not self.infer.infer_at_eos(token):
				response = np.append(response, token)
				if streamer is not None:
					streamer.put(token)

			last_token_id = token

		if streamer is not None:
			streamer.end()

		if (pos > self.ext_config.max_sequence_length - self.ext_config.pos_margin):
			logger.warning(f"auto reset since hit pos margin, "
				f"pos ({pos}) > max_len ({self.ext_config.max_sequence_length}) - margin ({self.ext_config.pos_margin})")
			pos = self.reset(user_id)

		if use_past_key_values == False:
			pos = self.reset(user_id)

		self.multi_user_ctx.update_user_pos(user_ctx.id, pos)
		if position is not None:
			if position:
				position[0] = pos
			else:
				position.append(pos)

		logger.debug(f"[generate_logits_until]: "
			f"user_ctx: user_id: {user_ctx.id}, handle: {user_ctx.handle}, pos: {pos}")
		return response.astype(np.uint32).reshape(-1)

	def reset(
		self,
		user_id: Optional[list] = None
	):
		r"""
		Args:
			user_id (`list`, *optional*):
				Indices the user ID for current inference. Users need specify this parameters if enable multi user.
		Returns (`numpy.uint32`):
			return the position for current user after reset
		"""
		user_ctx = self.multi_user_get(user_id)
		pos = self.infer.infer_user_reset(self.model_handle, user_ctx.handle)
		self.multi_user_ctx.update_user_pos(user_ctx.id, pos)
		return pos

	def __valid_max_length(self, input_ids_length, **kwargs):
		max_length = kwargs.get("max_length")
		max_new_tokens = kwargs.get("max_new_tokens")
		valid_max_len = self.ext_config.max_sequence_length

		if max_length is not None and max_length < valid_max_len:
			valid_max_len = max_length

		if max_new_tokens is not None and (max_new_tokens + input_ids_length) < valid_max_len:
			valid_max_len = input_ids_length + max_new_tokens

		return valid_max_len

	def __get_supported_args(self, **kwarg):
		unsupported_arg_list = []
		for i, item in enumerate(kwarg):
			if item == "do_sample" or     \
				item == "top_p" or        \
				item == "top_k" or        \
				item == "temperature" or  \
				item == "max_length" or   \
				item == "max_new_tokens":
				pass
			else:
				unsupported_arg_list.append(item)

		if len(unsupported_arg_list):
			for i in range(len(unsupported_arg_list)):
				kwarg.pop(unsupported_arg_list[i], None)
			logger.warning_once(f"generate unsupported kwargs: {unsupported_arg_list}, auto removed.")
			logger.warning_once(f"generate supported kwargs: {kwarg}")

		return kwarg
