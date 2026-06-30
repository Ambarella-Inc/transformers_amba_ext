import numpy as np
from typing import Optional
from ..inference.backend_shepd.backend_shepd import infer_shepd
from ..inference.backend_dist_ai.backend_dist_ai import infer_dist_ai
from ..inference.infer_configuration import backend_type

class inference_runtime():
	def __init__(self, config):
		if config.backend == backend_type.SHEPD:
			self.backend = infer_shepd(config)
		elif config.backend == backend_type.DIST_AI:
			self.backend = infer_dist_ai(config)
		else:
			raise ValueError(f"Unsupported backend: {config.backend}")

	@property
	def backend_config(self):
		return self.backend.config

	@property
	def backend_eos_token(self):
		return self.backend.config.eos_token_id

	def __getattr__(self, name):
		return getattr(self.backend, name)

	def infer_get_version(self, *args, **kwargs):
		return self.backend.infer_get_version(*args, **kwargs)

	def infer_init(self, *args, **kwargs):
		self.backend.infer_init(*args, **kwargs)

	def infer_exit(self, *args, **kwargs):
		self.backend.infer_exit(*args, **kwargs)

	def infer_model_init(self, *args, **kwargs):
		return self.backend.infer_model_init(*args, **kwargs)

	def infer_model_release(self, *args, **kwargs):
		self.backend.infer_model_release(*args, **kwargs)

	def infer_user_init(self, *args, **kwargs):
		return self.backend.infer_user_init(*args, **kwargs)

	def infer_user_encode(self, *args, **kwargs):
		return self.backend.infer_user_encode(*args, **kwargs)

	def infer_user_decode(self, *args, **kwargs):
		return self.backend.infer_user_decode(*args, **kwargs)

	def infer_user_preprocess(self, *args, **kwargs):
		return self.backend.infer_user_preprocess(*args, **kwargs)

	def infer_user_run_ids(self, *args, **kwargs):
		return self.backend.infer_user_run_ids(*args, **kwargs)

	def infer_user_bcast_ids(self, *args, **kwargs):
		return self.backend.infer_user_bcast_ids(*args, **kwargs)

	def infer_user_run_logits(self, *args, **kwargs):
		return self.backend.infer_user_run_logits(*args, **kwargs)

	def infer_user_run_embeddings(self, *args, **kwargs):
		return self.backend.infer_user_run_embeddings(*args, **kwargs)

	def infer_user_release(self, *args, **kwargs):
		self.backend.infer_user_release(*args, **kwargs)

	def infer_input_text_cvt(self, *args, **kwargs):
		return self.backend.infer_input_text_cvt(*args, **kwargs)

	def infer_input_token_cvt(self, *args, **kwargs):
		return self.backend.infer_input_token_cvt(*args, **kwargs)

	def infer_user_reset(self, *args, **kwargs):
		return self.backend.infer_user_reset(*args, **kwargs)

	def infer_at_eos(self, *args, **kwargs):
		return self.backend.infer_at_eos(*args, **kwargs)

	def infer_at_end(self, *args, **kwargs):
		return self.backend.infer_at_end(*args, **kwargs)

	def map_vit_mode_for_llava_ov(self, *args, **kwargs):
		return self.backend.map_vit_mode_for_llava_ov(*args, **kwargs)

	def map_vit_mode_for_vlm(self, *args, **kwargs):
		return self.backend.map_vit_mode_for_vlm(*args, **kwargs)