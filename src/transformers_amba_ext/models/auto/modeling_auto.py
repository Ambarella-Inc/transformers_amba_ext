from typing import Optional, Dict
from transformers import AutoConfig
from ..llama import LlamaForCausalLM
from ..gemma import GemmaForCausalLM
from ..gemma2 import Gemma2ForCausalLM
from ..phi3 import Phi3ForCausalLM
from ..qwen2 import Qwen2ForCausalLM

from ...utils import logging

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = {
	"llama": LlamaForCausalLM,
	"gemma": GemmaForCausalLM,
	"gemma2": Gemma2ForCausalLM,
	"phi3": Phi3ForCausalLM,
	"qwen2": Qwen2ForCausalLM,
}

def _get_model_class(model_type, model_mapping):
	for arch in model_mapping:
		if arch == model_type:
			return model_mapping[arch]
	return None

class AutoModelForCausalLM():
	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_path: Optional[str] = None,
		device_ip: Optional[str] = None,
		device_port: Optional[int] = None,
		log_level: Optional[int] = None,
		**kwargs,
	):
		config = AutoConfig.from_pretrained(pretrained_model_path)
		model_type = getattr(config, "model_type", None)
		if model_type is None:
			raise ValueError(f"can't get valid {model_type} from the {pretrained_model_path}. "
				f"Please check if there are a config.json in the model_path. "
				f"If not, users need to create a soft link from model_desc.json or info/config.json "
				f"to config.json in the model_path.")

		supported_models = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
		model_class = _get_model_class(model_type, supported_models)
		if model_class is None:
			raise ValueError(f"can't get valid model_class with {model_type} from supported_models list. "
				f"Please check if there are a config.json in the {pretrained_model_path}. "
				f"If not, users need to create a soft link from model_desc.json or info/config.json "
				f"to config.json in the model_path.")

		if kwargs:
			print(f"{cls.__name__}: unsupported kwargs: {kwargs}")
		return model_class(pretrained_model_path, device_ip, device_port, log_level)