
__version__ = "1.7.0"
__mod_time__ = "20260624"

_import_structure = {
	# Models
	"models": [],
	"models.auto": ["AutoModelForCausalLM", "AutoModelForVision"],
	"models.llama": ["LlamaForCausalLM"],
	"models.gemma": ["GemmaForCausalLM"],
	"models.gemma2": ["Gemma2ForCausalLM"],
	"models.phi3": ["Phi3ForCausalLM"],
	"models.qwen2": ["Qwen2ForCausalLM"],
	"models.llava": ["LlavaLlamaForCausalLM"],
	"models.llava_onevision": ["LlavaOnevisionForConditionalGeneration"],
	"models.vlm": ["VLMForCausalLM"],
	"models.gpt_oss": ["GptOssForCausalLM"],
	"models.qwen3_moe": ["Qwen3MoeForCausalLM"],
	# Inference configuration
	"inference": ["vit_mode"],
}

from .models.auto import (
	AutoModelForCausalLM,
	AutoModelForVision,
)

from .models.llama import (
	LlamaForCausalLM,
)

from .models.gemma import (
	GemmaForCausalLM,
)

from .models.gemma2 import (
	Gemma2ForCausalLM,
)

from .models.phi3 import (
	Phi3ForCausalLM,
)

from .models.qwen2 import (
	Qwen2ForCausalLM,
)

from .models.llava import (
	LlavaLlamaForCausalLM,
)

from .models.llava_onevision import (
	LlavaOnevisionForConditionalGeneration,
 )
from .models.vlm import (
	VLMForCausalLM,
)

from .models.gpt_oss import (
	GptOssForCausalLM,
)

from .models.qwen3_moe import (
	Qwen3MoeForCausalLM,
)

# Unified vit_mode for all VLM models
from .inference.infer_configuration import (
	vit_mode,
)
