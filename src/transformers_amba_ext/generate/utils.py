import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, Callable, List

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
	EncoderNoRepeatNGramLogitsProcessor,
	EncoderRepetitionPenaltyLogitsProcessor,
	EpsilonLogitsWarper,
	EtaLogitsWarper,
	ExponentialDecayLengthPenalty,
	ForcedBOSTokenLogitsProcessor,
	ForcedEOSTokenLogitsProcessor,
	HammingDiversityLogitsProcessor,
	InfNanRemoveLogitsProcessor,
	LogitNormalization,
	LogitsProcessorList,
	MinLengthLogitsProcessor,
	MinNewTokensLengthLogitsProcessor,
	MinPLogitsWarper,
	NoBadWordsLogitsProcessor,
	NoRepeatNGramLogitsProcessor,
	PrefixConstrainedLogitsProcessor,
	RepetitionPenaltyLogitsProcessor,
	SequenceBiasLogitsProcessor,
	SuppressTokensAtBeginLogitsProcessor,
	SuppressTokensLogitsProcessor,
	TemperatureLogitsWarper,
	TopKLogitsWarper,
	TopPLogitsWarper,
	TypicalLogitsWarper,
	UnbatchedClassifierFreeGuidanceLogitsProcessor,
	WatermarkLogitsProcessor,
)

class GenerationMode():
	"""
	Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
	"""

	# Non-beam methods
	CONTRASTIVE_SEARCH = "contrastive_search"
	GREEDY_SEARCH = "greedy_search"
	SAMPLE = "sample"
	ASSISTED_GENERATION = "assisted_generation"
	DOLA_GENERATION = "dola_generation"
	# Beam methods
	BEAM_SEARCH = "beam_search"
	BEAM_SAMPLE = "beam_sample"
	CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
	GROUP_BEAM_SEARCH = "group_beam_search"


class GenerationMixin():
	def __init__(self):
		pass

	def __prepare_generation_config(
		self,
		**kwargs: Dict
	) -> Tuple[GenerationConfig]:
		self.do_sample = kwargs.get("do_sample")
		self.top_k = kwargs.get("top_k")
		self.penalty_alpha = kwargs.get("penalty_alpha")

		generation_config = GenerationConfig(**kwargs)
		return generation_config

	def __get_generation_mode(self):
		if self.do_sample is not None and self.do_sample is False:
			if (self.top_k is not None and self.top_k > 1
				and self.penalty_alpha is not None and self.penalty_alpha > 0):
				generation_mode = GenerationMode.CONTRASTIVE_SEARCH
			else:
				generation_mode = GenerationMode.GREEDY_SEARCH
		else:
			generation_mode = GenerationMode.SAMPLE

		return generation_mode

	def __get_logits_processor(
		self,
		generation_config: GenerationConfig,
		prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
	)-> LogitsProcessorList:
		processors = LogitsProcessorList()

		if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
			raise NotImplementedError("Todo")
			processors.append(
				UnbatchedClassifierFreeGuidanceLogitsProcessor(
					generation_config.guidance_scale,
					self,
					unconditional_ids=negative_prompt_ids,
					unconditional_attention_mask=negative_prompt_attention_mask,
					use_cache=model_kwargs["use_cache"],
				)
			)
		if generation_config.sequence_bias is not None:
			raise NotImplementedError("Todo")
			processors.append(SequenceBiasLogitsProcessor(sequence_bias=generation_config.sequence_bias))

		if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
			raise NotImplementedError("Todo")
			processors.append(
				HammingDiversityLogitsProcessor(
					diversity_penalty=generation_config.diversity_penalty,
					num_beams=generation_config.num_beams,
					num_beam_groups=generation_config.num_beam_groups,
				)
			)
		if (
			generation_config.encoder_repetition_penalty is not None
			and generation_config.encoder_repetition_penalty != 1.0
		):
			raise NotImplementedError("Todo")
			processors.append(
				EncoderRepetitionPenaltyLogitsProcessor(
					penalty=generation_config.encoder_repetition_penalty,
					encoder_input_ids=encoder_input_ids,
				)
			)
		if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
			raise NotImplementedError("Todo")
			processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
		if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
			raise NotImplementedError("Todo")
			processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
		if (
			generation_config.encoder_no_repeat_ngram_size is not None
			and generation_config.encoder_no_repeat_ngram_size > 0
		):
			raise NotImplementedError("Todo")
			processors.append(
				EncoderNoRepeatNGramLogitsProcessor(
					generation_config.encoder_no_repeat_ngram_size,
					encoder_input_ids,
				)
			)
		if generation_config.bad_words_ids is not None:
			raise NotImplementedError("Todo")
			processors.append(
				NoBadWordsLogitsProcessor(
					generation_config.bad_words_ids,
					generation_config._eos_token_tensor,
				)
			)
		""" _eos_token_tensor can't be found
		if (
			generation_config.min_length is not None
			and generation_config._eos_token_tensor is not None
			and generation_config.min_length > 0
		):
			raise NotImplementedError("Todo")
			processors.append(
				MinLengthLogitsProcessor(
					generation_config.min_length,
					generation_config._eos_token_tensor,
					device=device,
				)
			)
		if (
			generation_config.min_new_tokens is not None
			and generation_config._eos_token_tensor is not None
			and generation_config.min_new_tokens > 0
		):
			raise NotImplementedError("Todo")
			processors.append(
				MinNewTokensLengthLogitsProcessor(
					input_ids_seq_length,
					generation_config.min_new_tokens,
					generation_config._eos_token_tensor,
					device=device,
				)
			)
		"""
		if prefix_allowed_tokens_fn is not None:
			raise NotImplementedError("Todo")
			processors.append(
				PrefixConstrainedLogitsProcessor(
					prefix_allowed_tokens_fn,
					generation_config.num_beams // generation_config.num_beam_groups,
				)
			)
		if generation_config.forced_bos_token_id is not None:
			raise NotImplementedError("Todo")
			processors.append(
				ForcedBOSTokenLogitsProcessor(
					generation_config.forced_bos_token_id,
				)
			)
		if generation_config.forced_eos_token_id is not None:
			raise NotImplementedError("Todo")
			processors.append(
				ForcedEOSTokenLogitsProcessor(
					generation_config.max_length,
					generation_config.forced_eos_token_id,
					device=device,
				)
			)
		if generation_config.remove_invalid_values is True:
			raise NotImplementedError("Todo")
			processors.append(InfNanRemoveLogitsProcessor())
		if generation_config.exponential_decay_length_penalty is not None:
			raise NotImplementedError("Todo")
			processors.append(
				ExponentialDecayLengthPenalty(
					generation_config.exponential_decay_length_penalty,
					generation_config._eos_token_tensor,
					input_ids_seq_length,
				)
			)
		if generation_config.suppress_tokens is not None:
			raise NotImplementedError("Todo")
			processors.append(
				SuppressTokensLogitsProcessor(
					generation_config.suppress_tokens,
					device=device,
				)
			)
		if generation_config.begin_suppress_tokens is not None:
			raise NotImplementedError("Todo")
			begin_index = input_ids_seq_length
			begin_index = (
				begin_index
				if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
				else begin_index + 1
			)
			processors.append(
				SuppressTokensAtBeginLogitsProcessor(
					generation_config.begin_suppress_tokens,
					begin_index,
					device=device,
				)
			)

		"""
		the forced_decoder_ids field was ported from 4.45.0 and removed after transformers>=4.53.0,
		we can skip this check to ensure it can work with transformers>=4.53.0.
		"""
		#if generation_config.forced_decoder_ids is not None:
		#	raise NotImplementedError("Todo")
		#	# TODO (sanchit): move this exception to GenerationConfig.validate() when TF & FLAX are aligned with PT
		#	raise ValueError(
		#		"You have explicitly specified `forced_decoder_ids`. Please remove the `forced_decoder_ids` argument "
		#		"in favour of `input_ids` or `decoder_input_ids` respectively.",
		#	)
		if generation_config.watermarking_config is not None:
			raise NotImplementedError("Todo")
			processors.append(
				WatermarkLogitsProcessor(
					vocab_size=self.config.vocab_size,
					device=device,
					greenlist_ratio=generation_config.watermarking_config.greenlist_ratio,
					bias=generation_config.watermarking_config.bias,
					hashing_key=generation_config.watermarking_config.hashing_key,
					seeding_scheme=generation_config.watermarking_config.seeding_scheme,
					context_width=generation_config.watermarking_config.context_width,
				)
			)

		# TODO (joao): find a strategy to specify the order of the processors
		# processors = self._merge_criteria_processor_list(processors, logits_processor)

		# Processors previously known as `LogitsWarpers`, only applied with sampling strategies
		if generation_config.do_sample:
			# In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
			# better score (i.e. keep len(list(generation_config._eos_token_tensor)) + 1)
			if generation_config.num_beams > 1:
				raise NotImplementedError("Todo")
				if isinstance(generation_config._eos_token_tensor, list):
					min_tokens_to_keep = len(generation_config._eos_token_tensor) + 1
				elif isinstance(generation_config._eos_token_tensor, torch.Tensor):
					min_tokens_to_keep = generation_config._eos_token_tensor.shape[0] + 1
				else:
					min_tokens_to_keep = 2
			else:
				min_tokens_to_keep = 1

			# the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
			# all samplers can be found in `generation_utils_samplers.py`
			if generation_config.temperature is not None and generation_config.temperature != 1.0:
				processors.append(TemperatureLogitsWarper(generation_config.temperature))
			if generation_config.top_k is not None and generation_config.top_k != 0:
				processors.append(
					TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep)
				)
			if generation_config.top_p is not None and generation_config.top_p < 1.0:
				processors.append(
					TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep)
				)
			if generation_config.min_p is not None:
				# Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
				processors.append(
					MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep)
				)
			if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
				processors.append(
					TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
				)
			if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
				processors.append(
					EpsilonLogitsWarper(
						epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep
					)
				)
			if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
				processors.append(
					EtaLogitsWarper(
						epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep, device=device
					)
				)

		# `LogitNormalization` should always be the last logit processor, when present
		if generation_config.renormalize_logits is True:
			processors.append(LogitNormalization())
		return processors

	def __sample(
		self,
		input_ids: torch.Tensor,
		input_logits: torch.FloatTensor,
		logits_processor: LogitsProcessorList,
		generation_config: GenerationConfig,
	):
		next_token_scores = logits_processor(input_ids, input_logits)
		if self.do_sample:
			probs = nn.functional.softmax(next_token_scores, dim=-1)
			# TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
			next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
		else:
			next_tokens = torch.argmax(next_token_scores, dim=-1)
		return next_tokens

	def sample(
		self,
		input_ids: np.array,
		input_logits: np.array,
		**kwargs: Dict
	):
		input_ids_tensor = torch.Tensor(input_ids).to(torch.uint32)
		input_logits_tensor = torch.Tensor(input_logits).to(torch.float32)

		generate_config = self.__prepare_generation_config(**kwargs)
		prepared_logits_processor = self.__get_logits_processor(generate_config)
		generation_mode = self.__get_generation_mode()

		result = 0
		if generation_mode == GenerationMode.ASSISTED_GENERATION:
			raise NotImplementedError("Todo")
		elif generation_mode == GenerationMode.DOLA_GENERATION:
			raise NotImplementedError("Todo")
		elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
			raise NotImplementedError("Todo")
		elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
			result = self.__sample(
				input_ids=input_ids_tensor,
				input_logits=input_logits_tensor,
				logits_processor=prepared_logits_processor,
				generation_config = generate_config)
		elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
			raise NotImplementedError("Todo")
		elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
			raise NotImplementedError("Todo")
		elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
			raise NotImplementedError("Todo")

		return result
