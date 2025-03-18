
from typing import Optional, Tuple, Union, Dict
from functools import partial, reduce
from PIL import Image
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)

class SigLipImageProcessor:
	def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
		crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
		crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

		self.image_mean = image_mean
		self.image_std = image_std
		self.size = size
		self.resample = resample
		self.rescale_factor = rescale_factor
		self.data_format = data_format
		self.crop_size = crop_size

	def preprocess(self, images, return_tensors):
		if isinstance(images, Image.Image):
			images = [images]
		else:
			# to adapt video data
			images = [to_numpy_array(image) for image in images]
			assert isinstance(images, list)

		transforms = [
			convert_to_rgb,
			to_numpy_array,
			partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
			partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
			partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
			partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
		]

		images = reduce(lambda x, f: [*map(f, x)], transforms, images)
		data = {"pixel_values": images}

		return BatchFeature(data=data, tensor_type=return_tensors)
