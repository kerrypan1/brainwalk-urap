"""
File Summary
------------
This is the shared interface for all model wrappers in this project
Every adapter in `models/` should inherit from `BaseVLM` so inference can call
all models the same way
"""
from abc import ABC, abstractmethod


class BaseVLM(ABC):
	"""Common contract all VLM wrappers follow in this pipeline"""

	name = "base"

	@abstractmethod
	def run(self, video_path, prompt):
		"""
		Run the model on one video clip and one prompt string

		Expected return type across implementations:
		- `text_output`: generated text prediction
		- `embedding_or_none`: optional extra output (currently unused)
		"""
		raise NotImplementedError
