"""
File Summary
------------
Wraps LLaVA-NeXT-Video behind the shared `BaseVLM` interface
Input videos are assumed to already be short pre-cut clips, so sampling is
just first N frames instead of full-video temporal resampling
"""
from __future__ import annotations

from .base_vlm import BaseVLM

# Video decode and model stack
import av
import numpy as np
import torch
from transformers import (
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
    BitsAndBytesConfig,
)


class VideoLLaVA(BaseVLM):
    name = "llava"

    def __init__(self, model_id=None, num_frames=16, max_new_tokens=256):
        # Use the large 7B LLaVA-NeXT-Video checkpoint by default
        self.model_id = model_id or "llava-hf/LLaVA-NeXT-Video-7B-hf"
        self.num_frames = int(num_frames)
        self.max_new_tokens = int(max_new_tokens)

        # 4-bit load keeps memory usage manageable while preserving quality
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        # Processor handles prompt formatting and video tensor conversion
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_id, use_fast=True)

        # Filled by scripts/inference.py before clip processing starts
        self.prompt_text = ""

    def _decode_first_n_frames_rgb(self, video_path: str) -> np.ndarray:
        """
        Decode up to N RGB frames from the start of a clip
        Returns shape (T, H, W, 3) uint8
        """
        frames = []
        container = av.open(video_path)
        try:
            # Sequential decode is enough because clip.py already cuts clips to length
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= self.num_frames:
                    break
        finally:
            container.close()

        if not frames:
            # Caller handles empty clips safely
            return np.empty((0,), dtype=np.uint8)

        return np.stack(frames, axis=0)

    def run(self, video_path, prompt=None):
        # Keep project behavior where prompt is loaded externally
        prompt_text = self.prompt_text

        clip = self._decode_first_n_frames_rgb(str(video_path))
        if clip.size == 0:
            return "", None

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Convert role and content conversation into model-expected chat text
        templated = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Processor builds tokenized prompt and video tensors in one call
        inputs = self.processor(
            text=templated,
            videos=clip,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, torch.float16)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Strip input prompt tokens and decode only generated continuation
        prompt_len = inputs.input_ids.shape[1]
        generated = output[0][prompt_len:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        return text, None
