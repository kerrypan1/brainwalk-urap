"""
File Summary
------------
LLaVA small-model wrapper (OneVision 0.5B) that mirrors the larger LLaVA
adapter API while reducing compute and memory requirements
"""
from __future__ import annotations

from .base_vlm import BaseVLM

# Video decode and model stack
import av
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    BitsAndBytesConfig,
)


class VideoLLaVASmall(BaseVLM):
    name = "llava_s"

    def __init__(self, model_id=None, num_frames=16, max_new_tokens=256):
        # Use the compact OneVision checkpoint by default
        self.model_id = model_id or "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        self.num_frames = int(num_frames)
        self.max_new_tokens = int(max_new_tokens)

        # 4-bit quantization keeps VRAM usage low
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)

        # Set by scripts/inference.py before running clips
        self.prompt_text = ""

    def _decode_first_n_frames_rgb(self, video_path: str) -> np.ndarray:
        """Decode first N RGB frames from a pre-cut clip"""
        frames = []
        container = av.open(video_path)
        try:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= self.num_frames:
                    break
        finally:
            container.close()

        if not frames:
            return np.empty((0,), dtype=np.uint8)
        return np.stack(frames, axis=0)

    def run(self, video_path, prompt=None):
        # Allow an optional direct prompt override while keeping default behavior
        prompt_text = self.prompt_text if prompt is None else str(prompt)
        clip = self._decode_first_n_frames_rgb(str(video_path))
        if clip.size == 0:
            return "", None

        # Keep the same conversation schema as the larger LLaVA wrapper
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Build model input text and video tensors
        templated = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
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

        # Decode only the generated response and skip prompt tokens
        prompt_len = inputs.input_ids.shape[1]
        generated = output[0][prompt_len:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        return text, None
