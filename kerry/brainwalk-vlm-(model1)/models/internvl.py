from __future__ import annotations

from .base_vlm import BaseVLM

# Video decode and model dependencies
import av
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForCausalLM, AutoTokenizer


class InternVL(BaseVLM):
    """
    File Summary
    ------------
    InternVL wrapper that adapts OpenGVLab checkpoints to the shared
    `BaseVLM.run(video_path, prompt)` interface used by `scripts/inference.py`
    """

    name = "internvl"

    def __init__(self, model_name=None, num_frames=16, max_new_tokens=256):
        # Use InternVL2-2B by default unless caller overrides it
        self.model_name = model_name or "OpenGVLab/InternVL2-2B"
        self.num_frames = int(num_frames)
        self.max_new_tokens = int(max_new_tokens)

        # Keep runtime device and dtype in one place
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Load model once at init so clip runs stay light
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=False,
                device_map=None,
                trust_remote_code=True,
            )
            .eval()
            .to(self.device)
        )

        # InternVL uses custom tokenizer/model code so trust_remote_code stays on
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            trust_remote_code=True,
        )

        # Prompt text is injected by scripts/inference.py before inference starts
        self.prompt_text = ""

        # Reuse one transform pipeline for all decoded frames
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _decode_first_n_frames(self, video_path: str) -> list[Image.Image]:
        """
        Decode the first N frames from a pre-cut clip
        Since we are using pre-cut 16 frame clips, this will just decode all the frames
        """
        frames: list[Image.Image] = []
        container = av.open(video_path)
        try:
            # Sequential decode is simple and reliable for short generated clips
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="rgb24")
                frames.append(Image.fromarray(arr))
                if len(frames) >= self.num_frames:
                    break
        finally:
            container.close()
        return frames

    def run(self, video_path, prompt=None):
        # Keep existing project behavior where prompt_text is preloaded externally
        prompt_text = self.prompt_text

        frames = self._decode_first_n_frames(str(video_path))
        if not frames:
            # Return empty output for unreadable or empty clips
            return "", None

        # Stack transformed frames into a model-ready tensor batch
        pixel_values = torch.stack([self.transform(img) for img in frames]).to(
            device=self.device,
            dtype=self.dtype,
        )

        # InternVL chat expects FrameX placeholders in the question text
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(pixel_values.shape[0])])
        question = video_prefix + prompt_text

        generation_config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}

        with torch.inference_mode():
            # num_patches_list aligns with one patch group per frame
            response, _history = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                num_patches_list=[1] * pixel_values.shape[0],
                history=None,
                return_history=True,
            )

        # Normalize output to a plain text string regardless of return container type
        text = response[0] if isinstance(response, (list, tuple)) else response
        return text, None
