"""Local backend — Qwen2.5-VL-7B-Instruct (transformers).

One model instance serves both roles:
  * ``describe`` — board image(s) + prompt -> text (VL stage),
  * ``complete`` — text-only system+user -> text (synthesis stage),
so loading it once covers the whole free/offline path. 4-bit quantization keeps it
inside a Kaggle T4's 16 GB alongside nothing else (stages run sequentially).
"""

from __future__ import annotations

from typing import Any, Dict, List


class LocalQwenBackend:
    def __init__(self, cfg: Dict[str, Any]):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_id = cfg.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1024))

        load_kwargs: Dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
        if cfg.get("load_in_4bit", True):
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        print(f"[local_qwen] loading {self.model_id} (4bit={cfg.get('load_in_4bit', True)})")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, **load_kwargs
        )
        # Cap vision tokens: a full-res board screenshot can otherwise expand into tens of
        # thousands of tokens and OOM a T4. ~1.0 MP (max_pixels) is plenty to read a board.
        min_pixels = int(cfg.get("min_pixels", 256 * 28 * 28))
        max_pixels = int(cfg.get("max_pixels", 1280 * 28 * 28))
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def _generate(self, messages: List[Dict[str, Any]]) -> str:
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # -- VLBackend ------------------------------------------------------------
    def describe(self, image_paths: List[str], prompt: str) -> str:
        content: List[Dict[str, Any]] = [{"type": "image", "image": p} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        return self._generate([{"role": "user", "content": content}])

    # -- LLMBackend -----------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        return self._generate(messages)
