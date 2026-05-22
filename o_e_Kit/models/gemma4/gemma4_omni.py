import os
import random
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

from o_e_Kit.utils.config_utils import load_config
from o_e_Kit.utils.utils import load_audio, load_video
from o_e_Kit.utils.video_utils import extract_audio_from_video


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../configs/generation_configs.json",
)


class Gemma4OmniEvalModel:
    """
    Gemma 4 多模态评测封装 (E2B / E4B)

    支持文本、图像、视频、音频的多模态理解，通过 HuggingFace Transformers 原生接口调用。
    视频帧提取复用 OmniEvalKit 的 load_video() 工具，音频加载复用 load_audio()。
    """

    def __init__(
        self,
        model_path: str,
        ckpt_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        config_path: Optional[str] = None,
        dataset_generation_config_path: Optional[str] = None,
        auto_device_map: bool = False,
        quantization: str = "none",
        attn_implementation: Optional[str] = None,
        max_inp_length: int = 32768,
        cpu_offload: bool = False,
    ) -> None:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        if dataset_generation_config_path:
            self.dataset_configs = load_config(dataset_generation_config_path)
            print(
                f"✅ Loaded omni dataset generation configs from: "
                f"{dataset_generation_config_path}"
            )
        else:
            self.dataset_configs = load_config(DEFAULT_CONFIG_PATH)
            print(
                f"✅ Using default omni dataset generation configs: "
                f"{DEFAULT_CONFIG_PATH}"
            )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path
        self.max_inp_length = max_inp_length

        self._init_model_and_processor(
            model_path,
            config_path,
            auto_device_map,
            quantization,
            attn_implementation,
            cpu_offload,
        )

        torch.cuda.empty_cache()

    def _init_model_and_processor(
        self,
        model_path: str,
        config_path: Optional[str],
        auto_device_map: bool,
        quantization: str,
        attn_implementation: Optional[str],
        cpu_offload: bool,
    ) -> None:
        print("Loading Gemma 4 model...")

        model_kwargs = {}

        if auto_device_map:
            model_kwargs["device_map"] = "auto"

        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        if config_path:
            model_kwargs["config"] = config_path

        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )

        if not auto_device_map:
            self.model = self.model.to(self.device)

        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            padding_side="left",
        )

        try:
            model_device = getattr(self.model, "device", None)
        except Exception:
            model_device = None

        if model_device is not None:
            self.device = model_device

        print(f"  device: {self.device}")
        try:
            print(f"  model dtype: {self.model.dtype}")
        except Exception:
            pass

    def get_generation_config(self, dataset_name: str) -> Dict[str, Any]:
        config = self.dataset_configs.get(dataset_name)
        if config is None:
            raise ValueError(
                f"No omni generation config found for dataset '{dataset_name}'"
            )

        return {
            "max_tokens": int(config.get("max_tokens", 256)),
            "user_prompt": config.get(
                "user_prompt", "{media}\n{question}\n{options}"
            ),
            "system_prompt": config.get("system_prompt", ""),
            "max_frames": int(config.get("max_frames", 64)),
            "max_fps": float(config.get("max_fps", 1.0)),
            "load_av": bool(config.get("load_av", False)),
            "num_beams": int(config.get("num_beams", 1)),
            "do_sample": bool(config.get("sampling", False)),
            "temperature": float(config.get("temperature", 1.0)),
            "top_p": float(config.get("top_p", 1.0)),
            "top_k": int(config.get("top_k", 50)),
            "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
        }

    def _build_options_prompt(self, choices: List[str]) -> str:
        if not choices:
            return ""
        keys = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        return "".join(
            f"{k}. {c}\n" for k, c in zip(keys[: len(choices)], choices)
        )

    def _extract_audio_from_video(self, video_path: str) -> Optional[np.ndarray]:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            audio_path = extract_audio_from_video(video_path, output_path=tmp_path)
            waveform = load_audio(audio_path, sr=16000)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return waveform
        except Exception as e:
            print(f"Audio extraction from video failed: {e}")
            return None

    def build_messages(
        self, dataset_name: str, paths: Dict[str, Any], item: Dict[str, Any]
    ) -> (List[Dict[str, Any]], Dict[str, Any]):
        gen_config = self.get_generation_config(dataset_name)
        user_prompt_template = gen_config["user_prompt"]
        system_prompt = gen_config["system_prompt"]
        max_frames = gen_config["max_frames"]
        max_fps = gen_config["max_fps"]
        load_av = gen_config["load_av"]

        question = item.get("question", item.get("prompt", ""))
        choices = item.get("choices", [])
        options_prompt = self._build_options_prompt(choices)
        sqa_context = item.get("sqa_context", "")

        prompt = (
            user_prompt_template.replace("{question}", question)
            .replace("{options}", options_prompt.rstrip())
            .replace("{sqa_context}", sqa_context)
            .replace("{media}", "")
            .strip()
        )

        messages: List[Dict[str, Any]] = []
        user_content: List[Dict[str, Any]] = []

        video_path = paths.get("video_path")
        if video_path and os.path.exists(video_path):
            frames = load_video(video_path, max_frames=max_frames, max_fps=max_fps)
            if frames:
                user_content.append({"type": "video", "video": frames})
            if load_av:
                waveform = self._extract_audio_from_video(video_path)
                if waveform is not None:
                    user_content.append({"type": "audio", "audio": waveform})

        image_path = paths.get("image_path")
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            user_content.append({"type": "image", "image": image})

        audio_path = paths.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            waveform = load_audio(audio_path, sr=16000)
            user_content.append({"type": "audio", "audio": waveform})

        audio_paths_dict = paths.get("audio_paths_dict") or {}
        for _, p in audio_paths_dict.items():
            if p and os.path.exists(p):
                waveform = load_audio(p, sr=16000)
                user_content.append({"type": "audio", "audio": waveform})

        image_paths_dict = paths.get("image_paths_dict") or {}
        for _, p in image_paths_dict.items():
            if p and os.path.exists(p):
                image = Image.open(p).convert("RGB")
                user_content.append({"type": "image", "image": image})

        video_paths_dict = paths.get("video_paths_dict") or {}
        for _, p in video_paths_dict.items():
            if p and os.path.exists(p):
                frames = load_video(p, max_frames=max_frames, max_fps=max_fps)
                if frames:
                    user_content.append({"type": "video", "video": frames})

        if prompt:
            user_content.append({"type": "text", "text": prompt})

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        messages.append({"role": "user", "content": user_content})
        return messages, gen_config

    def generate(
        self,
        dataset_name: str,
        paths: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
        modality: str = "omni",
        **unused_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if modality not in {"omni", "audio", "video", "image"}:
            print(f"[Gemma4] Unexpected modality: {modality}, treat as omni.")

        for path, item in zip(paths, items):
            messages, gen_config = self.build_messages(dataset_name, path, item)

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                processor_kwargs={"do_sample_frames": False},
            ).to(self.device, dtype=self.model.dtype)

            try:
                input_len = inputs["input_ids"].shape[-1]
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=int(gen_config.get("max_tokens", 256)),
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    num_beams=int(gen_config.get("num_beams", 1)),
                    do_sample=bool(gen_config.get("do_sample", False)),
                    temperature=float(gen_config.get("temperature", 1.0)),
                    top_p=float(gen_config.get("top_p", 1.0)),
                    top_k=int(gen_config.get("top_k", 50)),
                    repetition_penalty=float(gen_config.get("repetition_penalty", 1.0)),
                )
                response = self.processor.decode(
                    output_ids[0][input_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except Exception as e:
                print(f"[Gemma4] Generation failed: {e}")
                response = ""

            results.append(
                {
                    "response": response,
                    "sequence": str(messages),
                }
            )

        return results
