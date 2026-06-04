import torch

from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_PATH = "google/gemma-4-12B-it"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    attn_implementation="sdpa",
)
model = model.to(device)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
            {"type": "text", "text": "Tell me what you see, then transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            {"type": "audio", "audio": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/journal1.wav"}
        ],
    },
]

# Apply chat template and tokenize in one step (processor handles URL download internally)
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(device, dtype=model.dtype)

# Generate text response (Gemma 4 outputs text only, no audio/TTS)
input_len = inputs["input_ids"].shape[-1]
output_ids = model.generate(**inputs, max_new_tokens=256)

# Decode with special tokens so parse_response can extract the clean answer
response = processor.decode(
    output_ids[0][input_len:],
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False,
)

try:
    result = processor.parse_response(response)
    if isinstance(result, dict):
        text = result.get("content") or result.get("text") or ""
    else:
        text = str(result)
except (AttributeError, Exception):
    text = response

print(text.strip())
