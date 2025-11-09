from transformers import AutoModelForCausalLM, AutoTokenizer
import os


model = AutoModelForCausalLM.from_pretrained(
  "Qwen/Qwen3-VL-8B-Instruct",
  device_map="cuda",
  torch_dtype='auto',
  attn_implementation="eager",
  trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-8B-Instruct")

from transformers import pipeline

generator = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  return_full_text=False,
  max_new_tokens=500,
  do_sample=False,
)

messages = [
  {
    "role": "user",
    "content": "치킨으로 재밌는 농담 하나만 해줘"
  }
]

output = generator(messages)
print(output[0]["generated_text"])