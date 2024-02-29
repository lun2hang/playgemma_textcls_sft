from transformers import AutoTokenizer, pipeline
import torch
import bitsandbytes
import accelerate

model = "/DATA/jupyter/personal/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={
            "torch_dtype": torch.bfloat16,
        },
    device="cuda",
)

'''
    使用float16加载、量化加载
    model_kwargs={
            "torch_dtype": torch.bfloat16,
            "quantization_config": {"load_in_4bit": True}
        },
'''

messages = [
    {"role": "user", "content": "Who are you? Please, answer in English."},
]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
print(outputs[0]["generated_text"][len(prompt):])
