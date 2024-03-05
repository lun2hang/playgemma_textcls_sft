#import packages
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import bitsandbytes
from peft import TaskType, get_peft_model, LoraConfig

#This python script is a rl project for gemmaxb-it. 
#It will use the 1/lengths of summarization of imdb comment to make the response more concise

#config
dataset_imdb_local = "/DATA/jupyter/personal/imdb/plain_text"
gemma_2b_it_local = "/DATA/jupyter/personal/gemma-2b-it"
gemma_2b_it_ppo_local = "/DATA/jupyter/personal/gemma-2b-it-lora-ppo"

summarization_number_eval = 1
instruct = ".Preceding text is a imdb movie review.Please help summarize it."

#load dataset imdb
imdb = load_dataset(
    path = "parquet", 
    data_dir = dataset_imdb_local, 
    data_files = {'unsupervised': 'unsupervised-00000-of-00001.parquet'}
    )

#use gemma2b-it to summarize the comments
tokenizer = AutoTokenizer.from_pretrained(gemma_2b_it_local)
pipeline = pipeline(
    "text-generation",
    model=gemma_2b_it_local,
    model_kwargs={
            "torch_dtype": torch.bfloat16,
        },
    device="cuda",
)
'''
    使用float16加载、量化加载,使用量化加载时默认cuda，需要去掉device配置

    model_kwargs={
            "torch_dtype": torch.bfloat16,
            "quantization_config": {"load_in_4bit": True}
        },
'''

summarization_len = 0
for i in range(summarization_number_eval):
    comments = imdb["unsupervised"][i]['text']
    messages = [
        {"role": "user", "content": comments + instruct}
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
    summarization_len += len(outputs[0]["generated_text"][len(prompt):])
#   print(comments)
#   print(outputs[0]["generated_text"][len(prompt):])

#calculate len_avg
summarization_len /= summarization_number_eval
print(summarization_len)

#Build lora model for training，量化在计算时需要反量化，慢但是省显存:加参数 quantization_config = {"load_in_4bit": True}
model = AutoModelForCausalLM.from_pretrained(gemma_2b_it_local, device_map="auto", torch_dtype=torch.bfloat16)

# 创建loRA参数
config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules={"q_proj", "k_proj", "v_proj", "o_proj"}, r=8, lora_alpha=32)
# 模型合并
model = get_peft_model(model, config)

# 加载Lora后的推理测试,通过
'''
input_text = imdb["unsupervised"][0]['text'] + instruct
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=256)
print(input_text)
print(tokenizer.decode(outputs[0]))
'''

#PPO trainer，rewards =  / len_response,the short the better


#save model

#load trained model to summarize the comments

#calculate len_avg and check if it is shorter

print("end")