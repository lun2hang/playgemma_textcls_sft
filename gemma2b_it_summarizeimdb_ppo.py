#import packages
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import bitsandbytes
from peft import TaskType, get_peft_model, LoraConfig, PeftConfig, PeftModel
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

#This python script is a rl project for gemmaxb-it. 
#It will use the 1/lengths of summarization of imdb comment to make the response more concise

#config
dataset_imdb_local = "/DATA/jupyter/personal/imdb/plain_text"
gemma_2b_it_local = "/DATA/jupyter/personal/gemma-2b-it"
gemma_2b_it_ppo_local = "/DATA/jupyter/personal/gemma-2b-it-lora-ppo"

summarization_number_baseline = 100
summarization_number_eval = 100
instruct = ".Preceding text is a imdb movie review.Please help summarize it."

ppo_training_steps = 20
ppo_rewards_textlen_threshhold = 500

#load dataset imdb
imdb = load_dataset(
    path = "parquet", 
    data_dir = dataset_imdb_local, 
    data_files = {'unsupervised': 'unsupervised-00000-of-00001.parquet'}
    )

#use gemma2b-it to summarize the comments
tokenizer = AutoTokenizer.from_pretrained(gemma_2b_it_local)
tokenizer.pad_token = tokenizer.eos_token

textgen = pipeline(
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
for i in range(summarization_number_baseline):
    comments = imdb["unsupervised"][i]['text']
    messages = [
        {"role": "user", "content": comments + instruct}
    ]
    prompt = textgen.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = textgen(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    summarization_len += len(outputs[0]["generated_text"][len(prompt):])
    print("step: %d, gen_len_avg: %f \n" % (i, summarization_len/(i+1)))
#   print(comments)
#   print(outputs[0]["generated_text"][len(prompt):])

#calculate len_avg
summarization_len /= summarization_number_baseline
print("the avg summarization length BEFORE PPO: %d" % summarization_len)
summarization_len_before = summarization_len

# 创建loRA参数
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules={"q_proj", "k_proj", "v_proj", "o_proj"}, r=8, lora_alpha=32)
#Build lora model for training，量化在计算时需要反量化，慢但是省显存:加参数 quantization_config = {"load_in_4bit": True}
model = AutoModelForCausalLMWithValueHead.from_pretrained(gemma_2b_it_local, device_map="auto", torch_dtype=torch.bfloat16, peft_config=lora_config)


# 加载Lora后的推理测试,通过
'''
input_text = imdb["unsupervised"][0]['text'] + instruct
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=256)
print(input_text)
print(tokenizer.decode(outputs[0]))
'''
#PPO trainer，rewards =  / len_response,the short the better
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, ref_model=None, tokenizer=tokenizer)

#training loop
train_data_length = imdb['unsupervised'].num_rows

for i in range(ppo_training_steps):
    data_index = i % train_data_length
    query_text = imdb["unsupervised"][data_index]['text'] + instruct
    query_tensor = tokenizer.encode(query_text, return_tensors="pt").to("cuda")
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 256,
    }
    response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
    response_txt = tokenizer.decode(response_tensor[0])
    #rewards 低于基线输出长度有rewards，越短越好
    reward = 1.0 - min(ppo_rewards_textlen_threshhold,len(response_txt))/ppo_rewards_textlen_threshhold
    print("train step: %d reward: %f \n" % (i,reward))
    reward = [torch.tensor(reward, device=model.pretrained_model.device)]

    #参数需要是list
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
#save model
model.save_pretrained(gemma_2b_it_ppo_local)

#load trained model to summarize the comments
peft_config = PeftConfig.from_pretrained(gemma_2b_it_ppo_local)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
inference_model = PeftModel.from_pretrained(base_model, gemma_2b_it_ppo_local)

textgen = pipeline(
    "text-generation",
    model=inference_model,
    tokenizer=tokenizer,
    model_kwargs={
            "torch_dtype": torch.bfloat16,
        },
    device="cuda",
)

summarization_len = 0
for i in range(summarization_number_eval):
    comments = imdb["unsupervised"][i]['text']
    #对话模型使用pipeline推理需要拼对话格式，否则原文输出。这是为了便于管理对话上下文
    messages = [
        {"role": "user", "content": comments + instruct}
    ]
    prompt = textgen.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = textgen(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    summarization_len += len(outputs[0]["generated_text"][len(prompt):])
    print("step: %d, gen_len_avg: %f \n" % (i, summarization_len/(i+1)))
#    print(comments)
#    print(outputs[0]["generated_text"][len(prompt):])

#calculate len_avg
summarization_len /= summarization_number_eval
print("the avg summarization length AFTER PPO: %d" % summarization_len)
print("the avg summarization length BEFORE PPO: %d" % summarization_len_before)
#calculate len_avg and check if it is shorter

print("end")