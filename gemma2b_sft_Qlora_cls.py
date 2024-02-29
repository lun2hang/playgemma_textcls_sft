from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
from peft import TaskType, get_peft_model, LoraConfig

dataset_imdb_local = "/DATA/jupyter/personal/imdb/plain_text"
gemma_2b_local = "/DATA/jupyter/personal/gemma-2b"
gemma_2b_sft_local = "/DATA/jupyter/personal/gemma-2b-sft"
metrics_local = "/DATA/jupyter/personal/playgemma/metrics/accuracy"

imdb = load_dataset(
    path = "parquet", 
    data_dir = dataset_imdb_local, 
    data_files = {'train': 'train-00000-of-00001.parquet', 'test': 'test-00000-of-00001.parquet'}
    )

print(imdb["test"][0])

tokenizer = AutoTokenizer.from_pretrained(gemma_2b_local)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length = 2048)
tokenized_imdb = imdb.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load(metrics_local)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    gemma_2b_local, 
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id, 
    torch_dtype = torch.bfloat16, 
    #量化有反量化过程，慢但是省显存
#    quantization_config = {"load_in_4bit": True}
    )

# 创建loRA参数
config = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules={"q_proj", "k_proj", "v_proj", "o_proj"}, r=8, lora_alpha=32)
# 模型合并
model = get_peft_model(model, config)


training_args = TrainingArguments(
    output_dir= gemma_2b_sft_local,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print("end")
