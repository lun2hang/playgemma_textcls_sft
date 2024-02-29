from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch

dataset_imdb_local = "/DATA/jupyter/personal/imdb/plain_text"
model_distilbert_base_uncased_local = "/DATA/jupyter/personal/distilbert-base-uncased"
model_distilbert_base_uncased_sft_local = "/DATA/jupyter/personal/distilbert-base-uncased-sft"
metrics_local = "/DATA/jupyter/personal/playgemma/metrics/accuracy"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

imdb = load_dataset(
    path = "parquet", 
    data_dir = dataset_imdb_local, 
    data_files = {'train': 'train-00000-of-00001.parquet', 'test': 'test-00000-of-00001.parquet'}
    )

imdb["test"][0]

tokenizer = AutoTokenizer.from_pretrained(model_distilbert_base_uncased_local)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length = 512)
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
    model_distilbert_base_uncased_local, num_labels=2, id2label=id2label, label2id=label2id
    )

training_args = TrainingArguments(
    output_dir= model_distilbert_base_uncased_sft_local,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
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

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

from transformers import pipeline
classifier = pipeline("sentiment-analysis", model = model_distilbert_base_uncased_sft_local )
classifier(text)

