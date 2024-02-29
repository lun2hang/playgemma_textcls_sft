from transformers import pipeline

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
model_distilbert_base_uncased_sft_local = "/DATA/jupyter/personal/distilbert-base-uncased-sft/checkpoint-3126"
classifier = pipeline("sentiment-analysis", model = model_distilbert_base_uncased_sft_local )
res = classifier(text)
print (res)
print("end")