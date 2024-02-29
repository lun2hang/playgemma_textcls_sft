from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

gemma2_2b_local = "/DATA/jupyter/personal/gemma-2b"
gemma2b_sft_local = "/DATA/jupyter/personal/gemma-2b-sft/checkpoint-3124"


peft_config = PeftConfig.from_pretrained(gemma2b_sft_local)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
inference_model = PeftModel.from_pretrained(base_model, gemma2b_sft_local)


classifier = pipeline("sentiment-analysis", model = inference_model, tokenizer =tokenizer)

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
res = classifier(text)
print(text)
print(res)


text1 = "this is the worst sic-fic movie,i have ever  seen"
res = classifier(text1)
print(text1)
print(res)

text2 = "this is the best sic-fic movie,i have ever seen.wish more movies like this to be made "
res = classifier(text2)
print(text2)
print(res)

print("end")