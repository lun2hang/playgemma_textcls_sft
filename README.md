This project include python scripts for 
1)sft distibert for sentiment analysis
2)Lora/Qlora peft gemma2b for sentiment analysis
3)accuracy 96%+ after 2 epochs

frameworks:transformers、peft、datasets、pytorch
models: distilbert、gemma-2b
datasets：imdb
platform：all scripts work on GPU 4090  with 24G mem,ubuntu linux
offline mode：all datasets、models and metrics are predownloaded,all the scripts run in offline mode
