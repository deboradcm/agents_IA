from transformers import AutoModelForSequenceClassification, AutoTokenizer

#Carregar um modelo pré-treinado usando a biblioteca transformers
model_name = "Qwen/Qwen2-1.5B-Instruct"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)