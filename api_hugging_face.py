"""
#Fazendo inferência em modelos os baixando localmente

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)  # Usando o modelo para geração de texto
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = []
for step in range(5):
    messages.append({"role": "user", "content": input('Fale algo: ')})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=500, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": generated_text.split('assistant')[-1].strip('\n')})
    print(f'==========\n{generated_text}\n')
"""

#Fazendo inferência em modelos hospedados na plataforma HuggingFace

from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("HUGGING_FACE")

api = InferenceClient(model="Qwen/Qwen2-1.5B-Instruct", token=api_key)

messages = [{"role": "user", "content": "Olá, tudo bom?"}]
response = api.chat_completion(messages)
print(response.choices[0].message.content)