import os
from dotenv import load_dotenv
from langchain_openai import OpenAI 
from langchain_core.prompts import ChatPromptTemplate

# Carregar a chave da API do arquivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Criar o modelo OpenAI com a chave da API
llm = OpenAI(api_key=api_key)

# Criar o template de chat
template = ChatPromptTemplate.from_messages([
    ("system", "Você é um bot de IA prestativo. Sua especialidade é {specialty}."),
    ("human", "Explique o conceito de {concept} com base na sua experiência.")
])

specialties = ["psychology", "economics", "politics"]
concept = "time"

# Iterar sobre as especialidades e gerar as respostas
for s in specialties:
    prompt = template.format_messages(specialty=s, concept=concept)
    output = llm.invoke(prompt)  # Chamar o modelo com o prompt preenchido
    print(output[:100], end="\n" + "-" * 25 + '\n')  # Exibir os primeiros 100 caracteres
