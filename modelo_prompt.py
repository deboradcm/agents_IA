import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# Carregar as variáveis de ambiente
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Criar o modelo de linguagem
llm = OpenAI(api_key=api_key)

# Criar o template de prompt
query_template = "Fale sobre {book_name} de {author}."
prompt = PromptTemplate(input_variables=["book_name", "author"], template=query_template)

# Criar a chain conectando o prompt ao modelo
chain = prompt | llm

# Invocar a chain com os parâmetros
output = chain.invoke({"book_name": "Deathly Hallows", "author": "J.K. Rowling"})

# Exibir o resultado
print(output[:100])  # Exibe os primeiros 100 caracteres da resposta
print(type(chain))