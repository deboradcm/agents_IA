import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage 
from langchain_core.prompts import PromptTemplate



# Carregar as variáveis do arquivo .env
load_dotenv()

# Acessar a chave da API
openai_api_key = os.getenv("OPENAI_API_KEY")

#llm = OpenAI(openai_api_key=openai_api_key)
#question = "Messi é o melhor jogador de futebol de todos os tempos?"
#output = llm.invoke(question)
#print(output[:300])

#llm = OpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-instruct")
#question = "What is special about the number 73?"
#output = llm.invoke(question)
#print(output[:100])

chat_model = ChatOpenAI(openai_api_key=openai_api_key, model='gpt-4o-mini')
messages = [SystemMessage(content='Você é um pirata mal-humorado.'),
           HumanMessage(content="E aí?")]
output = chat_model.invoke(messages)
print(type(output)) # Verificar o tipo da resposta
print(output.content)  #Exibe a mensagem de texto que o modelo produziu como resposta à interação
print(output.dict()) #Exibe o conteúdo da resposta em um formato de dicionário, para que seja possível analisar ou manipular a resposta de forma programática.

