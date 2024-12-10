from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
import os

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#Preparar os Documentos

# Carregar o documento
loader = TextLoader("/home/iartes/agents_IA/Knowledge_base.txt")
documents = loader.load()
# Dividir os documentos em blocos
text_splitter = CharacterTextSplitter(chunk_size=1010, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# Criar embeddings
embeddings = OpenAIEmbeddings()
# Criar um armazenamento vetorial
vectorstore = Chroma.from_documents(texts, embeddings)

# Criar uma cadeia de controle de qualidade baseada em recuperação

# Criar uma cadeia de perguntas e respostas baseada em recuperação
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())

# Consulta ao sistema

query = "Quais as diferenças entre oChatGPT e os agentes de IA autônomos"
result = qa.invoke(query)
print(result)


