from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
import os

from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from transformers import DPRQuestionEncoder, DPRContextEncoder

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Preparar os Documentos

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

# Criar uma cadeia de perguntas e respostas baseada em recuperação
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())

# Definir a ferramenta de cálculo
class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Useful for when you need to answer questions about math"
    
    def _run(self, query: str):
        try:
            return str(eval(query))
        except:
            return "I couldn't calculate that. Please make sure your input is a valid mathematical expression."

# Criar instâncias das ferramentas
search = DuckDuckGoSearchRun()
calculator = CalculatorTool()

# Definir as ferramentas
tools = [
    Tool(name="Search", func=search.run, description="Useful for when you need to answer questions about current events"),
    Tool(name="RAG-QA", func=qa.run, description="Useful for when you need to answer questions about AI and machine learning"),
    Tool(name="Calculator", func=calculator._run, description="Useful for when you need to perform mathematical calculations")
]

# Inicializar o agente
agent = initialize_agent(tools, OpenAI(temperature=0), agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Consultar o agente
query = "Qual é a diferença entre aprendizado supervisionado e não supervisionado? Além disso, qual é 15% de 80?"
result = qa.invoke(query) 
print(result)





