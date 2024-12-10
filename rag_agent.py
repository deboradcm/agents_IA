from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
import os

from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from transformers import DPRQuestionEncoder, DPRContextEncoder
from transformers import T5ForConditionalGeneration, T5Tokenizer

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Inicializando os modelos
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def expand_query(query):
    input_text = f"expand query: {query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=3)
    expanded_queries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return expanded_queries


# Função para codificar passagens
def encode_passages(passages):
    # Garantir que as passagens sejam passadas como lista de strings
    if isinstance(passages, list):
        passages = [str(passage) for passage in passages]
    else:
        passages = [str(passages)]

    # Tokenizando as passagens com max_length e truncação
    inputs = tokenizer(passages, max_length=512, truncation=True, padding=True, return_tensors="pt")
    return context_encoder(**inputs).pooler_output



# Function to encode query
def encode_query(query):
    # Tokenizar a consulta primeiro
    inputs = tokenizer(query, max_length=512, truncation=True, padding=True, return_tensors="pt")
    
    # Passar os inputs tokenizados para o encoder de consulta
    return question_encoder(**inputs).pooler_output

# Preparar os Documentos

# Carregar o documento
loader = TextLoader("/home/iartes/agents_IA/Knowledge_base.txt")
documents = loader.load()

# Verificar se os documentos têm o atributo 'page_content'
for doc in documents:
    if isinstance(doc, Document):
        print(f"Conteúdo do documento: {doc.page_content[:200]}")  # Exibir um trecho do conteúdo
    else:
        print("Documento não é do tipo esperado.")

# Dividir os documentos em blocos
text_splitter = CharacterTextSplitter(chunk_size=1010, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Codificar passagens e consultas
encoded_passages = [encode_passages(doc) for doc in texts]
encoded_queries = [encode_query("Qual é a diferença entre aprendizado supervisionado e não supervisionado?")]

# Criar embeddings
#embeddings = OpenAIEmbeddings()

# Criar um armazenamento vetorial
vectorstore = Chroma.from_documents(encoded_passages)

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

def iterative_retrieval(initial_query, max_iterations=3):
    query = initial_query
    for _ in range(max_iterations):
        result = qa.run(query)
        clarification = agent.run(f"Based on this result: '{result}', what follow-up question should I ask to get more specific information?")
        if clarification.lower().strip() == "none":
            break
    query = clarification
    return result
# Use this in your agent's process

# Consultar o agente
query = "Qual é a diferença entre aprendizado supervisionado e não supervisionado? Além disso, qual é 15% de 80?"
result = qa.invoke(query) 
print(result)








