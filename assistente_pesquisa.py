import os
import pandas as pd
from datasets import load_dataset
from pymongo import MongoClient
from urllib.parse import quote_plus
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor
from langchain_fireworks import ChatFireworks
from langchain.agents import tool
from langchain_community.document_loaders import ArxivLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_compressors import LLMLinguaCompressor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
import openai


os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREWORKS_API_KEY"] = ""
os.environ["MONGO_URI"] = "mongodb+srv://deboramedeiros:rxqTetnCfLWCXTzH@cluster0.6rm44.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tls=true&connectTimeoutMS=30000&socketTimeoutMS=60000"

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

data = load_dataset("MongoDB/subset_arxiv_papers_with_emebeddings") # carregando um dataset do Hugging Face Datasets
dataset_df = pd.DataFrame(data["train"])

print(dataset_df.head())

# Inicializar o cliente Python do MongoDB
client = MongoClient(MONGO_URI, appname="devrel.content.ai_agent_firechain.python")

DB_NAME = "agent_demo"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]

#Excluir quaisquer registros existentes na coleção
collection.delete_many({})

#Ingestão de Dados
records = dataset_df.to_dict('records')
collection.insert_many(records)

print("Ingestão de dados concluída no MongoDB")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)

# Criação de Armazenamento de Vetores
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding= embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="abstract"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# Configurar LLM usando Fireworks AI
llm = ChatFireworks(
    model="accounts/fireworks/models/firefunction-v1",
    max_tokens=500)

# Criação de ferramentas de agente

# Definição de ferramenta personalizada
@tool
def get_metadata_information_from_arxiv(word: str) -> list:
  """
  Busca e retorna metadados de no máximo dez documentos do arXiv que correspondem à palavra-chave fornecida na consulta.

  Argumentos:
    word (str): A consulta de pesquisa para encontrar documentos relevantes no arXiv.

  Retorna:
    list: Metadados sobre os documentos que correspondem à consulta.
  """

  docs = ArxivLoader(query=word, load_max_docs=10).load()
  # Extraia apenas os metadados de cada documento
  metadata_list = [doc.metadata for doc in docs]
  return metadata_list


@tool
def get_information_from_arxiv(word: str) -> list:
    """
    Busca e retorna metadados de um único artigo de pesquisa do arXiv que corresponde à palavra-chave fornecida.

    Argumentos:
        word (str): O ID do artigo do arXiv (por exemplo, '704.0001').

    Retorna:
        list: Dados sobre o artigo que corresponde ao ID fornecido.
    """
    doc = ArxivLoader(query=word, load_max_docs=1).load()
    return doc


# Se você criou um recuperador com recursos de compactação na célula opcional de uma célula anterior, poderá substituir 'retriever' por 'compression_retriever'
# Caso contrário, você também pode criar um procedimento de compactação como uma ferramenta para o agente, conforme mostrado na função de definição da ferramenta `compress_prompt_using_llmlingua`
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="knowledge_base",
    description="Isso serve como a fonte de conhecimento base do agente e contém alguns registros de artigos de pesquisa do Arxiv. Esta ferramenta é usada como o primeiro passo para esforços de exploração e pesquisa"
)

compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")

@tool
def compress_prompt_using_llmlingua(prompt: str, compression_rate: float = 0.5) -> str:
    """
    Comprime um dado ou prompt longo usando o LLMLinguaCompressor.

    Argumentos:
        data (str): O dado ou prompt a ser comprimido.
        compression_rate (float): A taxa de compressão dos dados (o padrão é 0.5).

    Retorna:
        str: O dado ou prompt comprimido.
    """
    compressed_data = compressor.compress_prompt(
        prompt,
        rate=compression_rate,
        force_tokens=["!", ".", "?", "\n"],
        drop_consecutive=True
    )
    return compressed_data

tools = [retriever_tool, get_metadata_information_from_arxiv, get_information_from_arxiv, compress_prompt_using_llmlingua]

# Criação de prompt do agente

agent_purpose = """
Você é um assistente de pesquisa útil equipado com várias ferramentas para ajudar com suas tarefas de forma eficiente.
Você tem acesso ao histórico de conversas armazenado em sua entrada como chat_history.
Você é econômico e utiliza a ferramenta compress_prompt_using_llmlingua sempre que determinar que um prompt ou o histórico de conversação é muito longo.
Você traduz sua resposta final para pt-br.
Abaixo estão as instruções sobre quando e como usar cada ferramenta em suas operações.

1. get_metadata_information_from_arxiv

Objetivo: Buscar e retornar metadados de até dez documentos do arXiv que correspondem a uma palavra-chave fornecida.
Quando Usar: Use esta ferramenta quando precisar reunir metadados sobre vários artigos de pesquisa relacionados a um tópico específico.
Exemplo: Se você for solicitado a fornecer uma visão geral dos artigos recentes sobre "machine learning", use esta ferramenta para buscar metadados de documentos relevantes.

2. get_information_from_arxiv

Objetivo: Buscar e retornar metadados de um único artigo de pesquisa do arXiv usando o ID do artigo.
Quando Usar: Use esta ferramenta quando precisar de informações detalhadas sobre um artigo específico identificado pelo seu ID do arXiv.
Exemplo: Se você for solicitado a recuperar informações detalhadas sobre o artigo com o ID "704.0001", use esta ferramenta.

3. retriever_tool

Objetivo: Servir como seu conhecimento base, contendo registros de artigos de pesquisa do arXiv.
Quando Usar: Use esta ferramenta como o primeiro passo para exploração e esforços de pesquisa ao lidar com tópicos cobertos pelos documentos na base de conhecimento.
Exemplo: Ao iniciar uma pesquisa sobre um novo tópico bem documentado no repositório arXiv, use esta ferramenta para acessar os artigos relevantes.

4. compress_prompt_using_llmlingua

Objetivo: Comprimir prompts longos ou históricos de conversação usando o LLMLinguaCompressor.
Quando Usar: Use esta ferramenta sempre que determinar que um prompt ou histórico de conversação é muito longo para ser processado de forma eficiente.
Exemplo: Se você receber uma consulta ou contexto de conversação muito longo que exceda os limites típicos de tokens, compacte-o usando esta ferramenta antes de prosseguir com o processamento.

"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_purpose),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

# Memória do agente usando MongoDB

def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(MONGO_URI, session_id, database_name=DB_NAME, collection_name="history")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=get_session_history("latest_agent_session")
)

# Criação de Agente

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
)

# Execução de Agente

agent_executor.invoke({"input": "Obtenha uma lista de artigos de pesquisa sobre o tópico tráfico humano]."})
