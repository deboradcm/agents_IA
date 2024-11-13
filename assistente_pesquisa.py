import os
import pandas as pd
from datasets import load_dataset
from pymongo import MongoClient
from urllib.parse import quote_plus


os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREWORKS_API_KEY"] = ""








data = load_dataset("MongoDB/subset_arxiv_papers_with_emebeddings") # carregando um dataset do Hugging Face Datasets
dataset_df = pd.DataFrame(data["train"])

print(dataset_df.head())

# Inicializar o cliente Python do MongoDB
client = MongoClient(MONGO_URI, appname="devrel.content.ai_agent_firechain.python")

DB_NAME = "agent_demo"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]

# Excluir quaisquer registros existentes na coleção
collection.delete_many({})

# Ingestão de Dados
records = dataset_df.to_dict('records')
collection.insert_many(records)

print("Ingestão de dados concluída no MongoDB")
