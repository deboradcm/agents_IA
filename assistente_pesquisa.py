import os
import pandas as pd
from datasets import load_dataset
from pymongo import MongoClient

import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREWORKS_API_KEY"] = ""
os.environ["MONGO_URI"] = "mongodb+srv://deboramedeiros:mecanica33@cluster0.6rm44.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

data = load_dataset("MongoDB/subset_arxiv_papers_with_emebeddings") # carregando um dataset do Hugging Face Datasets
dataset_df = pd.DataFrame(data["train"])

print(dataset_df.head())

# Initialize MongoDB python client
client = MongoClient(MONGO_URI, appname="devrel.content.ai_agent_firechain.python")

DB_NAME = "agent_demo"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]

