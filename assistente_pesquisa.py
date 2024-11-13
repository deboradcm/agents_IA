import os
import pandas as pd
from datasets import load_dataset

import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREWORKS_API_KEY"] = ""
os.environ["MONGO_URI"] = ""

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

data = load_dataset("MongoDB/subset_arxiv_papers_with_emebeddings")
dataset_df = pd.DataFrame(data["train"])

print(dataset_df.head())
