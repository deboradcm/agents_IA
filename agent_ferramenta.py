# pip install -Uq arxiv

#from langchain_community.tools import ArxivQueryRun
#tool = ArxivQueryRun()
#print(tool.invoke('Photosynthesis')[:250])


import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY não foi definida.")

# Carregar as ferramentas
tools = load_tools(
    ["arxiv", "dalle-image-generator"], 
    dalle_api_key=api_key  # Passa explicitamente a chave para a DALL-E
)

# Chamar a ferramenta arXiv
print(tools[0].invoke("Kaggle")[:150])

from langchain_community.agent_toolkits.load_tools import get_all_tool_names
print(get_all_tool_names()[:10]) # 

# Generate an image with DallE
output_image_url = tools[1].invoke("A magnificent chain in a cartoon.")
output_image_url
print(output_image_url)