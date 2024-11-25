import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools

# Carregar variáveis de ambiente
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY não foi definida.")

# Carregar a ferramenta DALL-E
dalle_tool = load_tools(["dalle-image-generator"], dalle_api_key=api_key)[0]  # Ferramenta DALL-E

# Gerar uma imagem com DALL-E
output_image_url = dalle_tool.invoke("A wicked witch")
print("Image URL:", output_image_url)
