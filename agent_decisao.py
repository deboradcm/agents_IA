#Um agente capaz de explicar qualquer tópico por meio de três mídias: texto, imagem ou vídeo. 
#Mais especificamente, com base na pergunta feita, o agente decidirá se deve explicar o tópico em qual formato.

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage 


from langchain_community.tools import WikipediaQueryRun  # pip install wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool  # pip install youtube_search
from langchain_community.tools.openai_dalle_image_generation import (
   OpenAIDALLEImageGenerationTool
)
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

from langgraph.prebuilt import create_react_agent
from pprint import pprint


wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(description="A tool to explain things in text format. Use this tool if you think the user’s asked concept is best explained through text.", api_wrapper=wiki_api_wrapper)
print(wikipedia.invoke("Mobius strip"))

dalle_api_wrapper = DallEAPIWrapper(model="dall-e-3", size="1792x1024")
dalle = OpenAIDALLEImageGenerationTool(
   api_wrapper=dalle_api_wrapper, description="A tool to generate images. Use this tool if you think the user’s asked concept is best explained through an image."
)
output = dalle.invoke("A mountain bike illustration.")
print(output)

youtube = YouTubeSearchTool(
   description="A tool to search YouTube videos. Use this tool if you think the user’s asked concept can be best explained by watching a video."
)
result = youtube.run("Oiling a bike's chain")
print("YouTube Search Result:", result)

tools = [wikipedia, dalle, youtube]

chat_model = ChatOpenAI(api_key=api_key)
model_with_tools = chat_model.bind_tools(tools)

response = model_with_tools.invoke([HumanMessage("What's up?!")])
print(f"Text response: {response.content}")
print(f"Tools used in the response: {response.tool_calls}")

response = model_with_tools.invoke([
   HumanMessage("Can you generate an image of a mountain bike?")
])
print(f"Text response: {response.content}")
print(f"Tools used in the response: {response.tool_calls}")

system_prompt = SystemMessage("You are a helpful bot named Chandler.")
agent = create_react_agent(chat_model, tools, state_modifier=system_prompt)

response = agent.invoke({"messages": HumanMessage("What's up?")})
pprint(response["messages"])

response = agent.invoke({"messages": [
   HumanMessage('Explain how photosynthesis works.')
]})
print(len(response['messages']))

for message in response['messages']:
   print(
       f"{message.__class__.__name__}: {message.content}"
   )  # Print message class name and its content
   print("-" * 20, end="\n")

def execute(agent, query):
   response = agent.invoke({'messages': [HumanMessage(query)]})
  
   for message in response['messages']:
       print(
           f"{message.__class__.__name__}: {message.content}"
       )  # Print message class name and its content
      
       print("-" * 20, end="\n")
  
   return response

system_prompt = SystemMessage(
   """
   You are a helpful bot named Chandler. Your task is to explain topics
   asked by the user via three mediums: text, image or video.
  
   If the asked topic is best explained in text format, use the Wikipedia tool.
   If the topic is best explained by showing a picture of it, generate an image
   of the topic using Dall-E image generator and print the image URL.
   Finally, if video is the best medium to explain the topic, conduct a YouTube search on it
   and return found video links.
   """
)


agent = create_react_agent(chat_model, tools, state_modifier=system_prompt)
response = execute(agent, query='Explain the Fourier Series visually.')

# Adicionando memoria no agente
response = execute(agent, query="What did I ask you in the previous query?")

