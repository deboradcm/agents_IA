from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.tavily_search import TavilySearchResults

#estado armazena as mensagens entre o usuário e o agente.
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

#Utilizando o modelo de LLM Claude para responder às perguntas.
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

#Compilar e Visualizar o Gráfico
graph = graph_builder.compile()
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

#Adicionando Ferramentas para Buscar na Web
tool = TavilySearchResults(max_results=2)
llm_with_tools = llm.bind_tools([tool])

def chatbot_with_tools(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot_with_tools", chatbot_with_tools)

#Definir Lógica Condicional
def route_tools(state: State):
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"

graph_builder.add_conditional_edges("chatbot_with_tools", route_tools, {"tools": "tools", "__end__": END})
graph_builder.add_edge("tools", "chatbot_with_tools")

#Executar o Agente
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        break
    stream_graph_updates(user_input)
