from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")

@tool
def get_weather(city: str)-> str:
    """Get the current weather for a city."""
    return f"It's sunny in city {city}."

llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("What's the weather in Pune?")
if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = get_weather.invoke(tool_call["args"])
    print("tool result:", result)

messages = [
    HumanMessage(content="What's the weather in Pune?"),
    response,
    ToolMessage(content=result, tool_call_id=tool_call["id"])
]
final_response = llm_with_tools.invoke(messages)
print("final answer: ",final_response.content)
print("Content: ",response.content)
print("tool_calls: ",response.tool_calls)
