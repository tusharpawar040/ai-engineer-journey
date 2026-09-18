from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages":response}

workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user1"}}

response1 = app.invoke({"messages":[HumanMessage("My name is Tush")]},config=config)
print(response1["messages"][-1].content)

config2 = {"configurable": {"thread_id": "user2"}}
response2 = app.invoke({"messages":[HumanMessage("What is my name?")]},config=config2)
print(response2["messages"][-1].content)

