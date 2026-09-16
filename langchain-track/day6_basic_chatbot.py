from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")

messages = [SystemMessage(content="You are a friendly assistant.")]

while True:
    user_input = input("You: ")
    if user_input.lower() in ('quit','exit'):
        break
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages)
    print("Bot: ",response.content)
    messages.append(AIMessage(content=response.content))