from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")

system_msg = SystemMessage(content="You are the helpful assistance that always answers in exactly one senetence")
human_msg =  HumanMessage(content="What is the capital of Japan?")

response = llm.invoke([system_msg, human_msg])
print(response.content)

