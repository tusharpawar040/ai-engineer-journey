from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")
response = llm.invoke("What is national bird of india?")
print(response.content)
print("\n------full-response-object--------------")
print(response)
print("\n------usage-----------------------------")
print(response.usage_metadata)