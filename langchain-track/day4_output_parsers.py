from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")

class Capital(BaseModel):
    country: str
    capital: str

structured_llm = llm.with_structured_output(Capital)
result = structured_llm.invoke("What is the capital of France?")
print(result)
print(result.capital)
