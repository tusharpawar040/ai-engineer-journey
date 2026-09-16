from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")

template = PromptTemplate.from_template("Tell me one fact about {topic}.")
parser = StrOutputParser()

chain = template | llm | parser

result = chain.invoke({"topic":"octopuses"})
print(result)
