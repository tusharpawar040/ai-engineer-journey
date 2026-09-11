from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")
template = PromptTemplate.from_template("Who is the {role} of {country}?")
prompt = template.invoke({"role":"pm","country":"india"})
response = llm.invoke(prompt)
print(response.content)

print("----------what .invoke() returns-----------")
print(prompt)
print(type(prompt))

formatted = template.format(role="president", country="usa")
print("\n---- what .format() returns ----")
print(formatted)
print(type(formatted))