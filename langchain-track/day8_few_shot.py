from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")

examples = [
    {"word":"happy","antonym":"sad"},
    {"word":"tall","antonym":"short"}
]

example_template = PromptTemplate.from_template("Word: {word}\nAntonym: {antonym}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the antonym of the word.",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
)

prompt = few_shot_prompt.invoke({"input":"girl"})
response = llm.invoke(prompt)
print(prompt)
print("------------------------")
print(response.content)
