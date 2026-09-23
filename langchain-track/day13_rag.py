"""
Day 13: RAG(Retrieval Augmented Generation) pipeline, end to end.
Load -> split -> embed -> store -> retrive -> generate
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")


# Stage 1: Load
# Normally you'd use a real loader (Textloader, PDFLoader). For this first
# run we skip file I/O and just write the source text directly, so you can
# focus on the pipeline itself before adding real files.

raw_text = """
Langchain is framework for building applications powered by language models.
It provides tools for prompt templates, output parsers, and chaining components 
together using LCEL, the pipe operator.

Langraph is a library built on top of LangChain for creating stateful,
multi-step applications using a graph structure. It supports persistence
through checkpointers, allowing conversations to be resumed across sessions.

Groq is an LLM inference provider known for very fast response times. It
offers a free tier and supports open models like Llama and GPT-OSS through
an API compatible with the OpenAI format.

RAG stands for Retrieval-Augmented-Generation. It combines a retrieval step, 
searching a vector database for relevant text, with a generation step, where
an LLM uses that retrieved text to answer a question.
"""

document = Document(page_content=raw_text)



# Stage 2:Split
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents([document])

print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"----- chunk {i} -----")
    print(chunk.page_content)
    print()


# Stage 3 + 4: Embed + Store
# HuggingFaceEmbeddings runs locally, no API key or cost involved
 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks,embeddings)


# Stage 5: Retrieve

question = "What is langgraph used for?"
retriever = vectorstore.as_retriever(search_kwargs={"k":2})
relevant_chunks = retriever.invoke(question)

print("---Retrieved chunks---")
for chunk in relevant_chunks:
    print(chunk.page_content)
    print()


# Stage 6: Generate

context = "\n".join(chunk.page_content for chunk in relevant_chunks)

prompt = f"""Answer the question using only the context below. If the context
doesn't contain the answer, say so.

Context:
{context}

Question:{question}
"""

response = llm.invoke(prompt)
print("---- Final Answer ----")
print(response.content)
