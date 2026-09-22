"""
Day 10: AI powered resume analyser & job matcher
Combines: structured output (Day 4), prompts, and a two-step pipeline.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")


# ----Section 1: Define the shape of parsed resume ----
class ResumeAnalysis(BaseModel):
    name: str 
    skills: List[str]
    experience_years: int 
    summary: str = Field(description="A 1-2 sentence summary of the candidate")

# ---- Section 2: sample resume (stand-in for a real-file) ----
# sample_resume = """
# Tushar Pawar is a software engineer with 3 years of experience in Python, 
# data analysis, and a machine learning. He has worked on building data 
# pipelines using pandas and scikit-learn, and is currently learning deep 
# learning with Pytorch and LangChain for LLM applications. He is comfortable 
# with git, REST APIs, and basic cloud deployment.
# """
sample_resume = """
Tushar Pawar is a software engineer with 3 years of experience in Python, 
data analysis, and machine learning. He has worked on building data 
pipelines using pandas and scikit-learn. He is comfortable with git,
REST APIs, and basic cloud deployment.
"""

# ---- Section 3: parse the resume into structured data ----
resume_parser = llm.with_structured_output(ResumeAnalysis)
parsed_resume = resume_parser.invoke(
    f"Extract structured information from this resume:\n{sample_resume}"
)

print("--- Parsed Resume ---")
print(parsed_resume)
print() 

# Section 4: define the shape of a match result
class JobMatch(BaseModel):
    match_score: int = Field(description="0-100, how well the resume fits the job")
    matching_skills: List[str]
    missing_skills: List[str]
    verdict: str = Field(description="1-2 sentence explanation of the score")

# Section 5: a sample job description
job_description = """
We are hiring a Machine Learning Engineer. Required skills: Python, 
Pytorch, LangChain, experience building LLM-powered applications, and 
familiarity with cloud deployment. 2+ years experience preferred.
"""

# Section 6: compare parsed resume against the job
matcher = llm.with_structured_output(JobMatch)
match_result = matcher.invoke(
    f"""Compare this candidate against the job description and score the match.
    
Candidate: {parsed_resume.model_dump()}

Job description: {job_description}
"""
)

print("--- Job Match Result ---")
print(match_result)
