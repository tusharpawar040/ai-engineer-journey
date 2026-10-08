from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# load the model once at startup, not at every request
clf = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english'
)

class TextIn(BaseModel):
    text:str

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: TextIn):
    result = clf(data.test)[0]
    return {"label": result["label"],"score": round(result["score"],4)}