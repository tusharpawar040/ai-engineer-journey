from transformers import pipeline

clf = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print(clf("I love learning AI engineering"))
print(clf(["This is terrible","Pretty good, could be better","I have seen worse"]))

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
enc = tok("I love learning AI engineering")

print(enc["input_ids"])
print(tok.convert_ids_to_tokens(enc["input_ids"]))