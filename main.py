import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model and tokenizer
model_path = "./intent_classifier"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define FastAPI app
app = FastAPI()

# Define request model
class TextRequest(BaseModel):
    text: str

# Inference function
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# API endpoint
@app.post("/predict")
def predict(request: TextRequest):
    intent = predict_intent(request.text)
    return {"intent": intent}
