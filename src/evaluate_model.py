import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import streamlit as st

# Load the tokenizer and the trained model state
model_dir = '/Users/abbassyed/distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=6)
model.load_state_dict(torch.load('best_model_state.bin'))
model.eval()

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Emotion label mapping
label_map = {0: "Neutral", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}

# Function to predict the emotion for a given prompt
def predict_emotion(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()

    # Map predicted label to emotion
    return label_map.get(predicted_label, "Unknown")

# Streamlit app
st.title("Emotion Analyzer")

# User input
user_input = st.text_input("Enter a sentence to analyze the emotion:")

if user_input:
    # Predict emotion
    predicted_emotion = predict_emotion(user_input)
    st.write(f"Predicted Emotion: {predicted_emotion}")