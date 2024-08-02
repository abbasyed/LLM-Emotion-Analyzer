import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the tokenizer and the trained model state
model_dir = '/Users/abbassyed/distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=6)
model.load_state_dict(torch.load('best_model_state.bin'))
model.eval()

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Emotion label mapping including neutral
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


# Interactive loop to input prompts and get predictions
def main():
    while True:
        prompt = input("Enter a prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        predicted_emotion = predict_emotion(prompt)
        print(f"Predicted Emotion: {predicted_emotion}")
        print("-" * 30)


if __name__ == "__main__":
    main()