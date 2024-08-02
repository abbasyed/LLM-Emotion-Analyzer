import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, AdamW

# Paths to your tokenized data files
paths = {
    'X_train': '/Users/abbassyed/PycharmProjects/LLM-Emotion-Analyzer/data/X_train.csv',
    'attention_masks_train': '/Users/abbassyed/PycharmProjects/LLM-Emotion-Analyzer/data/attention_masks_train.csv',
    'y_train': '/Users/abbassyed/PycharmProjects/LLM-Emotion-Analyzer/data/y_train.csv',
    'X_val': '/Users/abbassyed/PycharmProjects/LLM-Emotion-Analyzer/data/X_val.csv',
    'attention_masks_val': '/Users/abbassyed/PycharmProjects/LLM-Emotion-Analyzer/data/attention_masks_val.csv',
    'y_val': '/Users/abbassyed/PycharmProjects/LLM-Emotion-Analyzer/data/y_val.csv',
    'X_test': '/Users/abbassyed/PycharmProjects/LLM-Emotion-Analyzer/data/X_test.csv',
    'attention_masks_test': '/Users/abbassyed/PycharmProjects/LLM-Emotion-Analyzer/data/attention_masks_test.csv',
    'y_test': '/Users/abbassyed/PycharmProjects/LLM-Emotion-Analyzer/data/y_test.csv'
}

# Load the tokenized data
X_train = pd.read_csv(paths['X_train'], header=None)
attention_masks_train = pd.read_csv(paths['attention_masks_train'], header=None)
y_train = pd.read_csv(paths['y_train'], header=None)

X_val = pd.read_csv(paths['X_val'], header=None)
attention_masks_val = pd.read_csv(paths['attention_masks_val'], header=None)
y_val = pd.read_csv(paths['y_val'], header=None)

X_test = pd.read_csv(paths['X_test'], header=None)
attention_masks_test = pd.read_csv(paths['attention_masks_test'], header=None)
y_test = pd.read_csv(paths['y_test'], header=None)

# Debugging: Print sample data
print("Sample input_ids:", X_train.iloc[0].values)
print("Sample attention_mask:", attention_masks_train.iloc[0].values)
print("Sample label:", y_train.iloc[0].values)


# Create Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, encodings, attention_masks, labels):
        self.encodings = encodings
        self.attention_masks = attention_masks
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings.iloc[idx].values),
            'attention_mask': torch.tensor(self.attention_masks.iloc[idx].values),
            'labels': torch.tensor(self.labels.iloc[idx].values)
        }
        return item

    def __len__(self):
        return len(self.labels)


# Create datasets
train_dataset = CustomDataset(X_train, attention_masks_train, y_train)
val_dataset = CustomDataset(X_val, attention_masks_val, y_val)
test_dataset = CustomDataset(X_test, attention_masks_test, y_test)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Debugging: Print first batch
for batch in train_dataloader:
    print("First batch:", batch)
    break

# Load the model from the local directory
model_dir = '/Users/abbassyed/distilbert-base-uncased'
model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=6)

# Training setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-6)  # Adjusted learning rate

# Initialize best validation loss to a high value
best_val_loss = float('inf')

# Training loop
model.train()
for epoch in range(3):  # Training for 3 epochs
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Calculate loss and update model parameters
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation loop with corrected accuracy calculation
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Calculate validation loss
            loss = outputs.loss
            total_eval_loss += loss.item()

            # Calculate validation accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions = (predictions == labels).sum().item()
            total_eval_accuracy += correct_predictions

    # Calculate average validation loss
    avg_val_loss = total_eval_loss / len(val_dataloader)

    # Calculate average validation accuracy as a percentage
    avg_val_accuracy = total_eval_accuracy / len(y_val) * 100

    print(f"Epoch {epoch + 1}")
    print(f"Validation Loss: {avg_val_loss}")
    print(f"Validation Accuracy: {avg_val_accuracy}%")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model_state.bin')
        print("Best model saved!")

    model.train()  # Back to training mode

# Optionally, evaluate on the test dataset
model.eval()
total_test_accuracy = 0

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions = (predictions == labels).sum().item()
        total_test_accuracy += correct_predictions

avg_test_accuracy = total_test_accuracy / len(y_test) * 100
print(f"Test Accuracy: {avg_test_accuracy}%")