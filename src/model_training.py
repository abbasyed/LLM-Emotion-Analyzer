import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import ParameterSampler
import math

def load_data():
    X_train = pd.read_csv('../data/X_train.csv').values
    X_val = pd.read_csv('../data/X_val.csv').values
    X_test = pd.read_csv('../data/X_test.csv').values
    y_train = pd.read_csv('../data/y_train.csv').values.flatten()
    y_val = pd.read_csv('../data/y_val.csv').values.flatten()
    y_test = pd.read_csv('../data/y_test.csv').values.flatten()
    attention_masks_train = pd.read_csv('../data/attention_masks_train.csv').values
    attention_masks_val = pd.read_csv('../data/attention_masks_val.csv').values
    attention_masks_test = pd.read_csv('../data/attention_masks_test.csv').values

    y_train = pd.Series(y_train).dropna().values
    y_val = pd.Series(y_val).dropna().values
    y_test = pd.Series(y_test).dropna().values

    print(f"Lengths: X_train={len(X_train)}, y_train={len(y_train)}, attention_masks_train={len(attention_masks_train)}")
    print(f"Lengths: X_val={len(X_val)}, y_val={len(y_val)}, attention_masks_val={len(attention_masks_val)}")
    print(f"Lengths: X_test={len(X_test)}, y_test={len(y_test)}, attention_masks_test={len(attention_masks_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, attention_masks_train, attention_masks_val, attention_masks_test

def balance_data(X_train, y_train, attention_masks_train):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    attention_masks_resampled, _ = ros.fit_resample(attention_masks_train, y_train)
    return X_resampled, y_resampled, attention_masks_resampled

def train_incremental(X_train, y_train, attention_masks_train, X_val, y_val, attention_masks_val, params, model, optimizer, scheduler, batch_size, device, epochs):
    train_inputs = torch.tensor(X_train)
    validation_inputs = torch.tensor(X_val)
    train_labels = torch.tensor(y_train).long()
    validation_labels = torch.tensor(y_val).long()
    attention_masks_train = torch.tensor(attention_masks_train)
    attention_masks_val = torch.tensor(attention_masks_val)

    train_data = TensorDataset(train_inputs, attention_masks_train, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, attention_masks_val, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Calculate number of batches per epoch
    num_batches = math.ceil(len(X_train) / batch_size)
    print(f"Number of batches per epoch: {num_batches}")

    best_val_accuracy = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_inputs = batch[0].to(device)
            batch_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)

            model.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(batch_inputs, attention_mask=batch_masks, labels=batch_labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        eval_accuracy = 0
        nb_eval_steps = 0
        preds, true_labels = [], []

        for batch in validation_dataloader:
            batch_inputs = batch[0].to(device)
            batch_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(batch_inputs, attention_mask=batch_masks)

            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

            preds.extend(np.argmax(logits, axis=1))
            true_labels.extend(label_ids)

            eval_accuracy += np.sum(np.argmax(logits, axis=1) == label_ids)
            nb_eval_steps += 1

        avg_val_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation accuracy for epoch {epoch + 1}: {avg_val_accuracy}")

        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    return best_val_accuracy, model

def random_search(X_train, y_train, attention_masks_train, X_val, y_val, attention_masks_val, n_iter=10):
    param_grid = {
        'batch_size': [16, 32],
        'learning_rate': [2e-5, 3e-5, 5e-5],
        'epochs': [3, 4, 5]
    }

    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))
    best_accuracy = 0
    best_params = {}
    best_model = None

    for params in param_list:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/MiniLM-L12-H384-uncased", num_labels=6)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=params['learning_rate'], eps=1e-8)
        total_steps = len(X_train) // params['batch_size'] * params['epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        accuracy, model = train_incremental(X_train, y_train, attention_masks_train, X_val, y_val, attention_masks_val, params, model, optimizer, scheduler, params['batch_size'], device, params['epochs'])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model

    return best_accuracy, best_params, best_model

def evaluate_model(model, X_val, y_val, attention_masks_val):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    validation_inputs = torch.tensor(X_val).to(device)
    validation_labels = torch.tensor(y_val).long().to(device)
    attention_masks_val = torch.tensor(attention_masks_val).to(device)

    validation_data = TensorDataset(validation_inputs, attention_masks_val, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=32)

    model.eval()
    preds, true_labels = [], []

    for batch in validation_dataloader:
        batch_inputs = batch[0].to(device)
        batch_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(batch_inputs, attention_mask=batch_masks)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()

        preds.extend(np.argmax(logits, axis=1))
        true_labels.extend(label_ids)

    print("Classification Report:\n", classification_report(true_labels, preds, target_names=['joy', 'love', 'anger', 'fear', 'surprise', 'other']))

    cm = confusion_matrix(true_labels, preds)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, attention_masks_train, attention_masks_val, attention_masks_test = load_data()

    best_accuracy, best_params, best_model = random_search(X_train, y_train, attention_masks_train, X_val, y_val, attention_masks_val, n_iter=5)
    print(f"Best accuracy: {best_accuracy}")
    print(f"Best parameters: {best_params}")

    evaluate_model(best_model, X_val, y_val, attention_masks_val)

    torch.save(best_model.state_dict(), "best_model.pt")