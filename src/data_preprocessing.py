import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def preprocess_data_for_bert(file_path):
    df = pd.read_csv(file_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(text):
        return tokenizer(text, padding='max_length', truncation=True, max_length=128)

    df['text'] = df['text'].astype(str)  # Ensure text is in string format

    tokenized_texts = df['text'].apply(lambda x: tokenize_function(x))
    input_ids = [x['input_ids'] for x in tokenized_texts]
    attention_masks = [x['attention_mask'] for x in tokenized_texts]

    labels = df['label'].values

    # Ensure consistency in lengths
    assert len(input_ids) == len(attention_masks) == len(labels), "Inconsistent lengths of input ids, attention masks, and labels"

    # Handle NaN values
    df = df.dropna(subset=['label'])
    labels = df['label'].values
    input_ids = [input_ids[i] for i in df.index]
    attention_masks = [attention_masks[i] for i in df.index]

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(input_ids, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    attention_masks_train, attention_masks_temp = train_test_split(attention_masks, test_size=0.3, random_state=42)
    attention_masks_val, attention_masks_test = train_test_split(attention_masks_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, attention_masks_train, attention_masks_val, attention_masks_test

if __name__ == "__main__":
    file_path = '../data/cleaned_data.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, attention_masks_train, attention_masks_val, attention_masks_test = preprocess_data_for_bert(file_path)

    # Save the preprocessed data
    pd.DataFrame(X_train).to_csv('../data/X_train.csv', index=False)
    pd.DataFrame(X_val).to_csv('../data/X_val.csv', index=False)
    pd.DataFrame(X_test).to_csv('../data/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('../data/y_train.csv', index=False)
    pd.DataFrame(y_val).to_csv('../data/y_val.csv', index=False)
    pd.DataFrame(y_test).to_csv('../data/y_test.csv', index=False)
    pd.DataFrame(attention_masks_train).to_csv('../data/attention_masks_train.csv', index=False)
    pd.DataFrame(attention_masks_val).to_csv('../data/attention_masks_val.csv', index=False)
    pd.DataFrame(attention_masks_test).to_csv('../data/attention_masks_test.csv', index=False)