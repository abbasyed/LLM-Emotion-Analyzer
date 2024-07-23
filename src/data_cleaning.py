import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Print the first few rows and column names to understand the structure
    print(df.head())
    print(df.columns)

    # Assuming the text column is named 'text' and sentiment column is 'label'
    df['text'] = df['text'].apply(clean_text)

    return df


if __name__ == "__main__":
    file_path = '/Users/abbassyed/Documents/LLM-Sentiments-Analyzer/twitter text.csv'
    cleaned_data = load_and_clean_data(file_path)
    cleaned_data.to_csv('../data/cleaned_data.csv', index=False)