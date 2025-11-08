# LLM Emotion Analyzer

A deep learning-based emotion analyzer that uses DistilBERT to classify text into six emotion categories: Neutral, Joy, Love, Anger, Fear, and Surprise.

## Live Demo

**Try it now:** [https://llm-emotion-analyzer-3x2jsogrgsk6yedqjqwymt.streamlit.app/](https://llm-emotion-analyzer-3x2jsogrgsk6yedqjqwymt.streamlit.app/)

The app is deployed on Streamlit Community Cloud and is free to use. Simply enter any text and get instant emotion predictions!

## Overview

This project implements a fine-tuned DistilBERT model for emotion classification from text. It includes a complete pipeline from data preprocessing to model training and an interactive web interface for real-time emotion prediction.

## Features

- Text preprocessing and cleaning (removes HTML tags, special characters, stopwords)
- DistilBERT-based emotion classification model
- Support for 6 emotion categories: Neutral, Joy, Love, Anger, Fear, Surprise
- Interactive Streamlit web interface for testing
- Model training with validation and test evaluation
- Pre-trained model checkpoint saving

## Project Structure

```
LLM-Emotion-Analyzer/
├── data/                           # Data files (CSV format)
│   ├── cleaned_data.csv           # Cleaned text data
│   ├── X_train.csv                # Training input IDs
│   ├── X_val.csv                  # Validation input IDs
│   ├── X_test.csv                 # Test input IDs
│   ├── y_train.csv                # Training labels
│   ├── y_val.csv                  # Validation labels
│   ├── y_test.csv                 # Test labels
│   ├── attention_masks_train.csv  # Training attention masks
│   ├── attention_masks_val.csv    # Validation attention masks
│   └── attention_masks_test.csv   # Test attention masks
├── src/
│   ├── data_cleaning.py           # Text cleaning and preprocessing
│   ├── data_preprocessing.py      # Tokenization and data splitting
│   ├── model_training.py          # Model training script
│   ├── test_model.py              # Command-line testing script
│   ├── evaluate_model.py          # Streamlit web app
│   └── best_model_state.bin       # Saved model checkpoint
└── README.md
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- pandas
- scikit-learn
- nltk
- streamlit
- Pre-downloaded DistilBERT model at `/Users/abbassyed/distilbert-base-uncased`

### Install Dependencies

```bash
pip install torch transformers pandas scikit-learn nltk streamlit
```

## Usage

### 1. Data Cleaning

Clean raw text data by removing HTML tags, special characters, and stopwords:

```bash
cd src
python data_cleaning.py
```

This reads the raw data and outputs `cleaned_data.csv` to the `data/` directory.

### 2. Data Preprocessing

Tokenize the cleaned data using BERT tokenizer and split into train/validation/test sets:

```bash
python data_preprocessing.py
```

This generates tokenized input IDs, attention masks, and labels for all splits.

### 3. Model Training

Train the DistilBERT model on the preprocessed data:

```bash
python model_training.py
```

The script will:
- Train for 3 epochs
- Display validation loss and accuracy after each epoch
- Save the best model as `best_model_state.bin`
- Evaluate on the test set

### 4. Testing with Sample Prompts

Test the trained model with predefined sample prompts:

```bash
python test_model.py
```

### 5. Interactive Web App

Launch the Streamlit web interface for real-time emotion analysis:

```bash
streamlit run evaluate_model.py
```

Access the app at `http://localhost:8501` and enter any text to get emotion predictions.

## Model Details

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Architecture**: DistilBertForSequenceClassification
- **Number of Labels**: 6 (Neutral, Joy, Love, Anger, Fear, Surprise)
- **Max Sequence Length**: 512 tokens
- **Optimizer**: AdamW with learning rate 1e-6
- **Batch Size**: 8
- **Training Epochs**: 3

### Emotion Label Mapping

```python
{
    0: "Neutral",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}
```

## Example Usage

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model
tokenizer = DistilBertTokenizer.from_pretrained('/Users/abbassyed/distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('/Users/abbassyed/distilbert-base-uncased', num_labels=6)
model.load_state_dict(torch.load('best_model_state.bin'))
model.eval()

# Predict emotion
text = "I am so happy today!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
predicted_label = torch.argmax(outputs.logits, dim=-1).item()
```

## Performance

The model achieves validation accuracy metrics after training, which are displayed during the training process. Test accuracy is evaluated on the held-out test set.

## Notes

- Ensure the DistilBERT model is downloaded locally at the specified path
- The model uses GPU if available, otherwise falls back to CPU
- For better Streamlit performance, consider installing the Watchdog module

## Deployment

### Deploy to Streamlit Community Cloud

This app is configured for easy deployment to Streamlit Community Cloud (free hosting).

#### Prerequisites

1. Create a GitHub account if you don't have one
2. Push this repository to GitHub
3. Sign up for [Streamlit Community Cloud](https://streamlit.io/cloud)

#### Steps to Deploy

1. **Push your code to GitHub:**

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit (Git LFS will handle the large model file)
git commit -m "Prepare for Streamlit deployment"

# Add your GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/LLM-Emotion-Analyzer.git

# Push to GitHub (main branch)
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/LLM-Emotion-Analyzer`
   - Set the main file path: `src/evaluate_model.py`
   - Click "Deploy"

3. **Wait for deployment:**
   - Streamlit will install dependencies from `requirements.txt`
   - Git LFS will download the model file (255MB)
   - The app will be live at: `https://YOUR_USERNAME-llm-emotion-analyzer.streamlit.app`

#### Important Notes for Deployment

- The model file is tracked with Git LFS (configured in `.gitattributes`)
- Streamlit Community Cloud has a 1GB memory limit - the app fits within this
- First deployment may take 5-10 minutes due to model download
- The app runs on CPU (GPU not available in free tier)

#### Troubleshooting

If deployment fails:
- Check that Git LFS is properly installed: `git lfs install`
- Verify the model file is tracked: `git lfs ls-files`
- Ensure all dependencies are in `requirements.txt`
- Check Streamlit Cloud logs for specific error messages

### Alternative Deployment Options

#### Hugging Face Spaces

You can also deploy to Hugging Face Spaces:
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload your files or connect via Git
4. The app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/emotion-analyzer`

#### Docker Deployment

For deploying to AWS/GCP/Azure:
1. Create a `Dockerfile` in the project root
2. Build and push to a container registry
3. Deploy to your cloud provider's container service

## Future Improvements

- Add support for more emotion categories
- Implement confidence scores for predictions
- Add batch prediction capabilities
- Create REST API endpoint
- Add model explainability features (attention visualization)
- Reduce model size with quantization for faster deployment

## License

This project is available for educational and research purposes.
