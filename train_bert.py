import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove the Reuters tag completely to prevent data leakage
    text = re.sub(r'^.*?-?\s*\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?Reuters\s*-\s*', '', text, flags=re.IGNORECASE)
    # Removing common fake news footers
    text = re.sub(r'Featured image.*$', '', text, flags=re.IGNORECASE)
    # Remove raw URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    return text.strip()

# Load data
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')

# Clean the dataset BEFORE training
print("Cleaning texts to prevent data leakage...")
df_fake['text'] = df_fake['text'].apply(clean_text)
df_true['text'] = df_true['text'].apply(clean_text)

df_fake['label'] = 0
df_true['label'] = 1

df = pd.concat([df_fake, df_true]).sample(frac=1).reset_index(drop=True)
df['full_text'] = df['title'] + " " + df['text']

# Split
X_train, X_temp, y_train, y_temp = train_test_split(df['full_text'], df['label'], test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)

def tokenize(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

train_encodings = tokenize(X_train)
val_encodings = tokenize(X_val)
test_encodings = tokenize(X_test)

# Dataset class
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, y_train)
val_dataset = NewsDataset(val_encodings, y_val)
test_dataset = NewsDataset(test_encodings, y_test)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Metrics
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments (GPU optimized)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  
    dataloader_pin_memory=True,
    dataloader_num_workers=0,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
print("\nValidation Results:")
trainer.evaluate()

print("\nTest Results:")
trainer.evaluate(test_dataset)