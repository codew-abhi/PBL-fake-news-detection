import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^.*?-?\s*\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?Reuters\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Featured image.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    return text.strip()

if __name__ == "__main__":
    print("Loading data...")
    df_fake = pd.read_csv('Fake.csv')
    df_true = pd.read_csv('True.csv')

    print("Cleaning texts...")
    df_fake['text'] = df_fake['text'].apply(clean_text)
    df_true['text'] = df_true['text'].apply(clean_text)

    df_fake['label'] = 0
    df_true['label'] = 1

    df = pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
    df['full_text'] = df['title'] + " " + df['text']

    print("Splitting dataset...")
    # Using the exact same seed and test_split sizes as train_bert.py to keep the test set identical!
    X_train, X_temp, y_train, y_temp = train_test_split(df['full_text'], df['label'], test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("Vectorizing text using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)

    print("Evaluating Test set...")
    y_pred = lr_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "True"]))

    print("Saving models...")
    joblib.dump(lr_model, 'lr_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "True"])
    
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Logistic Regression Confusion Matrix (Acc: {acc:.4f})")
    plt.savefig("lr_confusion_matrix.png")
    print("Saved -> lr_confusion_matrix.png")
