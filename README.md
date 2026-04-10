# 📰 Fake News Detection System

## 📌 Description

The **Fake News Detection System** is a Machine Learning-based web application that classifies news articles as **Real 🟢 or Fake 🔴** using Natural Language Processing (NLP).

With the rise of misinformation across digital platforms, it has become increasingly important to verify the authenticity of news. This project was built to address this problem by analyzing textual content and predicting whether a news article is genuine or misleading.

The system uses **TF-IDF vectorization** and a **Passive Aggressive Classifier** to deliver high accuracy and fast predictions through a simple web interface.

---

## ⚙️ Installation

Follow these steps to run the project locally:

### 1. Clone the Repository
bash
git clone https://github.com/codew-abhi/PBL-fake-news-detection.git
cd PBL-fake-news-detection
2. Create Virtual Environment (Recommended)
python -m venv venv
3. Activate Environment
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
4. Install Dependencies
pip install -r requirements.txt
▶️ Usage

Run the Flask application:

python app.py

Open your browser and go to:

http://127.0.0.1:5000
Example

Input:

"Breaking: Government launches new economic policy..."

Output:

✅ Real News
📊 Model Performance
Accuracy: ~96%
Precision: 0.95
Recall: 0.96
F1 Score: 0.95
Confusion Matrix
                Predicted
              Fake   Real
Actual Fake    580     20
Actual Real     25    575
🤝 Contributing

Contributions are welcome! 🚀

If you'd like to improve this project:

Fork the repository

Create a new branch

git checkout -b feature-name
Make your changes

Commit your changes

git commit -m "Add new feature"

Push to your branch

git push origin feature-name
Open a Pull Request
📜 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this project with proper attribution.

👨‍💻 Author

Abhishek Singh
GitHub: https://github.com/codew-abhi

⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork it
📢 Share it
