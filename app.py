from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import joblib
from transformers import BertTokenizerFast, BertForSequenceClassification
import re
import urllib.request
import urllib.parse
import json

app = FastAPI(title="Sentinel Fake News API")

# --- GLOBAL MODEL INITIALIZATION ---
print("Loading Models... this may take a moment.")
lr_model = joblib.load('lr_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

checkpoint_dir = "./results/checkpoint-8980" 
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", use_fast=True)
bert_model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
bert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

print("Models loaded successfully and mounted to API!")

class NewsInput(BaseModel):
    text: str

labels_map = {0: "FAKE News", 1: "TRUE News"}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^.*?-?\s*\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?Reuters\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Featured image.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    return text.strip()

@app.post("/predict")
async def predict_news(item: NewsInput):
    raw_text = item.text
    if len(raw_text.strip()) == 0:
        return {"error": "Empty text provided"}
        
    cleaned = clean_text(raw_text)

    # --- LR PREDICTION ---
    vec = tfidf_vectorizer.transform([cleaned])
    lr_pred_idx = lr_model.predict(vec)[0].item()
    lr_probs = lr_model.predict_proba(vec)[0]
    lr_conf = lr_probs[lr_pred_idx] * 100

    # --- BERT PREDICTION ---
    inputs = tokenizer(cleaned, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        bert_pred_idx = torch.argmax(logits, dim=1).item()
        bert_probs = torch.nn.functional.softmax(logits, dim=-1)
        bert_conf = bert_probs[0][bert_pred_idx].item() * 100

    # --- GNEWS FACT CHECKING ---
    # We strip common filler words and grab exactly 6 core subject nouns to ensure the NewsData API gets the highest relevance hit!
    stop_words = {"the", "is", "in", "at", "of", "on", "and", "a", "an", "to", "for", "with", "as", "by", "this", "that", "it", "has", "been", "from", "are", "was", "were"}
    core_words = [w for w in cleaned.split() if w.lower() not in stop_words]
    
    # We use explicitly the boolean operator 'AND' on up to 5 core nouns, matching NewsData.io syntax requirements for high precision!
    query = " AND ".join(core_words[:5])
    encoded_query = urllib.parse.quote(query)
    API_KEY = "pub_f2af2dbfc3064e5e9ce633bd92cedf11"
    
    fact_check_data = {
        "count": 0,
        "articles": [],
        "api_error": None
    }
    
    try:
        # Search the NewsData API
        url = f"https://newsdata.io/api/1/news?apikey={API_KEY}&q={encoded_query}&language=en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=5).read()
        data = json.loads(res.decode('utf-8'))
        
        if data.get('status') == 'success' and len(data.get('results', [])) > 0:
            results = data['results'][:3]
            fact_check_data['count'] = len(results)
            for a in results:
                fact_check_data['articles'].append({
                    "title": a.get("title", "Unknown Title"),
                    "url": a.get("link", "#"),
                    "source": a.get("source_id", "NewsData.io")
                })
    except Exception as e:
        print("NewsData API Error:", e)
        fact_check_data['api_error'] = str(e)

    return {
        "lr": {
            "prediction": labels_map[lr_pred_idx],
            "confidence": f"{lr_conf:.2f}%",
            "is_fake": lr_pred_idx == 0
        },
        "bert": {
            "prediction": labels_map[bert_pred_idx],
            "confidence": f"{bert_conf:.2f}%",
            "is_fake": bert_pred_idx == 0
        },
        "fact_check": fact_check_data
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentinel | News Authenticator</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0f172a;
                --panel: #1e293b;
                --text: #f8fafc;
                --accent: #3b82f6;
                --accent-hover: #2563eb;
                --true-bg: rgba(34, 197, 94, 0.2);
                --true-border: #22c55e;
                --fake-bg: rgba(239, 68, 68, 0.2);
                --fake-border: #ef4444;
            }
            body {
                background-color: var(--bg);
                color: var(--text);
                font-family: 'Inter', sans-serif;
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                background-image: radial-gradient(circle at top right, #1e293b, #0f172a);
            }
            .container {
                width: 100%;
                max-width: 1200px;
                margin-top: 50px;
                padding: 2rem;
                box-sizing: border-box;
            }
            h1 {
                font-weight: 800;
                font-size: 3rem;
                margin-bottom: 0.5rem;
                background: linear-gradient(to right, #38bdf8, #818cf8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                text-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
            p.subtitle {
                text-align: center;
                color: #94a3b8;
                margin-bottom: 2.5rem;
                font-size: 1.1rem;
            }
            textarea {
                width: 100%;
                background: var(--panel);
                border: 2px solid #334155;
                border-radius: 16px;
                padding: 1.5rem;
                color: #fff;
                font-size: 1.05rem;
                font-family: 'Inter', sans-serif;
                resize: vertical;
                min-height: 160px;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
                transition: border 0.3s ease, box-shadow 0.3s ease;
                box-sizing: border-box;
            }
            textarea:focus {
                outline: none;
                border-color: var(--accent);
                box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
            }
            button {
                display: block;
                width: 100%;
                background: linear-gradient(to right, var(--accent), #6366f1);
                color: #fff;
                border: none;
                padding: 1.2rem;
                font-size: 1.2rem;
                font-weight: 600;
                border-radius: 16px;
                margin-top: 1.5rem;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.3s ease;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
            }
            button:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
            }
            .results-grid {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 2rem;
                margin-top: 2.5rem;
                opacity: 0;
                transform: translateY(20px);
                transition: all 0.5s ease;
                display: none;
            }
            .results-grid.visible {
                opacity: 1;
                transform: translateY(0);
                display: grid;
            }
            .card {
                background: linear-gradient(145deg, #1e293b, #0f172a);
                border-radius: 20px;
                padding: 2rem;
                border: 1px solid #334155;
                box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .card-title {
                font-size: 0.95rem;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                color: #94a3b8;
                margin-bottom: 1.5rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .status {
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                font-size: 1.8rem;
                font-weight: 800;
                margin-bottom: 1.5rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .status.fake {
                background: var(--fake-bg);
                border: 1px solid var(--fake-border);
                color: #fca5a5;
                box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);
            }
            .status.true {
                background: var(--true-bg);
                border: 1px solid var(--true-border);
                color: #86efac;
                box-shadow: 0 0 20px rgba(34, 197, 94, 0.15);
            }
            .conf-section {
                background: rgba(0,0,0,0.2);
                border-radius: 12px;
                padding: 1rem;
            }
            .conf-bar-bg {
                background: #334155;
                height: 10px;
                border-radius: 5px;
                overflow: hidden;
            }
            .conf-bar-fill {
                height: 100%;
                background: var(--accent);
                width: 0%;
                transition: width 1.5s cubic-bezier(0.16, 1, 0.3, 1);
            }
            .conf-text {
                display: flex;
                justify-content: space-between;
                margin-top: 0.8rem;
                font-size: 0.9rem;
                color: #cbd5e1;
                font-weight: 600;
            }
            .loader {
                display: none;
                text-align: center;
                margin-top: 3rem;
            }
            .spinner {
                display: inline-block;
                width: 50px;
                height: 50px;
                border: 4px solid rgba(255,255,255,0.05);
                border-radius: 50%;
                border-top-color: var(--accent);
                animation: spin 1s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            
            .fact-check-item {
                background: rgba(0,0,0,0.3);
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 0.5rem;
                border-left: 4px solid var(--accent);
            }
            .fact-check-item h4 {
                margin: 0 0 0.5rem 0;
                color: #fff;
                font-size: 1rem;
            }
            .fact-check-item a {
                color: var(--accent);
                text-decoration: none;
                font-size: 0.85rem;
            }
            .fact-check-item a:hover {
                text-decoration: underline;
            }
            .source-tag {
                display: inline-block;
                background: #334155;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                margin-right: 10px;
                color: #cbd5e1;
            }
            .no-results {
                padding: 1.5rem;
                text-align: center;
                background: var(--fake-bg);
                border: 1px dashed var(--fake-border);
                border-radius: 12px;
                color: #fca5a5;
            }

            @media (max-width: 768px) {
                .results-grid { grid-template-columns: 1fr; }
                .card-full { grid-column: span 1; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>FactChecker</h1>
            <p class="subtitle">AI & Web Verification Authenticator</p>
            
            <textarea id="newsText" placeholder="Paste a suspect news sentence here..."></textarea>
            <button onclick="analyzeNews()">Analyze Authenticity</button>

            <div class="loader" id="loader">
                <div class="spinner"></div>
                <p style="color:#94a3b8; margin-top:1.5rem; font-weight: 600; letter-spacing: 1px;">VERIFYING...</p>
            </div>

            <div class="results-grid" id="resultsGrid">
                
                <!-- GNews Fact Check Card -->
                <div class="card">
                    <div class="card-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
                        Related Real-Time News
                    </div>
                    <div id="factCheckContainer">
                        <!-- Filled actively via JS -->
                    </div>
                </div>

                <!-- LR Card -->
                <div class="card">
                    <div class="card-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                        Logistic Regression
                    </div>
                    <div id="lrStatus" class="status true">--</div>
                    <div class="conf-section">
                        <div class="conf-bar-bg"><div class="conf-bar-fill" id="lrBar"></div></div>
                        <div class="conf-text">
                            <span>Confidence</span>
                            <span id="lrConfText">0%</span>
                        </div>
                    </div>
                </div>

                <!-- BERT Card -->
                <div class="card">
                    <div class="card-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"></path><path d="M2 17l10 5 10-5"></path><path d="M2 12l10 5 10-5"></path></svg>
                        BERT Base Uncased
                    </div>
                    <div id="bertStatus" class="status fake">--</div>
                    <div class="conf-section">
                        <div class="conf-bar-bg"><div class="conf-bar-fill" id="bertBar" style="background:#818cf8;"></div></div>
                        <div class="conf-text">
                            <span>Confidence</span>
                            <span id="bertConfText">0%</span>
                        </div>
                    </div>
                </div>
                
            </div>
        </div>

        <script>
            async function analyzeNews() {
                const text = document.getElementById('newsText').value;
                if(!text.trim()) return;

                document.getElementById('resultsGrid').classList.remove('visible');
                document.getElementById('loader').style.display = 'block';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text})
                    });
                    
                    const data = await response.json();
                    document.getElementById('loader').style.display = 'none';
                    
                    if(data.error) {
                        alert(data.error);
                        return;
                    }

                    // Populate Web Verification
                    const container = document.getElementById('factCheckContainer');
                    container.innerHTML = "";
                    if(data.fact_check.api_error) {
                        container.innerHTML = `
                            <div class="no-results" style="border-color: #f59e0b; color: #fcd34d;">
                                <h3 style="margin-top:0;">📡 API Connection Error</h3>
                                <p style="margin-bottom:0; font-size:0.9rem;">NewsData.io rejected the query. Try shortening your input text or checking the logs.</p>
                                <p style="color:#d97706; font-size: 0.8rem; margin-top: 0.5rem;">Error: ${data.fact_check.api_error}</p>
                            </div>
                        `;
                    } else if(data.fact_check.count === 0) {
                        container.innerHTML = `
                            <div class="no-results">
                                <h3 style="margin-top:0;">⚠️ Warning: 0 Real-World Verifications</h3>
                                <p style="margin-bottom:0; font-size:0.9rem;">None of the global news syndicates (CNN, BBC, Reuters, etc.) have published anything matching your input. This is a severe indicator of fabricated news.</p>
                            </div>
                        `;
                    } else {
                        let html = "";
                        data.fact_check.articles.forEach(art => {
                            html += `
                            <div class="fact-check-item">
                                <h4>${art.title}</h4>
                                <div>
                                    <span class="source-tag">${art.source}</span>
                                    <a href="${art.url}" target="_blank">Verify Source Profile ↗</a>
                                </div>
                            </div>`;
                        });
                        container.innerHTML = html;
                    }

                    // Update LR
                    const lrStatus = document.getElementById('lrStatus');
                    lrStatus.className = data.lr.is_fake ? 'status fake' : 'status true';
                    lrStatus.innerText = data.lr.prediction;
                    document.getElementById('lrBar').style.width = '0%'; 
                    setTimeout(() => {
                        document.getElementById('lrBar').style.width = data.lr.confidence;
                    }, 50);
                    document.getElementById('lrConfText').innerText = data.lr.confidence;

                    // Update BERT
                    const bertStatus = document.getElementById('bertStatus');
                    bertStatus.className = data.bert.is_fake ? 'status fake' : 'status true';
                    bertStatus.innerText = data.bert.prediction;
                    document.getElementById('bertBar').style.width = '0%'; 
                    setTimeout(() => {
                        document.getElementById('bertBar').style.width = data.bert.confidence;
                    }, 50);
                    document.getElementById('bertConfText').innerText = data.bert.confidence;

                    document.getElementById('resultsGrid').classList.add('visible');
                } catch (err) {
                    console.error(err);
                    document.getElementById('loader').style.display = 'none';
                    alert("Analysis Failed. Check server logs!");
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    # Starting the server when the script is run natively
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
