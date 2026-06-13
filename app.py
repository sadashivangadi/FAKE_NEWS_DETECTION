import sqlite3
import re
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)
DB = "database.db"

# ---------------- DB ----------------
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT,
        password TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- LOAD TRAINED MODEL ----------------
model      = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- LOAD NEWS DATASET FOR SEARCH ----------------
_news_df = None

def get_news_df():
    global _news_df
    if _news_df is None:
        try:
            df = pd.read_csv("news.csv")
            df.dropna(subset=["text", "label"], inplace=True)
            df = df[df["label"].isin(["real", "fake"])].reset_index(drop=True)
            _news_df = df
        except Exception:
            _news_df = pd.DataFrame(columns=["text", "label"])
    return _news_df

# ---------------- PAGES ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/index")
def index():
    return render_template("index.html")

# ---------------- SIGNUP ----------------
@app.route("/signup_api", methods=["POST"])
def signup_api():
    data = request.json
    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("INSERT INTO users(username,email,password) VALUES(?,?,?)",
                    (data["username"], data["email"], data["password"]))
        conn.commit()
        return jsonify({"msg": "success"})
    except Exception:
        return jsonify({"msg": "exists"})

# ---------------- LOGIN ----------------
@app.route("/login_api", methods=["POST"])
def login_api():
    data = request.json
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?",
                (data["username"], data["password"]))
    user = cur.fetchone()
    if user:
        return jsonify({"msg": "success"})
    return jsonify({"msg": "fail"})

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("news", "")
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    score = model.decision_function(vec)[0]
    confidence = round(float(np.abs(score) / (1 + np.abs(score)) * 100), 1)
    return jsonify({"result": result, "confidence": confidence})

# ---------------- SEARCH / AUTOCOMPLETE ----------------
@app.route("/search")
def search():
    query        = request.args.get("q", "").strip().lower()
    filter_label = request.args.get("filter", "all")   # all | real | fake
    limit        = int(request.args.get("limit", 8))

    df = get_news_df()

    if df.empty or not query:
        return jsonify({"results": []})

    if filter_label in ("real", "fake"):
        df = df[df["label"] == filter_label]

    mask    = df["text"].str.lower().str.contains(re.escape(query), na=False)
    matched = df[mask].head(limit)

    results = [
        {"text": row["text"], "label": row["label"]}
        for _, row in matched.iterrows()
    ]
    return jsonify({"results": results})

# ---------------- TRENDING ----------------
@app.route("/trending")
def trending():
    df = get_news_df()
    real_samples = df[df["label"] == "real"]["text"].head(5).tolist()
    fake_samples = df[df["label"] == "fake"]["text"].head(5).tolist()
    return jsonify({"real": real_samples, "fake": fake_samples})

if __name__ == "__main__":
    app.run(debug=True)
