"""
train_model.py
Run this once to train and save the fake news detection model.
Uses PassiveAggressiveClassifier - industry standard for fake news detection.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

# ── Real-world style training data ──────────────────────────────────────────
real_news = [
    "The Federal Reserve raised interest rates by 25 basis points to combat inflation",
    "NASA's James Webb Space Telescope captures stunning images of distant galaxies",
    "World Health Organization declares end to COVID-19 global health emergency",
    "Scientists publish new research on climate change impact on Arctic ice sheets",
    "US Supreme Court issues landmark ruling on voting rights",
    "Tech giants report quarterly earnings, stock markets respond positively",
    "United Nations Security Council holds emergency session on Middle East conflict",
    "New study links air pollution to increased risk of cardiovascular disease",
    "Global leaders gather at G20 summit to discuss economic cooperation",
    "European Union passes new data privacy regulations affecting tech companies",
    "Researchers develop new battery technology that charges in minutes",
    "Government announces infrastructure spending plan worth billions",
    "Stock market reaches record high as inflation data shows improvement",
    "Scientists confirm discovery of new species in Amazon rainforest",
    "International Space Station crew completes successful spacewalk",
    "Central bank keeps interest rates unchanged amid economic uncertainty",
    "New cancer treatment shows promising results in clinical trials",
    "Parliament passes new legislation on renewable energy targets",
    "Major earthquake strikes region, rescue teams deployed",
    "Health officials warn about rising cases of respiratory illness",
    "Technology company announces major layoffs amid restructuring",
    "Olympic committee announces host city for upcoming games",
    "New archaeological discovery sheds light on ancient civilization",
    "Climate scientists warn of record-breaking temperatures this year",
    "Government releases official unemployment figures showing improvement",
    "Pharmaceutical company receives approval for new vaccine",
    "International trade deal signed between major economies",
    "Astronomers detect unusual radio signals from distant star system",
    "New education policy aims to improve literacy rates nationwide",
    "Water conservation measures implemented amid prolonged drought",
    "Medical researchers publish findings on diabetes treatment",
    "City council approves new public transportation expansion plan",
    "Scientists develop biodegradable plastic alternative from seaweed",
    "Election results certified after thorough vote counting process",
    "Central government releases annual budget with increased healthcare spending",
    "Scientists warn that deforestation is accelerating at an alarming rate",
    "New study shows benefits of Mediterranean diet for heart health",
    "International criminal court issues arrest warrant for war crimes",
    "Tech startup raises funding to develop affordable clean energy solutions",
    "Government launches investigation into corporate price gouging",
]

fake_news = [
    "Bill Gates microchips hidden inside COVID vaccines control your mind",
    "Aliens have landed in Washington DC and are meeting with government officials secretly",
    "Drinking bleach cures cancer doctors dont want you to know",
    "The moon landing was staged in a Hollywood studio by NASA and the CIA",
    "5G towers are spreading coronavirus and causing birds to fall from the sky",
    "Government is adding fluoride to water to make people dumb and obedient",
    "Celebrities are being replaced by clones controlled by the deep state",
    "Secret society of elites controls all world governments and elections",
    "Facebook is secretly recording all your conversations to sell to advertisers secretly",
    "Eating bananas with milk will cause instant death according to hidden medical report",
    "The earth is flat and NASA has been lying to us for decades",
    "Miracle cure discovered that reverses aging doctors are hiding from public",
    "World leaders planning to reduce world population by 90 percent by 2025",
    "Sunscreen causes skin cancer and is part of a billion dollar conspiracy",
    "Vaccines cause autism confirmed by doctors who were silenced by pharmaceutical companies",
    "Ancient pyramids were built by aliens as landing pads for UFOs",
    "Chemtrails from airplanes are spreading mind control chemicals over cities",
    "Secret underground tunnels connect all major world capitals for elite travel",
    "The government can read your thoughts using satellites and smartphones",
    "Eating chocolate every day cures all diseases doctors wont tell you this",
    "Illuminati confirms new world order will begin next month",
    "Man grows third arm after receiving experimental government injection",
    "Scientists discover that the sun is actually cold and produces no heat",
    "Reptilian humanoids are secretly running major corporations and governments",
    "New law will allow government to enter your home without warning anytime",
    "WhatsApp will start charging users thousand rupees per message next week",
    "Forwarding this message to 20 people will give you free mobile recharge",
    "Government plans to shut down internet permanently next month",
    "Miracle plant found in jungle cures diabetes in 3 days no doctors needed",
    "Police are arresting people for not sharing government posts on social media",
    "Bank accounts will be frozen unless you update KYC through this link immediately",
    "New currency will replace all cash making your savings worthless overnight",
    "Secret footage shows politician admitting to stealing billions in coded message",
    "Actor found dead in suspicious circumstances covered up by mainstream media",
    "Ancient scroll discovered proves all modern religions are fake inventions",
    "Scientist fired for revealing that climate change is a government hoax",
    "Drinking lemon water every morning permanently cures all cancers",
    "Government secretly putting birth control in food supply to reduce population",
    "New phone model emits radiation that causes brain tumors in 30 days",
    "Viral video proves that this celebrity is actually a robot programmed by elites",
]

# Build dataframe
texts = real_news + fake_news
labels = ["real"] * len(real_news) + ["fake"] * len(fake_news)

# Also load news.csv if it has data
try:
    csv_df = pd.read_csv("news.csv")
    csv_df.dropna(subset=["text", "label"], inplace=True)
    csv_df = csv_df[csv_df["label"].isin(["real", "fake"])]
    texts += csv_df["text"].tolist()
    labels += csv_df["label"].tolist()
    print(f"Loaded {len(csv_df)} rows from news.csv")
except Exception as e:
    print(f"Could not load news.csv: {e}")

df = pd.DataFrame({"text": texts, "label": labels})

# Clean text
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean)

print(f"Total samples: {len(df)} | Real: {(df.label=='real').sum()} | Fake: {(df.label=='fake').sum()}")

# ── Train / Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ── Vectorizer ───────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2),
    max_features=10000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── Model: PassiveAggressiveClassifier ───────────────────────────────────────
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save model and vectorizer ─────────────────────────────────────────────────
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel saved as model.pkl and vectorizer.pkl")
