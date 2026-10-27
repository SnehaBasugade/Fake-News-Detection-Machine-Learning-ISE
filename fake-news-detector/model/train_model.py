"""
Fake News Detection - Model Training Script
Uses TF-IDF vectorizer + PassiveAggressiveClassifier
Run this script once to generate the model files: model.pkl and vectorizer.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. Generate synthetic training data
#    (Replace this block with real dataset loading
#     e.g. pd.read_csv("True.csv") / "Fake.csv"
#     from Kaggle: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
# ─────────────────────────────────────────────

def generate_synthetic_data():
    """Generate a small synthetic dataset for demo purposes."""
    real_news = [
        "Scientists discover new vaccine that prevents multiple diseases simultaneously",
        "Government announces new infrastructure plan worth billions",
        "Stock market reaches record high amid economic recovery",
        "New climate agreement signed by major world powers",
        "Research shows exercise improves mental health significantly",
        "NASA successfully launches new Mars rover mission",
        "World Health Organization updates COVID-19 guidelines",
        "Tech giant reports quarterly earnings above expectations",
        "Central bank raises interest rates to combat inflation",
        "New study links air pollution to increased health risks",
        "UN Security Council passes resolution on global security",
        "Scientists confirm discovery of water on distant moon",
        "Supreme Court rules on landmark civil rights case",
        "International trade deal finalized after years of negotiations",
        "New renewable energy project creates thousands of jobs",
        "Medical researchers develop breakthrough cancer treatment",
        "Federal Reserve announces new monetary policy measures",
        "Olympic committee selects host city for upcoming games",
        "Major earthquake causes widespread damage in coastal region",
        "Technology company launches innovative artificial intelligence product",
        "Government releases annual budget with focus on education",
        "New transportation bill passed to improve national infrastructure",
        "Astronomers detect gravitational waves from distant galaxy",
        "Health officials warn about seasonal flu outbreak",
        "International summit addresses global food security crisis",
        "Scientists announce breakthrough in quantum computing research",
        "National elections held peacefully with record voter turnout",
        "Global leaders meet to discuss climate change solutions",
        "New drug approved for treatment of rare genetic disorder",
        "Economic report shows steady growth in employment sector",
        "Archaeological discovery reveals ancient civilization artifacts",
        "United Nations releases annual human development report",
        "Researchers find link between diet and cognitive decline",
        "Major oil spill triggers environmental cleanup operation",
        "Space agency reveals plans for lunar exploration program",
        "New legislation aims to protect consumer data privacy",
        "Scientists warn of accelerating glacier melt worldwide",
        "Central bank releases quarterly economic forecast",
        "International court rules on territorial dispute case",
        "New guidelines issued for mental health treatment protocols",
        "Scientists develop new material with unique properties",
        "Government launches initiative to reduce carbon emissions",
        "Major bank reports record profits for fiscal year",
        "New study examines impact of social media on youth",
        "Diplomatic talks resume between rival nations",
        "Weather service issues warning for severe storm season",
        "University research reveals insights into aging process",
        "New public health campaign targets childhood obesity",
        "Technology firms invest heavily in cybersecurity measures",
        "Environmental groups celebrate new protected area designation",
    ]

    fake_news = [
        "Drinking bleach cures all known diseases instantly",
        "Aliens have secretly taken over the government last night",
        "5G towers are mind control devices planted by billionaires",
        "Eating chocolate every day makes you immortal scientists say",
        "Secret underground city of lizard people discovered in Antarctica",
        "Vaccines contain microchips that track your every movement",
        "The moon landing was filmed in a Hollywood basement studio",
        "Drinking battery acid is the newest health trend doctors say",
        "Local man claims to have walked on water after prayer",
        "Scientists confirm Earth is actually flat all along",
        "Government puts chemicals in water supply to control thoughts",
        "Celebrity reveals secret to living to 200 years old",
        "New world order controls all banks and elections globally",
        "Miracle pill eliminates all diseases in just one hour",
        "Birds are fake government surveillance drones not real animals",
        "Bill Gates admits to controlling weather through secret machines",
        "Scientists discover teleportation device hidden in pyramids",
        "Drinking urine three times daily cures cancer completely",
        "Secret society controls all major world events from bunker",
        "Man grows back lost limb after applying special cream",
        "Government sprays poison from planes to reduce population",
        "Ancient crystal heals all wounds instantly doctors hate it",
        "Scientist fired for discovering cure that cures everything",
        "Robots have secretly replaced all world leaders already",
        "Pizza shop basement houses secret criminal underground network",
        "Sunscreen causes more cancer than it prevents say experts",
        "Time traveler from 2050 warns of imminent catastrophic event",
        "New law allows government to spy inside your home 24/7",
        "Eating only sunlight keeps you healthy and full forever",
        "Fluoride in water causes people to become submissive slaves",
        "Secret tunnel network connects all major world capitals",
        "Billionaire reveals plan to move all humans to Mars next year",
        "Hidden technology suppressed by oil companies gives free energy",
        "Doctors paid by pharma companies to make patients sick",
        "Ancient texts reveal exact date of world ending event",
        "Hollywood celebrities are all clones created in secret labs",
        "New phone model can read your mind and sell your thoughts",
        "Eating magnets aligns your chakras and grants psychic powers",
        "Government builds secret army of mind controlled citizens",
        "Mysterious island appears overnight with no scientific explanation",
        "Man healed from paralysis by drinking special herbal water",
        "World government plans to microchip all humans by next year",
        "Scientists suppress evidence of immortality drug for profits",
        "Secret room in White House contains alien technology evidence",
        "Drinking mercury cures depression according to ancient wisdom",
        "Giant underground tunnels prove ancient giant human civilization",
        "New app allows corporations to watch you through your mirror",
        "Crop circles are messages from future humans warning us",
        "Government replaced tap water with mind altering chemicals",
        "Local woman cured blindness by staring at the sun daily",
    ]

    data = (
        [(text, "REAL") for text in real_news] +
        [(text, "FAKE") for text in fake_news]
    )
    df = pd.DataFrame(data, columns=["text", "label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_kaggle_data(true_path="True.csv", fake_path="Fake.csv"):
    """
    Load real Kaggle dataset if available.
    Dataset: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
    """
    try:
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        true_df["label"] = "REAL"
        fake_df["label"] = "FAKE"
        # Combine title + text for richer features
        true_df["text"] = true_df["title"].fillna("") + " " + true_df["text"].fillna("")
        fake_df["text"] = fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
        df = pd.concat([true_df[["text", "label"]], fake_df[["text", "label"]]], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Loaded Kaggle dataset: {len(df)} samples")
        return df
    except FileNotFoundError:
        print("Kaggle dataset not found. Using synthetic data instead.")
        return None


# ─────────────────────────────────────────────
# 2. Train and save the model
# ─────────────────────────────────────────────

def train_and_save():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Try Kaggle dataset first, fall back to synthetic
    df = load_kaggle_data(
        os.path.join(base_dir, "True.csv"),
        os.path.join(base_dir, "Fake.csv")
    )
    if df is None:
        df = generate_synthetic_data()
        print(f"Using synthetic dataset: {len(df)} samples")

    print(f"\nDataset distribution:\n{df['label'].value_counts()}\n")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )

    # PassiveAggressiveClassifier — fast and accurate for text
    classifier = PassiveAggressiveClassifier(
        max_iter=50,
        random_state=42,
        tol=1e-3,
        C=1.0,
    )

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save pipeline (contains both vectorizer + model)
    model_path = os.path.join(base_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to: {model_path}")

    return pipeline


if __name__ == "__main__":
    train_and_save()
