
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

TRAIN_FILE = "AmericanAir_labeled_training.csv"
GOLDEN_FILE = "AmericanAir_Golden_Set.csv"

train_df = pd.read_csv(TRAIN_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)

train_df["text"] = train_df["text"].fillna("").astype(str)
golden_df["text"] = golden_df["text"].fillna("").astype(str)

label_map = {
    "Flight Delay / Cancellation / Disruption": "flight_disruption",
    "Baggage Problem": "baggage_problem",
    "Compliment / Positive Feedback": "positive_feedback",
    "Check-in, Boarding & Overbooking Issue": "checkin_boarding_overbooking",
    "In-Flight Experience Complaint": "inflight_complaint",
    "Seat Assignment / Upgrade Request or Complaint": "seat_upgrade",
    "Customer Service Channel Complaint": "customer_service_complaint",
    "Refund / Compensation Request": "refund_compensation",
    "General Negative Sentiment (non-specific)": "general_negative",
    "AAdvantage / Loyalty Program Issue": "loyalty_program"
}

golden_df["intent"] = golden_df["intent"].map(label_map)

print("Training examples:", len(train_df))
print("Golden examples:", len(golden_df))

print("\nTraining distribution:")
print(train_df["intent"].value_counts())

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_df["text"])
X_golden = vectorizer.transform(golden_df["text"])

print("\nTF-IDF shape:", X_train.shape)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, train_df["intent"])

joblib.dump(
    model,
    "intent_classifier_model.pkl"
)

joblib.dump(
    vectorizer,
    "intent_tfidf_vectorizer.pkl"
)

print("\nSaved:")
print("intent_classifier_model.pkl")
print("intent_tfidf_vectorizer.pkl")

predictions = model.predict(X_golden)

accuracy = accuracy_score(
    golden_df["intent"],
    predictions
)

print("\n========== RESULTS ==========")
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        golden_df["intent"],
        predictions,
        digits=4
    )
)

cm = confusion_matrix(
    golden_df["intent"],
    predictions,
    labels=model.classes_
)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

cm_df.to_csv("intent_confusion_matrix.csv")

print("\nSaved: intent_confusion_matrix.csv")

majority_class = train_df["intent"].value_counts().idxmax()

majority_predictions = [
    majority_class
] * len(golden_df)

majority_accuracy = accuracy_score(
    golden_df["intent"],
    majority_predictions
)

print("\n========== MAJORITY BASELINE ==========")
print("Majority class:", majority_class)
print(
    "Majority Accuracy:",
    round(majority_accuracy, 4)
)

print("\nDone.")

