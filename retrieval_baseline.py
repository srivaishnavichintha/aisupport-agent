
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_FILE = "AmericanAir_Retrieval_Corpus_Labeled.csv"
GOLDEN_FILE = "AmericanAir_Golden_Set.csv"

corpus = pd.read_csv(CORPUS_FILE)
golden = pd.read_csv(GOLDEN_FILE)

corpus["text"] = corpus["text"].fillna("").astype(str)
golden["text"] = golden["text"].fillna("").astype(str)

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

golden["intent"] = golden["intent"].map(label_map)

print("Building retrieval index...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

corpus_vectors = vectorizer.fit_transform(corpus["text"])
golden_vectors = vectorizer.transform(golden["text"])

print("Corpus:", len(corpus))
print("TF-IDF shape:", corpus_vectors.shape)

similarities = cosine_similarity(
    golden_vectors,
    corpus_vectors
)

ks = [1, 3, 5, 10]

results = []

for i in range(len(golden)):

    scores = similarities[i]

    top_indices = np.argsort(scores)[::-1][:10]

    golden_intent = golden.iloc[i]["intent"]

    for k in ks:

        top_k = top_indices[:k]

        retrieved_intents = corpus.iloc[top_k]["intent"].tolist()

        relevant = sum(
            1
            for intent in retrieved_intents
            if intent == golden_intent
        )

        recall = 1 if relevant > 0 else 0
        precision = relevant / k

        results.append({
            "golden_index": i,
            "intent": golden_intent,
            "k": k,
            "recall": recall,
            "precision": precision
        })

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby("k")[["recall", "precision"]]
    .mean()
)

print("\n========== RETRIEVAL RESULTS ==========")
print(summary)

summary.to_csv("retrieval_metrics.csv")

examples = []

for i in range(min(20, len(golden))):

    top_indices = np.argsort(similarities[i])[::-1][:5]

    for rank, idx in enumerate(top_indices, 1):

        examples.append({
            "golden_text": golden.iloc[i]["text"],
            "golden_intent": golden.iloc[i]["intent"],
            "rank": rank,
            "retrieved_text": corpus.iloc[idx]["text"],
            "retrieved_intent": corpus.iloc[idx]["intent"],
            "similarity": round(
                float(similarities[i][idx]),
                4
            ),
            "response": corpus.iloc[idx]["response_text"]
        })

pd.DataFrame(examples).to_csv(
    "retrieval_examples.csv",
    index=False
)

print("\nSaved:")
print("retrieval_metrics.csv")
print("retrieval_examples.csv")
