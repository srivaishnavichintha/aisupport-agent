
import pandas as pd
import joblib

corpus = pd.read_csv("AmericanAir_Retrieval_Corpus.csv")

model = joblib.load("intent_classifier_model.pkl")
vectorizer = joblib.load("intent_tfidf_vectorizer.pkl")

corpus["text"] = corpus["text"].fillna("").astype(str)

X = vectorizer.transform(corpus["text"])

corpus["intent"] = model.predict(X)

corpus.to_csv(
    "AmericanAir_Retrieval_Corpus_Labeled.csv",
    index=False
)

print("Rows:", len(corpus))
print("Saved: AmericanAir_Retrieval_Corpus_Labeled.csv")
print("\nIntent distribution:")
print(corpus["intent"].value_counts())
