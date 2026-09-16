import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

actual_file = "AmericanAir_Golden_Conversations.csv"
generated_file = "AmericanAir_Golden_Responses.csv"

actual = pd.read_csv(actual_file, dtype=str).fillna("")
generated = pd.read_csv(generated_file, dtype=str).fillna("")

actual["tweet_id"] = actual["tweet_id"].str.strip()
generated["tweet_id"] = generated["tweet_id"].str.strip()

response_map = dict(zip(actual["tweet_id"], actual["text"]))
response_id_map = dict(zip(actual["tweet_id"], actual["response_tweet_id"]))

def get_actual_response(response_ids):
    responses = []

    for rid in str(response_ids).split(";"):
        rid = rid.strip()

        if rid in response_map:
            text = response_map[rid].strip()

            if text:
                responses.append(text)

    return " ".join(responses)

generated["response_tweet_id"] = generated["tweet_id"].map(response_id_map).fillna("")
generated["actual_response"] = generated["response_tweet_id"].apply(
    get_actual_response
)

generated = generated[
    generated["actual_response"].str.strip() != ""
].copy()

actual_text = generated["actual_response"].tolist()
generated_text = generated["generated_reply"].tolist()

vectorizer = TfidfVectorizer(stop_words="english")

all_text = actual_text + generated_text
X = vectorizer.fit_transform(all_text)

actual_vectors = X[:len(actual_text)]
generated_vectors = X[len(actual_text):]

similarities = []

for i in range(len(generated)):
    score = cosine_similarity(
        generated_vectors[i],
        actual_vectors[i]
    )[0][0]

    similarities.append(score)

generated["similarity"] = similarities

def similarity_category(score):
    if score >= 0.70:
        return "Strong Match"
    elif score >= 0.40:
        return "Moderate Match"
    elif score >= 0.20:
        return "Weak Match"
    else:
        return "Poor Match"

generated["match"] = generated["similarity"].apply(similarity_category)

print("\n======================================")
print("ACTUAL vs GENERATED RESPONSE CHECK")
print("======================================")

print(f"Total generated responses: {len(pd.read_csv(generated_file))}")
print(f"Responses with actual historical response: {len(generated)}")

print("\nAverage similarity:",
      round(generated["similarity"].mean(), 4))

print("\nMatch distribution:")
print(generated["match"].value_counts())

strong = (generated["similarity"] >= 0.70).sum()
moderate = (generated["similarity"] >= 0.40).sum()
weak_or_better = (generated["similarity"] >= 0.20).sum()

print("\n======================================")
print("MATCH ACCURACY")
print("======================================")

print(
    f"Strong Match Accuracy (>= 0.70): "
    f"{strong}/{len(generated)} = "
    f"{strong / len(generated) * 100:.2f}%"
)

print(
    f"Moderate-or-Strong Match (>= 0.40): "
    f"{moderate}/{len(generated)} = "
    f"{moderate / len(generated) * 100:.2f}%"
)

print(
    f"Weak-or-better Match (>= 0.20): "
    f"{weak_or_better}/{len(generated)} = "
    f"{weak_or_better / len(generated) * 100:.2f}%"
)

print("\n======================================")
print("BEST MATCHES")
print("======================================")

for _, row in generated.sort_values(
    "similarity", ascending=False
).head(5).iterrows():

    print("\nCustomer:")
    print(row["customer_message"])

    print("\nActual:")
    print(row["actual_response"])

    print("\nGenerated:")
    print(row["generated_reply"])

    print("\nSimilarity:",
          round(row["similarity"], 4))

print("\n======================================")
print("WORST MATCHES")
print("======================================")

for _, row in generated.sort_values(
    "similarity", ascending=True
).head(5).iterrows():

    print("\nCustomer:")
    print(row["customer_message"])

    print("\nActual:")
    print(row["actual_response"])

    print("\nGenerated:")
    print(row["generated_reply"])

    print("\nSimilarity:",
          round(row["similarity"], 4))

generated.to_csv(
    "AmericanAir_Response_Comparison.csv",
    index=False
)

print("\nSaved:")
print("AmericanAir_Response_Comparison.csv")