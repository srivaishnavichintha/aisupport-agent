import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

input_file = "AmericanAir_Training_Conversations.csv"
output_file = "AmericanAir_Retrieval_Corpus.csv"

df = pd.read_csv(input_file)

df["text"] = df["text"].fillna("").astype(str)
df["inbound"] = df["inbound"].astype(str).str.lower()

df["is_customer"] = df["inbound"].isin(["true", "1"])

df["response_text"] = ""

tweet_map = dict(zip(df["tweet_id"].astype(str), df["text"]))

for i, row in df.iterrows():
    response_id = str(row["response_tweet_id"])

    if response_id in tweet_map:
        df.at[i, "response_text"] = tweet_map[response_id]

customer_df = df[
    (df["is_customer"]) &
    (df["response_text"].str.strip() != "")
].copy()

customer_df = customer_df[
    customer_df["text"].str.len() >= 10
].copy()

customer_df = customer_df.drop_duplicates(
    subset=["tweet_id"]
)

customer_df[
    ["tweet_id", "created_at", "text", "response_text"]
].to_csv(
    output_file,
    index=False
)

print("Training conversations:", len(df))
print("Customer issues with responses:", len(customer_df))
print("Saved:", output_file)