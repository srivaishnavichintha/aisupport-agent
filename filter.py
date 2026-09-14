import pandas as pd

INPUT_FILE = "./twcs/twcs.csv"
OUTPUT_FILE = "AmericanAir_conversations.csv"

df = pd.read_csv(INPUT_FILE)

df["tweet_id"] = df["tweet_id"].astype(str)

# Find all tweets directly related to AmericanAir
american_ids = set(
    df.loc[
        df["author_id"].astype(str).str.lower().eq("americanair")
        | df["text"].astype(str).str.contains("@americanair", case=False, na=False),
        "tweet_id"
    ]
)

# Build connection between tweets
connections = {}

for _, row in df.iterrows():
    tweet_id = row["tweet_id"]

    if pd.notna(row["in_response_to_tweet_id"]):
        parent = str(row["in_response_to_tweet_id"])
        connections.setdefault(tweet_id, set()).add(parent)
        connections.setdefault(parent, set()).add(tweet_id)

    if pd.notna(row["response_tweet_id"]):
        responses = str(row["response_tweet_id"]).split(",")

        for response in responses:
            response = response.strip()

            if response:
                connections.setdefault(tweet_id, set()).add(response)
                connections.setdefault(response, set()).add(tweet_id)

# Find every tweet connected to an AmericanAir tweet
conversation_ids = set()
queue = list(american_ids)

while queue:
    current = queue.pop()

    if current in conversation_ids:
        continue

    conversation_ids.add(current)

    for connected in connections.get(current, []):
        if connected not in conversation_ids:
            queue.append(connected)

# Keep all tweets belonging to those conversations
result = df[df["tweet_id"].isin(conversation_ids)].copy()

# Sort conversations chronologically
result["created_at"] = pd.to_datetime(
    result["created_at"],
    errors="coerce"
)

result = result.sort_values("created_at")

result.to_csv(OUTPUT_FILE, index=False)

print("AmericanAir starting tweets:", len(american_ids))
print("Full conversation tweets:", len(result))
print("Saved to:", OUTPUT_FILE)