import pandas as pd

conversations = pd.read_csv("AmericanAir_conversations.csv")
golden = pd.read_csv("AmericanAir_Golden_Set.csv")

conversations["tweet_id"] = conversations["tweet_id"].astype(str)

golden_ids = set(
    golden["tweet_id"].astype(str)
)

print("Total conversation rows:", len(conversations))
print("Golden Set tweet IDs:", len(golden_ids))


# Find Golden Set tweets in the conversation dataset
golden_rows = conversations[
    conversations["tweet_id"].isin(golden_ids)
].copy()

print("Golden tweets found:", len(golden_rows))


# Get all tweets that are directly connected to Golden Set tweets
golden_conversation_ids = set()

for _, row in golden_rows.iterrows():

    golden_conversation_ids.add(row["tweet_id"])

    if pd.notna(row["in_response_to_tweet_id"]):
        golden_conversation_ids.add(
            str(row["in_response_to_tweet_id"])
        )

    if pd.notna(row["response_tweet_id"]):
        ids = str(row["response_tweet_id"]).split(",")

        for tweet_id in ids:
            golden_conversation_ids.add(tweet_id.strip())


# Expand connections until no new tweets are found
changed = True

while changed:

    changed = False

    related = conversations[
        conversations["tweet_id"].isin(golden_conversation_ids)
    ]

    for _, row in related.iterrows():

        connected_ids = []

        if pd.notna(row["in_response_to_tweet_id"]):
            connected_ids.append(
                str(row["in_response_to_tweet_id"])
            )

        if pd.notna(row["response_tweet_id"]):
            connected_ids.extend(
                str(row["response_tweet_id"]).split(",")
            )

        for tweet_id in connected_ids:

            tweet_id = tweet_id.strip()

            if tweet_id and tweet_id not in golden_conversation_ids:
                golden_conversation_ids.add(tweet_id)
                changed = True


# Split the complete conversation dataset
golden_conversations = conversations[
    conversations["tweet_id"].isin(golden_conversation_ids)
].copy()

training_conversations = conversations[
    ~conversations["tweet_id"].isin(golden_conversation_ids)
].copy()


# Save both datasets
golden_conversations.to_csv(
    "AmericanAir_Golden_Conversations.csv",
    index=False
)

training_conversations.to_csv(
    "AmericanAir_Training_Conversations.csv",
    index=False
)


print("\n========== RESULT ==========")

print(
    "Golden conversation rows:",
    len(golden_conversations)
)

print(
    "Training conversation rows:",
    len(training_conversations)
)

print(
    "Total:",
    len(golden_conversations) + len(training_conversations)
)
print("\nSaved:")
print("AmericanAir_Golden_Conversations.csv")
print("AmericanAir_Training_Conversations.csv")