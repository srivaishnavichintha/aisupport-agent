import pandas as pd

golden = pd.read_csv("AmericanAir_Golden_Set.csv")
train = pd.read_csv("AmericanAir_training_candidates.csv")

golden_text = set(
    golden["text"]
    .fillna("")
    .str.strip()
    .str.lower()
)

train_text = train["text"].fillna("").str.strip().str.lower()

overlap = train[train_text.isin(golden_text)].copy()

print("Golden Set rows:", len(golden))
print("Training rows:", len(train))
print("Exact text overlap:", len(overlap))

if len(overlap) > 0:
    print("\nExamples:")
    print(overlap[["tweet_id", "text", "candidate_intent"]].head(20))

    train = train[~train_text.isin(golden_text)]

    train.to_csv(
        "AmericanAir_training_candidates.csv",
        index=False
    )

    print("\nRemoved overlapping texts.")
    print("Final training rows:", len(train))
else:
    print("\nNo exact text leakage found.")