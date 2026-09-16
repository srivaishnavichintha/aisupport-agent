
import pandas as pd

files = [
    "AmericanAir_labeled_training.csv",
    "labeled_intents_batch7_8.csv",
    "labeled_training_batches_9_10.csv"
]

dfs = []

for file in files:
    df = pd.read_csv(file)
    print(file, ":", len(df), "rows")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

print("\nTotal before duplicate removal:", len(combined))
print("Unique tweet IDs:", combined["tweet_id"].nunique())

duplicates = combined[combined.duplicated("tweet_id", keep=False)]

print("Rows involved in duplicate IDs:", len(duplicates))

combined = combined.drop_duplicates(
    subset="tweet_id",
    keep="first"
)

print("\nTotal after duplicate removal:", len(combined))
print("Unique tweet IDs:", combined["tweet_id"].nunique())

valid_intents = {
    "flight_disruption",
    "baggage_problem",
    "positive_feedback",
    "checkin_boarding_overbooking",
    "inflight_complaint",
    "seat_upgrade",
    "customer_service_complaint",
    "refund_compensation",
    "general_negative",
    "loyalty_program"
}

invalid = combined[~combined["intent"].isin(valid_intents)]

print("Invalid intent rows:", len(invalid))

missing = combined[
    combined["tweet_id"].isna() |
    combined["text"].isna() |
    combined["intent"].isna()
]

print("Missing required values:", len(missing))

print("\nFinal intent distribution:")
print(combined["intent"].value_counts())

combined.to_csv(
    "AmericanAir_labeled_training_v2.csv",
    index=False
)

print("\nSaved: AmericanAir_labeled_training_v2.csv")
