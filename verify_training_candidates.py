import pandas as pd

file = "AmericanAir_labeled_training.csv"

df = pd.read_csv(file)

final_labels = {
    162328: "checkin_boarding_overbooking",
    178539: "customer_service_complaint",
    508981: "inflight_complaint",
    680100: "seat_upgrade",
    1253716: "baggage_problem",
    1484783: "checkin_boarding_overbooking",
    1798751: "baggage_problem",
    1897096: "baggage_problem",
    2487527: "baggage_problem",
    2547603: "refund_compensation",
    2768607: "checkin_boarding_overbooking"
}

for tweet_id, intent in final_labels.items():
    df.loc[df["tweet_id"] == tweet_id, "intent"] = intent
    df.loc[df["tweet_id"] == tweet_id, "confidence"] = "high"
    df.loc[df["tweet_id"] == tweet_id, "reason"] = "Manually resolved conflicting labels using the primary-intent taxonomy."

df.to_csv(file, index=False)

print("Fixed 11 conflicting labels.")
print("Saved:", file)

print("\nFinal intent distribution:")
print(df["intent"].value_counts())