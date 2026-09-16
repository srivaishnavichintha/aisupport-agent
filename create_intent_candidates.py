import pandas as pd
import re

input_file = "AmericanAir_customer_clean.csv"

df = pd.read_csv(input_file)

intents = {
    "flight_disruption": [
        "delay", "delayed", "cancel", "cancelled", "canceled",
        "flight", "missed connection", "tarmac", "diverted",
        "rescheduled", "disruption"
    ],

    "baggage_problem": [
        "bag", "baggage", "luggage", "lost bag", "missing bag",
        "damaged bag", "checked bag", "suitcase"
    ],

    "positive_feedback": [
        "thank", "thanks", "great", "awesome", "amazing",
        "love", "excellent", "wonderful", "best", "appreciate"
    ],

    "checkin_boarding_overbooking": [
        "check in", "check-in", "boarding", "board", "boarding pass",
        "gate", "overbooked", "denied boarding", "standby"
    ],

    "inflight_complaint": [
        "flight attendant", "attendant", "crew", "food", "drink",
        "wifi", "entertainment", "on board", "onboard", "in flight"
    ],

    "seat_upgrade": [
        "seat", "upgrade", "upgraded", "first class", "business class",
        "economy", "seat assignment", "seat selection"
    ],

    "customer_service_complaint": [
        "customer service", "agent", "representative", "call center",
        "phone agent", "rude", "unhelpful", "help desk", "support"
    ],

    "refund_compensation": [
        "refund", "refunded", "reimbursement", "compensation",
        "voucher", "credit", "money back", "charged"
    ],

    "general_negative": [
        "worst", "terrible", "horrible", "awful", "disappointed",
        "bad experience", "never again", "hate"
    ],

    "loyalty_program": [
        "aadvantage", "miles", "mileage", "points", "frequent flyer",
        "frequent-flyer", "elite status", "rewards"
    ]
}

for intent, keywords in intents.items():

    pattern = "|".join(
        re.escape(keyword)
        for keyword in keywords
    )

    matches = df[
        df["text"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ].copy()

    matches = matches.drop_duplicates(subset=["tweet_id"])

    sample_size = min(300, len(matches))

    if sample_size > 0:
        matches = matches.sample(
            n=sample_size,
            random_state=42
        )

    matches["candidate_intent"] = intent

    output_file = f"candidate_{intent}.csv"

    matches.to_csv(output_file, index=False)

    print(
        f"{intent}: {len(matches)} rows -> {output_file}"
    )

print("\nDone.")