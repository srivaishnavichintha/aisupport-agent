import pandas as pd

df = pd.read_csv("AmericanAir_conversations.csv")

customer = df[df["inbound"] == True].copy()
customer = customer[
    ["tweet_id", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]
]
customer = customer.dropna(subset=["text"])
customer = customer.drop_duplicates(subset=["tweet_id"])
customer.to_csv("AmericanAir_customer_messages.csv", index=False)
print("Customer messages:", len(customer))
print("Saved to: AmericanAir_customer_messages.csv")