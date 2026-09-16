import pandas as pd

llm = pd.read_csv("AmericanAir_LLM_Judge_Results.csv")
generated = pd.read_csv("AmericanAir_Golden_Responses.csv")

df = generated.merge(
    llm[[
        "tweet_id",
        "relevance",
        "helpfulness",
        "groundedness",
        "safety_accuracy",
        "overall"
    ]],
    on="tweet_id",
    how="inner"
)

df = df.sample(n=30, random_state=42)

out = df[[
    "tweet_id",
    "customer_message",
    "generated_reply",
    "relevance",
    "helpfulness",
    "groundedness",
    "safety_accuracy",
    "overall"
]].copy()

out["human_relevance"] = ""
out["human_helpfulness"] = ""
out["human_groundedness"] = ""
out["human_safety_accuracy"] = ""
out["human_overall"] = ""

out.to_csv("AmericanAir_Human_Evaluation.csv", index=False)

print("Human evaluation file created.")
print("Cases:", len(out))
print("Saved: AmericanAir_Human_Evaluation.csv")