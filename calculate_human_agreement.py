import pandas as pd

df = pd.read_csv("AmericanAir_Human_Evaluation.csv")

criteria = [
    "relevance",
    "helpfulness",
    "groundedness",
    "safety_accuracy",
    "overall"
]

print("======================================")
print("HUMAN vs LLM JUDGE AGREEMENT")
print("======================================")

print("Cases:", len(df))

for c in criteria:
    human = pd.to_numeric(df["human_" + c], errors="coerce")
    llm = pd.to_numeric(df[c], errors="coerce")

    exact = (human == llm).mean() * 100
    within_one = ((human - llm).abs() <= 1).mean() * 100

    print()
    print(c)
    print("Human average:", round(human.mean(), 2))
    print("LLM average:", round(llm.mean(), 2))
    print("Exact agreement:", round(exact, 2), "%")
    print("Agreement within 1 point:", round(within_one, 2), "%")

print()
print("Overall exact agreement:")

human_all = df[["human_" + c for c in criteria]].apply(
    pd.to_numeric, errors="coerce"
)

llm_all = df[criteria].apply(
    pd.to_numeric, errors="coerce"
)

exact_all = (human_all.values == llm_all.values).mean() * 100
within_one_all = (
    abs(human_all.values - llm_all.values) <= 1
).mean() * 100

print("Exact:", round(exact_all, 2), "%")
print("Within 1 point:", round(within_one_all, 2), "%")

print()
print("Done.")