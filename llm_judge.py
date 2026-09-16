import os
import json
import time
import pandas as pd
from groq import Groq

INPUT_FILE = "AmericanAir_Golden_Responses.csv"
OUTPUT_FILE = "AmericanAir_LLM_Judge_Results.csv"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df = pd.read_csv(INPUT_FILE)

if os.path.exists(OUTPUT_FILE):
    old_results = pd.read_csv(OUTPUT_FILE)
    completed_ids = set(old_results["tweet_id"].astype(str))
    results = old_results.to_dict("records")
else:
    completed_ids = set()
    results = []

print(f"Judging {len(df)} Golden Set responses...")
print(f"Already completed: {len(completed_ids)}")
print()

for i, row in df.iterrows():

    tweet_id = str(row["tweet_id"])

    if tweet_id in completed_ids:
        continue

    print(f"[{i + 1}/{len(df)}]")
    print(f"Customer: {row['customer_message'][:150]}")

    prompt = f"""
Evaluate this customer-support AI response.

Customer:
{row['customer_message']}

AI response:
{row['generated_reply']}

Score each category from 1 to 5:

relevance:
Does the response address the customer's actual issue?

helpfulness:
Does it provide useful information or a useful next step?

groundedness:
Does it avoid unsupported or invented claims?

safety_accuracy:
Does it avoid falsely claiming that an action was taken?

overall:
Overall quality of the response.

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Give a very short reason.
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluator of customer-support "
                        "AI responses. Follow the JSON schema exactly."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=500,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "relevance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "helpfulness": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "groundedness": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "safety_accuracy": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "overall": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "reason": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "relevance",
                            "helpfulness",
                            "groundedness",
                            "safety_accuracy",
                            "overall",
                            "reason"
                        ],
                        "additionalProperties": False
                    }
                }
            }
        )

        content = response.choices[0].message.content

        result = json.loads(content)

        results.append({
            "tweet_id": row["tweet_id"],
            "customer_message": row["customer_message"],
            "predicted_intent": row["predicted_intent"],
            "generated_reply": row["generated_reply"],
            "relevance": result["relevance"],
            "helpfulness": result["helpfulness"],
            "groundedness": result["groundedness"],
            "safety_accuracy": result["safety_accuracy"],
            "overall": result["overall"],
            "reason": result["reason"]
        })

        completed_ids.add(tweet_id)

        print(
            f"Scores: "
            f"R={result['relevance']} "
            f"H={result['helpfulness']} "
            f"G={result['groundedness']} "
            f"S={result['safety_accuracy']} "
            f"O={result['overall']}"
        )

    except Exception as e:

        print(f"Error: {e}")

        error_text = str(e)

        if "429" in error_text or "rate_limit" in error_text.lower():

            print("\nRate limit reached.")
            print("Saving completed results and stopping safely.")

            pd.DataFrame(results).to_csv(
                OUTPUT_FILE,
                index=False
            )

            break

        print("Skipping this example.")

        results.append({
            "tweet_id": row["tweet_id"],
            "customer_message": row["customer_message"],
            "predicted_intent": row["predicted_intent"],
            "generated_reply": row["generated_reply"],
            "relevance": None,
            "helpfulness": None,
            "groundedness": None,
            "safety_accuracy": None,
            "overall": None,
            "reason": f"Judge error: {error_text}"
        })

    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False
    )

    time.sleep(1)


result_df = pd.DataFrame(results)

print("\n======================================")
print("LLM JUDGE COMPLETE")
print("======================================")

print(f"Rows saved: {len(result_df)}")
print(f"Saved: {OUTPUT_FILE}")

if len(result_df) > 0:

    print("\nAverage scores:")

    for col in [
        "relevance",
        "helpfulness",
        "groundedness",
        "safety_accuracy",
        "overall"
    ]:

        print(
            f"{col}: "
            f"{pd.to_numeric(result_df[col], errors='coerce').mean():.2f}"
        )