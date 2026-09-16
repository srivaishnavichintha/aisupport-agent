import pandas as pd
import json
import time
import os
from groq import Groq

INPUT_FILE = "AmericanAir_Response_Comparison.csv"
OUTPUT_FILE = "AmericanAir_Historical_Response_Judge.csv"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

df = pd.read_csv(INPUT_FILE).fillna("")

if os.path.exists(OUTPUT_FILE):
    old = pd.read_csv(OUTPUT_FILE).fillna("")

    completed = set(
        old[
            old["comparison"].isin(
                ["Better", "Equivalent", "Worse"]
            )
        ]["tweet_id"].astype(str)
    )

    results = old.to_dict("records")
else:
    completed = set()
    results = []

print(f"Total cases: {len(df)}")
print(f"Already completed: {len(completed)}")

for i, row in df.iterrows():

    tweet_id = str(row["tweet_id"])

    if tweet_id in completed:
        continue

    print(f"\n[{i + 1}/{len(df)}]")

    customer = row["customer_message"]
    actual = row["actual_response"]
    generated = row["generated_reply"]

    prompt = f"""
Compare these two customer-support responses.

CUSTOMER MESSAGE:
{customer}

HISTORICAL RESPONSE:
{actual}

GENERATED RESPONSE:
{generated}

Evaluate the generated response against the historical response.

Do not require identical wording.

Judge:
1. Whether both responses address the same customer issue.
2. Whether the generated response provides an appropriate and useful response.
3. Whether the generated response contains unsupported claims.
4. Whether the generated response is better, equivalent, or worse than the historical response.

A generated response can be BETTER than the historical response.
The historical response is not automatically correct.

semantic_similarity:
1 = completely different
2 = somewhat related
3 = same general issue
4 = very similar meaning
5 = essentially equivalent

generated_quality:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

comparison:
Better = generated response is better overall
Equivalent = both are approximately equal
Worse = generated response is worse

Keep the reason short.
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict customer-support response evaluator."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=300,
            reasoning_effort="low",
            include_reasoning=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response_comparison",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "semantic_similarity": {
                                "type": "integer",
                                "enum": [1, 2, 3, 4, 5]
                            },
                            "generated_quality": {
                                "type": "integer",
                                "enum": [1, 2, 3, 4, 5]
                            },
                            "comparison": {
                                "type": "string",
                                "enum": [
                                    "Better",
                                    "Equivalent",
                                    "Worse"
                                ]
                            },
                            "reason": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "semantic_similarity",
                            "generated_quality",
                            "comparison",
                            "reason"
                        ],
                        "additionalProperties": False
                    }
                }
            }
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Empty response from LLM")

        result = json.loads(content)

        result["tweet_id"] = tweet_id
        result["customer_message"] = customer
        result["actual_response"] = actual
        result["generated_response"] = generated

        results.append(result)

        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            f"Similarity: {result['semantic_similarity']}/5 | "
            f"Quality: {result['generated_quality']}/5 | "
            f"{result['comparison']}"
        )

    except Exception as e:

        print(f"Error: {e}")

        failed = {
            "tweet_id": tweet_id,
            "customer_message": customer,
            "actual_response": actual,
            "generated_response": generated,
            "semantic_similarity": "",
            "generated_quality": "",
            "comparison": "Failed",
            "reason": str(e)
        }

        existing_ids = {
            str(x["tweet_id"])
            for x in results
        }

        if tweet_id not in existing_ids:
            results.append(failed)

        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )

    time.sleep(1)

print("\n======================================")
print("HISTORICAL RESPONSE COMPARISON COMPLETE")
print("======================================")

final = pd.DataFrame(results)

if len(final) == 0:
    print("No results available.")
else:

    final["semantic_similarity"] = pd.to_numeric(
        final["semantic_similarity"],
        errors="coerce"
    )

    final["generated_quality"] = pd.to_numeric(
        final["generated_quality"],
        errors="coerce"
    )

    valid = final[
        final["semantic_similarity"].notna()
        & final["generated_quality"].notna()
        & final["comparison"].isin(
            ["Better", "Equivalent", "Worse"]
        )
    ]

    failed_count = len(final) - len(valid)

    print(f"Rows saved: {len(final)}")
    print(f"Successful evaluations: {len(valid)}")
    print(f"Failed evaluations: {failed_count}")

    if len(valid) > 0:

        print(
            "\nAverage semantic similarity:",
            round(valid["semantic_similarity"].mean(), 2),
            "/ 5"
        )

        print(
            "Average generated quality:",
            round(valid["generated_quality"].mean(), 2),
            "/ 5"
        )

        print("\nComparison:")
        print(valid["comparison"].value_counts())

        print("\nComparison percentages:")

        percentages = (
            valid["comparison"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        print(percentages)

    print("\nSaved:")
    print(OUTPUT_FILE)