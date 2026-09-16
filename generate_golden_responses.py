import os
import json
import re
import pandas as pd
import joblib

from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity


GOLDEN_FILE = "AmericanAir_Golden_Set.csv"
CORPUS_FILE = "AmericanAir_Retrieval_Corpus_Labeled.csv"
OUTPUT_FILE = "AmericanAir_Golden_Responses.csv"


golden = pd.read_csv(GOLDEN_FILE)

corpus = pd.read_csv(CORPUS_FILE)

model = joblib.load(
    "intent_classifier_model.pkl"
)

vectorizer = joblib.load(
    "intent_tfidf_vectorizer.pkl"
)

golden["text"] = golden["text"].fillna("").astype(str)

corpus["text"] = corpus["text"].fillna("").astype(str)
corpus["response_text"] = corpus["response_text"].fillna("").astype(str)
corpus["intent"] = corpus["intent"].fillna("").astype(str)


label_map = {
    "AAdvantage / Loyalty Program Issue": "loyalty_program",
    "Baggage Problem": "baggage_problem",
    "Check-in, Boarding & Overbooking Issue": "checkin_boarding_overbooking",
    "Compliment / Positive Feedback": "positive_feedback",
    "Customer Service Channel Complaint": "customer_service_complaint",
    "Flight Delay / Cancellation / Disruption": "flight_disruption",
    "General Negative Sentiment (non-specific)": "general_negative",
    "In-Flight Experience Complaint": "inflight_complaint",
    "Refund / Compensation Request": "refund_compensation",
    "Seat Assignment / Upgrade Request or Complaint": "seat_upgrade"
}


def predict_intent(message):

    vector = vectorizer.transform([message])

    return model.predict(vector)[0]


def clean_historical_response(text):

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "[link removed]",
        text
    )

    text = re.sub(
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[phone number removed]",
        text
    )

    return text


def retrieve(message, intent, k=5):

    query_vector = vectorizer.transform([message])

    intent_mask = corpus["intent"] == intent

    filtered_corpus = corpus[intent_mask]

    if len(filtered_corpus) == 0:

        return []

    filtered_vectors = vectorizer.transform(
        filtered_corpus["text"]
    )

    scores = cosine_similarity(
        query_vector,
        filtered_vectors
    )[0]

    indices = scores.argsort()[::-1][:k]

    results = []

    for idx in indices:

        row = filtered_corpus.iloc[idx]

        results.append({
            "customer_issue": row["text"],
            "historical_response": clean_historical_response(
                row["response_text"]
            ),
            "similarity": float(scores[idx])
        })

    return results


def should_escalate(message, intent):

    text = message.lower()

    unresolved_signals = [
        "already contacted customer service",
        "already contacted support",
        "contacted customer service",
        "contacted support",
        "still haven't received",
        "still have not received",
        "still waiting",
        "no response",
        "nobody helped",
        "no one helped",
        "issue is unresolved",
        "still unresolved"
    ]

    reasons = {

        "refund_compensation":
            "The refund remains unresolved after the customer already contacted customer service.",

        "baggage_problem":
            "The baggage issue remains unresolved after the customer already contacted customer service.",

        "flight_disruption":
            "The flight disruption remains unresolved after the customer already contacted customer service.",

        "checkin_boarding_overbooking":
            "The check-in or boarding issue remains unresolved after the customer already contacted customer service.",

        "seat_upgrade":
            "The seat-related issue remains unresolved after the customer already contacted customer service.",

        "inflight_complaint":
            "The in-flight issue remains unresolved after the customer already contacted customer service.",

        "customer_service_complaint":
            "The customer has already contacted customer service but the issue remains unresolved.",

        "loyalty_program":
            "The loyalty-program issue remains unresolved after the customer already contacted customer service.",

        "general_negative":
            "The customer's complaint remains unresolved after the customer already contacted customer service."
    }

    for signal in unresolved_signals:

        if signal in text:

            return True, reasons.get(
                intent,
                "The customer's issue remains unresolved after already contacting customer service."
            )

    return False, ""


def build_prompt(
    message,
    intent,
    retrieved,
    forced_escalation,
    escalation_reason
):

    examples = ""

    for i, item in enumerate(retrieved, 1):

        examples += f"""
Example {i}

Customer issue:
{item["customer_issue"]}

Historical resolution:
{item["historical_response"]}
"""

    if forced_escalation:

        escalation_instruction = f"""
Escalation has already been determined by the application.

You MUST set:

"escalate": true

You MUST use this exact escalation reason:

"{escalation_reason}"

The application does NOT actually contact, notify,
transfer, or forward cases to human agents.

Do not claim that you performed any of these actions.

Instead, explain that the case needs review by
a human customer-service representative.
"""

    else:

        escalation_instruction = """
The application has not forced escalation.

You MUST set:

"escalate": false

"escalation_reason": ""

Do not claim that a human representative
has been contacted.
"""

    prompt = f"""
You are a careful customer-support assistant for American Airlines.

Customer message:

{message}

Predicted intent:

{intent}

Historical customer-support cases:

{examples}

Generate a helpful response to the current customer.

Rules:

- Use historical cases only as guidance.
- Do not copy a historical response blindly.
- Do not invent facts.
- Do not invent policies.
- Do not invent refund rules.
- Do not invent compensation.
- Do not invent phone numbers.
- Do not invent URLs or links.
- Do not copy phone numbers from historical examples.
- Do not copy URLs from historical examples.
- Do not claim that you checked the customer's booking.
- Do not claim that you checked the customer's account.
- Do not claim that an action has already been completed.
- Do not promise a refund.
- Do not promise compensation.
- Do not mention the historical examples.
- Be concise and professional.
- Respond directly to the customer's issue.
- Ask for missing information only when necessary.

{escalation_instruction}

Return exactly ONE valid JSON object.

Do not use markdown.

Do not write anything before or after the JSON.

Do not return an empty response.

Use exactly this structure:

{{
    "intent": "{intent}",
    "reply": "customer-facing response",
    "escalate": false,
    "escalation_reason": ""
}}
"""

    return prompt


def parse_json(content):

    if not content:

        return None

    content = content.strip()

    if content.startswith("```"):

        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = re.sub(
            r"\s*```$",
            "",
            content
        )

        content = content.strip()

    try:

        return json.loads(content)

    except json.JSONDecodeError:

        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1:

            try:

                return json.loads(
                    content[start:end + 1]
                )

            except json.JSONDecodeError:

                return None

    return None


def generate_response(
    client,
    prompt
):

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[

            {
                "role": "system",
                "content": (
                    "You are a careful customer-support assistant. "
                    "Return exactly one valid JSON object. "
                    "Do not use markdown. "
                    "Do not add explanations outside JSON. "
                    "Never invent facts, policies, phone numbers, "
                    "URLs, refunds, compensation, or completed actions."
                )
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,

        max_tokens=500
    )

    content = response.choices[0].message.content

    return parse_json(content)


def remove_false_escalation_claim(reply):

    if not isinstance(reply, str):

        return reply

    bad_phrases = [

        "i have escalated",
        "i've escalated",
        "i escalated",
        "i will escalate",
        "i'll escalate",

        "i have forwarded",
        "i've forwarded",
        "i forwarded",
        "i will forward",
        "i'll forward",

        "i have transferred",
        "i've transferred",
        "i transferred",
        "i will transfer",
        "i'll transfer",

        "i have connected you",
        "i've connected you",
        "i connected you",
        "i will connect you",
        "i'll connect you",

        "a representative has been notified",
        "someone will contact you",
        "someone from our team will contact you"
    ]

    text = reply.lower()

    for phrase in bad_phrases:

        if phrase in text:

            return (
                "I’m sorry that this issue remains unresolved. "
                "Since you have already contacted customer service, "
                "this case needs review by a human customer-service representative."
            )

    return reply


def main():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        print("ERROR: GROQ_API_KEY is not set.")

        return

    client = Groq(
        api_key=api_key
    )

    results = []

    total = len(golden)

    print(
        f"Generating responses for {total} Golden Set examples..."
    )

    for i, row in golden.iterrows():

        message = row["text"]

        original_label = row["intent"]

        expected_intent = label_map.get(
            original_label,
            original_label
        )

        print(
            f"\n[{i + 1}/{total}]"
        )

        print(
            "Customer:",
            message[:120]
        )

        predicted_intent = predict_intent(
            message
        )

        retrieved = retrieve(
            message,
            predicted_intent,
            k=5
        )

        forced_escalation, escalation_reason = should_escalate(
            message,
            predicted_intent
        )

        prompt = build_prompt(
            message,
            predicted_intent,
            retrieved,
            forced_escalation,
            escalation_reason
        )

        result = generate_response(
            client,
            prompt
        )

        if result is None:

            print(
                "WARNING: LLM returned invalid/empty response."
            )

            generated_reply = ""

        else:

            generated_reply = result.get(
                "reply",
                ""
            )

            generated_reply = remove_false_escalation_claim(
                generated_reply
            )

        results.append({

            "tweet_id":
                row.get("tweet_id", ""),

            "customer_message":
                message,

            "expected_intent":
                expected_intent,

            "predicted_intent":
                predicted_intent,

            "generated_reply":
                generated_reply,

            "escalate":
                forced_escalation,

            "escalation_reason":
                escalation_reason
        })

    output_df = pd.DataFrame(
        results
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n======================================"
    )

    print(
        "GOLDEN RESPONSE GENERATION COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        "Rows:",
        len(output_df)
    )

    print(
        "Saved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()