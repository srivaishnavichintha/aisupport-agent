import os
import json
import re
import pandas as pd
import joblib

from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity


CORPUS_FILE = "AmericanAir_Retrieval_Corpus_Labeled.csv"


corpus = pd.read_csv(CORPUS_FILE)

model = joblib.load(
    "intent_classifier_model.pkl"
)

vectorizer = joblib.load(
    "intent_tfidf_vectorizer.pkl"
)

corpus["text"] = corpus["text"].fillna("").astype(str)
corpus["response_text"] = corpus["response_text"].fillna("").astype(str)
corpus["intent"] = corpus["intent"].fillna("").astype(str)

corpus_vectors = vectorizer.transform(
    corpus["text"]
)


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

Therefore, NEVER claim that you performed any of these actions.

NEVER say:

- "I have escalated your case"
- "I've escalated your case"
- "I escalated your case"
- "I will escalate your case"
- "I'll escalate your case"
- "I have forwarded your case"
- "I've forwarded your case"
- "I will forward your case"
- "I'll forward your case"
- "I transferred your case"
- "I will transfer your case"
- "I connected you with an agent"
- "I will connect you with an agent"
- "A representative has been notified"
- "Someone will contact you"

Instead, explain naturally that the case needs
review by a human customer-service representative.
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

Below are historical customer-support cases and their responses.

Use them only as guidance for how similar issues were handled.

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
- Do not expose internal information.
- Do not mention the historical examples.
- Be concise and professional.
- Respond directly to the customer's issue.
- Ask for missing information only when necessary.
- If specific information is unavailable, say that a human representative needs to review the case.

{escalation_instruction}

Return exactly ONE JSON object.

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


def generate_response(prompt):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        print("ERROR: GROQ_API_KEY is not set.")

        print(
            'Run: $env:GROQ_API_KEY="YOUR_API_KEY_HERE"'
        )

        return None

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[

            {
                "role": "system",

                "content": (
                    "You are a careful customer-support assistant. "
                    "Return exactly one valid JSON object. "
                    "Do not return markdown. "
                    "Do not return explanations outside JSON. "
                    "Never invent phone numbers, URLs, policies, "
                    "refund information, compensation, or completed actions. "
                    "Never claim that you contacted, transferred, "
                    "notified, or escalated a customer case because "
                    "the application does not perform those actions."
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

    if not content:

        print("\nThe model returned an empty response.")

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

        result = json.loads(content)

        return result

    except json.JSONDecodeError:

        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1 and end > start:

            possible_json = content[start:end + 1]

            try:

                result = json.loads(possible_json)

                return result

            except json.JSONDecodeError:

                pass

        print("\nThe model returned invalid JSON:")

        print(repr(content))

        return None


def remove_false_escalation_claim(reply):

    if not isinstance(reply, str):

        return reply

    bad_phrases = [

        "i have escalated",
        "i've escalated",
        "i escalated",
        "i will escalate",
        "i'll escalate",
        "i am escalating",
        "i'm escalating",

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


message = input(
    "\nEnter customer message: "
)


intent = predict_intent(message)


print(
    "\nPredicted intent:",
    intent
)


retrieved = retrieve(
    message,
    intent,
    k=5
)


print(
    "\nRetrieved historical cases:"
)


for i, item in enumerate(
    retrieved,
    1
):

    print(
        "\n--- Case",
        i,
        "---"
    )

    print(
        "Customer:",
        item["customer_issue"]
    )

    print(
        "Response:",
        item["historical_response"]
    )

    print(
        "Similarity:",
        round(
            item["similarity"],
            4
        )
    )


forced_escalation, escalation_reason = should_escalate(
    message,
    intent
)


print(
    "\nEscalation rule:",
    forced_escalation
)


prompt = build_prompt(
    message,
    intent,
    retrieved,
    forced_escalation,
    escalation_reason
)


result = generate_response(
    prompt
)


if result:

    if forced_escalation:

        result["escalate"] = True

        result["escalation_reason"] = escalation_reason

        result["reply"] = remove_false_escalation_claim(
            result.get("reply", "")
        )

    else:

        result["escalate"] = False

        result["escalation_reason"] = ""

    result["intent"] = intent

    print(
        "\n========== FINAL RESPONSE =========="
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )