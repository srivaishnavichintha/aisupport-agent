# Hiver SDE Intern — Customer Support AI Agent

This project is my solution to the Hiver SDE Intern take-home assignment.

I used the **Customer Support on Twitter** dataset and selected **AmericanAir** as the brand. The system takes a customer message, identifies the main issue, finds similar historical support cases, generates a response using those cases, and decides whether the issue should be escalated.

The main goal was not just to build the agent, but also to evaluate how reliable its responses are.

---

## What the system does

For each customer message:

1. **Intent classification** — identifies the main support issue.
2. **Retrieval** — finds similar historical AmericanAir cases.
3. **Response generation** — generates a reply using the retrieved cases.
4. **Escalation** — flags unresolved cases for human review.

Example:

```json
{
  "intent": "flight_disruption",
  "reply": "I’m sorry to hear about the delay. Could you please share your flight number so we can look into the status?",
  "escalate": false,
  "escalation_reason": ""
}
```

---

## Dataset

I used the **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`) and selected AmericanAir because it has a large number of conversations and a good variety of support issues.

The raw dataset is not included in this repository.

---

## Intent Classification

I defined 10 intents from the AmericanAir data:

- `flight_disruption`
- `baggage_problem`
- `positive_feedback`
- `checkin_boarding_overbooking`
- `inflight_complaint`
- `seat_upgrade`
- `customer_service_complaint`
- `refund_compensation`
- `general_negative`
- `loyalty_program`

The classifier uses:

- TF-IDF (unigrams + bigrams)
- Logistic Regression
- Balanced class weights

The final training set contains **2,714 labeled examples**.

The evaluation uses a separate **197-example Golden Set**. Candidate labels were LLM-assisted and the final evaluation set was manually verified.

---

## Leakage Prevention

To avoid evaluation leakage, complete conversations containing Golden Set examples were removed from the training and retrieval data.

This prevents the system from seeing another message or response from the same conversation during evaluation.

---

## Retrieval

The system retrieves historical customer issues using:

- TF-IDF
- Cosine similarity
- Predicted-intent filtering

The retrieved customer messages and their historical brand responses are passed to the LLM as context.

I chose TF-IDF because it is lightweight, fast and reproducible for this project.

---

## Response Generation

The response generator uses the customer's message, predicted intent and retrieved historical cases.

The LLM is instructed to stay grounded in the retrieved examples and avoid inventing actions or information.

Model:

**OpenAI GPT-OSS 20B via Groq**

A deterministic guardrail is also used to prevent the model from falsely claiming that it contacted or transferred a case to a human.

---

## Escalation

Escalation is handled using deterministic rules.

For example, when a customer says they have already contacted customer service and the issue is still unresolved, the system can flag the case for human review.

The current project does not have a separately human-labeled escalation dataset, so escalation is treated as a rule-based component rather than a fully supervised model.

---

# Evaluation

The system is evaluated on the 197-example Golden Set.

### Intent Classification

- Accuracy
- Precision / Recall / F1
- Macro-F1
- Confusion matrix
- Majority-class baseline

### Retrieval

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- Precision@k

Retrieval relevance is measured using intent consistency, so these metrics are a proxy rather than a direct human relevance judgment.

### Response Quality

An LLM judge scores generated responses on:

- Relevance
- Helpfulness
- Groundedness
- Safety / Accuracy
- Overall quality

Each score is from 1–5.

### Historical Response Comparison

Where a Golden Set message has a directly linked historical brand response, the generated response is also compared against that original response.

The historical response is treated as a reference, not absolute ground truth, since some original responses are short or incomplete.

Final evaluation numbers will be reported in the accompanying report.

---

# Failure Analysis

The main failure cases observed during development include:

- overlapping intents, especially general complaints and customer-service complaints;
- short or ambiguous customer messages;
- retrieval finding the correct intent but not the best specific situation;
- generic historical responses that provide limited help;
- LLM responses that may be more useful than the original response but use substantially different wording.

The final report contains concrete examples and analysis of the major failure modes.

---

# What is misleading about my headline number?

No single metric represents the quality of the entire agent.

For example, retrieval Recall@10 can be high even when the retrieved case is not the best semantic match, because the current retrieval evaluation uses intent consistency as a proxy.

Similarly, an LLM judge score is not the same as real customer satisfaction.

The results therefore need to be considered together with classification metrics, retrieval metrics, response-quality evaluation and failure analysis.

---

# Decision Log

Some important decisions made during the project:

- Selected AmericanAir because of its data volume and variety of support issues.
- Defined 10 intents based on patterns in the data.
- Created a 197-example manually verified Golden Set.
- Removed complete Golden conversations from training/retrieval data to prevent leakage.
- Used TF-IDF + Logistic Regression for a simple, reproducible classifier.
- Used intent-filtered TF-IDF retrieval.
- Used historical brand responses as context for response generation.
- Added deterministic guardrails against false escalation claims.
- Used deterministic escalation rules rather than unrestricted LLM decisions.
- Used an LLM judge for response-quality evaluation.
- Treated historical responses as reference examples rather than perfect ground truth.

---

# Repository Structure

```text
support-agent/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── build_retrieval.py
├── conversations-split.py
├── train_intent_classifier.py
├── label_retrieval_corpus.py
├── retrieval_baseline.py
├── check-leakage.py
│
├── generate_response.py
├── generate_golden_responses.py
├── llm_judge.py
├── compare_response.py
├── compare_response_llm.py
│
├── AmericanAir_Golden_Set.csv
├── AmericanAir_Golden_Conversations.csv
├── AmericanAir_Training_Conversations.csv
├── AmericanAir_labeled_training_v2.csv
├── AmericanAir_Retrieval_Corpus.csv
├── AmericanAir_Retrieval_Corpus_Labeled.csv
│
├── intent_classifier_model.pkl
└── intent_tfidf_vectorizer.pkl
```

---

# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

## Set the Groq API key

PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

The API key is not stored in the repository.

## Run the application

```powershell
py generate_response.py
```

## Rebuild the classifier

```powershell
py train_intent_classifier.py
```

## Rebuild retrieval data

```powershell
py build_retrieval.py
py label_retrieval_corpus.py
```

## Run evaluation

```powershell
py retrieval_baseline.py
py generate_golden_responses.py
py llm_judge.py
py compare_response.py
py compare_response_llm.py
```

The LLM-based evaluation requires a Groq API key.

---

## Limitations

This is a take-home prototype rather than a production support system.

The main limitations are:

- TF-IDF retrieval is weaker than modern embedding-based retrieval.
- Retrieval evaluation uses intent consistency as a proxy for relevance.
- Escalation does not yet have an independently labeled evaluation set.
- LLM-based evaluation is not a replacement for human evaluation.
- The historical Twitter responses are noisy and sometimes incomplete.

With more time, I would improve semantic retrieval, create human-labeled retrieval and escalation evaluation sets, and compare LLM-judge results against human judgments.
