import pandas as pd
import re

input_file = "AmericanAir_customer_messages.csv"

df = pd.read_csv(input_file)

print("Original rows:", len(df))

# Remove rows with missing text
df = df.dropna(subset=["text"])

# Clean text
df["text"] = df["text"].astype(str).str.strip()

# Remove empty messages
df = df[df["text"] != ""]

# Remove exact duplicate messages
df = df.drop_duplicates(subset=["text"])

# Remove very short messages
short_df = df[df["text"].str.split().str.len() <= 3].copy()
df = df[df["text"].str.split().str.len() > 3]

# Keep messages that are actually related to AmericanAir
aa_pattern = r"@AmericanAir\b|AmericanAir|American Airlines"

related_df = df[
    df["text"].str.contains(
        aa_pattern,
        case=False,
        regex=True,
        na=False
    )
].copy()

# Flag possible thread fragments
related_df["possible_thread_fragment"] = (
    related_df["text"].str.match(r"^@\w+", na=False)
)

# Flag possible multi-intent messages
multi_intent_words = [
    "and also",
    "also",
    "as well as",
    "plus",
    "another issue",
    "another problem",
    "and my"
]

pattern = "|".join(re.escape(x) for x in multi_intent_words)

related_df["possible_multi_intent"] = (
    related_df["text"]
    .str.lower()
    .str.contains(pattern, regex=True, na=False)
)

# Save cleaned dataset
related_df.to_csv(
    "AmericanAir_customer_clean.csv",
    index=False
)
short_df.to_csv(
    "AmericanAir_short_messages.csv",
    index=False
)
print("After cleaning:", len(related_df))
print("Short messages saved:", len(short_df))
print("Saved: AmericanAir_customer_clean.csv")
print("Saved: AmericanAir_short_messages.csv")