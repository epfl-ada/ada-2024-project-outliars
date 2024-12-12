"""
This script estimates the probability that a specific link (X -> Y) exists in a Wikipedia-like dataset 
(Wikispeedia game). It processes only rows with missing probabilities in batches and saves the results 
incrementally to avoid data loss and allow resumption from where it left off.
"""

# Import libraries
from dotenv import load_dotenv
import os
import pandas as pd
import urllib.parse
import time
import re
from openai import OpenAI

# Initialize OpenAI client
load_dotenv()
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=OPEN_AI_KEY)

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.abspath(os.path.join(script_dir, "../../../data/paths-and-graph/links.tsv"))
output_file = os.path.abspath(os.path.join(script_dir, "../../data/link_probabilities.csv"))

# Step 1: Load data
if not pd.io.common.file_exists(output_file):
    # If output file doesn't exist, initialize it
    df = pd.read_csv(input_file, sep="\t", comment="#", header=None, names=["encoded_source", "encoded_target"])
    df["decoded_source"] = df["encoded_source"].apply(urllib.parse.unquote)
    df["decoded_target"] = df["encoded_target"].apply(urllib.parse.unquote)
    df["link_probability"] = None  # Initialize empty probabilities column
    df.to_csv(output_file, index=False)
else:
    # Load existing data
    df = pd.read_csv(output_file)

# Step 2: Define a function to estimate probabilities for a batch of pairs
def estimate_link_probability(pairs):
    try:
        # Format the LLM prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant that evaluates the likelihood of links between Wikipedia articles."},
            {
                "role": "user",
                "content": (
                    "For each pair of articles below, estimate the probability (between 0 and 1) that the second article is directly linked from the first article. "
                    "Only include numerical values, one for each pair. Here is the list:\n"
                    + "\n".join(f"{i+1}. {src} -> {tgt}" for i, (src, tgt) in enumerate(pairs))
                ),
            },
        ]

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        # Extract probabilities from response
        response_text = response.choices[0].message.content.strip()
        print(f"Batch request: {pairs}")
        print(f"Batch response: {response_text}")

        # Use regex to extract valid probabilities
        probabilities = re.findall(r"(?:0\.\d+|1\.0)", response_text)
        return [float(prob) for prob in probabilities]
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [None] * len(pairs)

# Step 3: Process missing probabilities in batches
batch_size = 20
unprocessed_df = df[df["link_probability"].isna()]  # Filter rows with missing probabilities

for i in range(0, len(unprocessed_df), batch_size):
    batch = unprocessed_df.iloc[i : i + batch_size]
    pairs = list(zip(batch["decoded_source"], batch["decoded_target"]))
    
    # Process the batch
    batch_probabilities = estimate_link_probability(pairs)

    # Handle mismatched or missing results
    if len(batch_probabilities) != len(batch):
        print(f"Warning: Batch size mismatch. Expected {len(batch)}, got {len(batch_probabilities)}.")
        batch_probabilities.extend([None] * (len(batch) - len(batch_probabilities)))

    # Update DataFrame with batch results
    try:
        unprocessed_df.loc[batch.index, "link_probability"] = batch_probabilities
        df.update(unprocessed_df)
        df.to_csv(output_file, index=False)  # Save progress
        print(f"Batch {i // batch_size + 1} processed and saved.")
    except Exception as e:
        print(f"Error updating batch: {e}. Continuing with the next batch.")
        continue

    # Respect rate limits
    time.sleep(1)

print("Processing complete!")
