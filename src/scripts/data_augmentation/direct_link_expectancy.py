"""
This script estimates the probability that a specific link (X -> Y) exists in a Wikipedia-like dataset 
(Wikispeedia game). For each pair of articles, we calculate P(Y | X), which is the likelihood that article X 
contains a direct link to article Y.

**Features:**
- Processes in batches and saves each batch to an output file immediately to prevent data loss.
- Skips rows with already computed probabilities, allowing the script to resume from where it left off.
- Handles 100,000+ links efficiently by only processing rows without probabilities.

**Examples:**
- X = "Ivica Olić", Y = "Football" → P(Y | X) ≈ 0.9 (High probability: "Football" is central to "Ivica Olić").
- X = "Football", Y = "Ivica Olić" → P(Y | X) ≈ 0.01 (Low probability: "Football" is broad, unlikely to link to "Ivica Olić").
- X = "Football", Y = "Cristiano Ronaldo" → P(Y | X) ≈ 0.9 (Football is broad, but Cristiano Ronaldo is the top dog).
"""

# import api key from .env.local 
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
client = OpenAI(api_key=OPEN_AI_KEY)  # Replace with your OpenAI API key

script_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
input_file = os.path.abspath(os.path.join(script_dir, "../../../data/paths-and-graph/links.tsv"))
output_file = os.path.abspath(os.path.join(script_dir, "../../../src/data/link_probabilities.csv"))

# Step 1: Load and decode articles
if not pd.io.common.file_exists(output_file):
    # If output file doesn't exist, create it by copying input and adding a column
    df = pd.read_csv(input_file, sep="\t", comment="#", header=None, names=["encoded_source", "encoded_target"])
    df["decoded_source"] = df["encoded_source"].apply(urllib.parse.unquote)
    df["decoded_target"] = df["encoded_target"].apply(urllib.parse.unquote)
    df["link_probability"] = None  # Initialize empty probabilities column
    df.to_csv(output_file, index=False)
else:
    # If output file exists, load it
    df = pd.read_csv(output_file)

# Step 2: Define a function for batch processing
def estimate_link_probability(pairs):
    try:
        # Extract source and target articles
        sources = [pair[0] for pair in pairs]
        targets = [pair[1] for pair in pairs]
        
        # Craft the prompt for estimating link probabilities
        messages = [
            {"role": "system", "content": "You are a helpful assistant that evaluates the likelihood of links between Wikipedia articles."},
            {
                "role": "user",
                "content": (
                    "For each pair of articles below, estimate the probability (between 0 and 1) that the second article is directly linked from the first article:\n Some examples include:  X = Ivica Olić, Y = Football → P(Y | X) ≈ 1, X = Football, Y = Ivica Olić → P(Y | X) ≈ 0.01 Football is broad, unlikely to link to Ivica Olić, X = Football, Y = Cristiano Ronaldo → P(Y | X) ≈ 0.9 (Football is broad, but Cristiano Ronaldo is the top dog). Only include numbers and no decription at all. Here is the list of X and their Y: \n"
                    + "\n".join(f"{i+1}. {src} -> {tgt}" for i, (src, tgt) in enumerate(pairs))
                ),
            },
        ]

        # API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        # Extract probabilities from the response
        response_text = response.choices[0].message.content.strip()
        print(f"Batch request: {pairs}")  # Print for debugging
        print(f"Batch response: {response_text}")  # Print raw LLM output

        # Use regex to extract probabilities
        probabilities = re.findall(r"0\.\d+", response_text)
        return [float(prob) for prob in probabilities]
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [None] * len(pairs)

# Step 3: Process links in batches
batch_size = 20

# Filter rows where probability is not yet computed
unprocessed_df = df[df["link_probability"].isna()]

for i in range(0, len(unprocessed_df), batch_size):
    batch = unprocessed_df.iloc[i : i + batch_size]
    pairs = list(zip(batch["decoded_source"], batch["decoded_target"]))
    batch_probabilities = estimate_link_probability(pairs)

    # Update probabilities in the DataFrame
    try:
        unprocessed_df.loc[batch.index, "link_probability"] = batch_probabilities
    except ValueError as e:
        print(f"Error assigning probabilities: {e}. Skipping this batch.")
        continue  # Skip to the next batch if assignment fails
    
    # Save updated DataFrame to file after each batch
    df.update(unprocessed_df)  # Update original DataFrame with new probabilities
    df.to_csv(output_file, index=False)

    print(f"Batch {i // batch_size + 1} processed and saved.")
    time.sleep(1)  # Respect API rate limits

print("Processing complete!")
