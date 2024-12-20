"""
This script integrates missing articles into the existing output file and estimates their link probabilities 
using an LLM. It ensures consistency with the current dataset format and saves progress after processing 
each article.
"""

# Import libraries
from dotenv import load_dotenv
import os
import pandas as pd
import re
from openai import OpenAI
import time

# Initialize OpenAI client
load_dotenv()
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=OPEN_AI_KEY)

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.abspath(os.path.join(script_dir, "../../../src/data/link_probabilities.csv"))

# Missing articles to add (source, target)
missing_articles = [
    ('Batman', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Finland', 'Åland'),
    ('Programming_language', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Consolation_of_Philosophy', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Abbasid', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Yttrium', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Company_(law)', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Rabbit', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Aircraft', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Communication', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Railway_post_office', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Electron_beam_welding', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
    ('Actinium', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'),
]

# Step 1: Define a function to estimate probabilities using the LLM
def estimate_link_probability_single(pair):
    try:
        # Craft the prompt for a single pair
        source, target = pair
        messages = [
            {"role": "system", "content": "You are a helpful assistant that evaluates the likelihood of links between Wikipedia articles."},
            {
                "role": "user",
                "content": f"Estimate the probability (between 0 and 1) that the article '{target}' is directly linked from the article '{source}'. Only provide the probability as a numerical value (e.g., 0.85)."
            },
        ]

        # API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        # Extract probability from the response
        response_text = response.choices[0].message.content.strip()
        print(f"Processing pair: {pair}")
        print(f"Response: {response_text}")

        # Use regex to capture the probability
        match = re.match(r"0\.\d+|1\.0", response_text)
        if match:
            return float(match.group(0))
        else:
            print(f"Warning: No valid probability found for {pair}.")
            return None
    except Exception as e:
        print(f"Error processing pair {pair}: {e}")
        return None

# Step 2: Load existing CSV or initialize a new DataFrame
if os.path.exists(output_file):
    df = pd.read_csv(output_file, header=None, names=["source", "target", "decoded_source", "decoded_target", "link_probability"])
else:
    df = pd.DataFrame(columns=["source", "target", "decoded_source", "decoded_target", "link_probability"])

# Step 3: Process each missing article
for source, target in missing_articles:
    # Check if the pair is already in the DataFrame
    if ((df["source"] == source) & (df["target"] == target)).any():
        print(f"Skipping already processed pair: {source} -> {target}")
        continue

    # Estimate link probability
    probability = estimate_link_probability_single((source, target))

    # Append the new row to the DataFrame
    new_row = {
        "source": source,
        "target": target,
        "decoded_source": source,
        "decoded_target": target,
        "link_probability": probability,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save progress to the file
    df.to_csv(output_file, index=False, header=False)
    print(f"Pair {source} -> {target} processed and saved.")

    # Respect rate limits
    time.sleep(1)

print("All missing articles processed!")
