import pandas as pd
import re
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=OPEN_AI_KEY)

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.abspath(os.path.join(script_dir, "../../data/topic_fame_updated2.csv"))
output_file_path = os.path.abspath(os.path.join(script_dir, "../../data/topic_fame_updated3.csv"))

# Load data
df = pd.read_csv(file_path)

# Select rows with missing fame scores
missing_scores = df[df["fame_score"].isna()].copy()

# Function to estimate fame scores using OpenAI
def get_fame_scores_numbers(article_names):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that evaluates the fame of Wikipedia article subjects. The fame score is a number between 1 and 9, inclusive. Provide exactly one score for each article in the list, in the same order, and make sure there are no omissions."},
            {
                "role": "user",
                "content": f"Estimate the fame of the following Wikipedia articles and provide a list of numerical values only, in the same order:\n{', '.join(article_names)}"
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        response_text = response.choices[0].message.content.strip()
        print(f"Batch request: {article_names}")
        print(f"Batch response: {response_text}")

        # Enhanced regex to capture fame scores
        scores = re.findall(r"(?:\d+\.\s*)?([1-9])", response_text)

        # Ensure we have the correct number of scores
        if len(scores) != len(article_names):
            print(f"Warning: Batch size mismatch. Expected {len(article_names)}, got {len(scores)}. Filling missing values.")
            scores.extend([None] * (len(article_names) - len(scores)))

        return [float(score) if score is not None else None for score in scores]
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [None] * len(article_names)


# Process in batches and save after each batch
batch_size = 10

for i in range(0, len(missing_scores), batch_size):
    batch = missing_scores.iloc[i : i + batch_size]
    articles = batch["decoded_article"].tolist()

    # Get fame scores for the current batch
    batch_scores = get_fame_scores_numbers(articles)

    # Check for length mismatch
    if len(batch_scores) != len(articles):
        print(f"Warning: Batch size mismatch. Expected {len(articles)}, got {len(batch_scores)}.")
        batch_scores = [None] * len(articles)  # Fill with None to avoid breaking

    # Update the DataFrame with the batch results
    try:
        missing_scores.loc[batch.index, "fame_score"] = batch_scores
        df.update(missing_scores)

        # Save updated DataFrame to the same file
        df.to_csv(output_file_path, index=False)
        print(f"Batch {i // batch_size + 1} processed and saved.")
    except Exception as e:
        print(f"Error updating batch: {e}. Continuing with next batch.")
        continue  # Skip to the next batch

    # Respect rate limits
    time.sleep(1)

print("Processing complete!")
