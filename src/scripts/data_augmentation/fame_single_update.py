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
file_path = os.path.abspath(os.path.join(script_dir, "../../data/topic_fame_updated3.csv"))
output_file_path = os.path.abspath(os.path.join(script_dir, "../../data/topic_fame_updated3.csv"))

# Load data
df = pd.read_csv(file_path)

# Select rows with missing fame scores
missing_scores = df[df["fame_score"].isna()].copy()

# Function to estimate fame scores for a single article
def get_fame_score(article_name):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that evaluates the fame of Wikipedia article subjects. The fame score is a number between 1 and 9, inclusive. Provide exactly one score for the given article."},
            {
                "role": "user",
                "content": f"Estimate the fame of the following Wikipedia article:\n{article_name}"
            },
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        response_text = response.choices[0].message.content.strip()
        print(f"Article request: {article_name}")
        print(f"Response: {response_text}")

        # Enhanced regex to capture a single fame score
        match = re.match(r"([1-9])", response_text)
        if match:
            return float(match.group(1))
        else:
            print(f"Warning: No valid score found for article {article_name}.")
            return None
    except Exception as e:
        print(f"Error processing article {article_name}: {e}")
        return None

# Process each missing article individually and save after each one
for index, row in missing_scores.iterrows():
    article_name = row["decoded_article"]

    # Get fame score for the current article
    fame_score = get_fame_score(article_name)

    # Update the DataFrame
    try:
        missing_scores.at[index, "fame_score"] = fame_score
        df.at[index, "fame_score"] = fame_score

        # Save updated DataFrame to the same file
        df.to_csv(output_file_path, index=False)
        print(f"Article '{article_name}' processed and saved.")
    except Exception as e:
        print(f"Error updating article '{article_name}': {e}. Continuing to next article.")
        continue  # Skip to the next article

    # Respect rate limits
    time.sleep(1)

print("Processing complete!")
