# Since LLMs are not completely reliable, we use this to check if the fame scores are correct and update them if necessary.

import pandas as pd
import re
import time
from openai import OpenAI


client = OpenAI(api_key="**********")  # Replace ********** with your OpenAI API key


file_path = "topic_fame2.csv"  
df = pd.read_csv(file_path)


missing_scores = df[df["fame_score"].isna()].copy()


def get_fame_scores_numbers(article_names):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that evaluates the fame of Wikipedia article subjects. The fame score is a number between 0 and 10. For reference, Cristiano Ronaldo, USA, and Jesus are 10. 1 is something most 90%+ people have never heard of."},
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
        
        # Use regex to extract numbers
        scores = re.findall(r"\d+\.?\d*", response_text)  
        return [float(score) for score in scores]
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [None] * len(article_names)


batch_size = 10
fame_scores = []

for i in range(0, len(missing_scores), batch_size):
    batch = missing_scores.iloc[i : i + batch_size]
    articles = batch["decoded_article"].tolist()
    batch_scores = get_fame_scores_numbers(articles)
    fame_scores.extend(batch_scores if len(batch_scores) == len(articles) else [None] * len(articles))
    time.sleep(1)  # Respect rate limits


missing_scores["fame_score"] = fame_scores
df.update(missing_scores)


df.to_csv("topic_fame_updated.csv", index=False)
