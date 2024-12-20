import pandas as pd
import urllib.parse
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Load environment variables (for OPENAI API key)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Input files
LINKS_FILE = "/Users/karlovrancic/Library/Mobile Documents/com~apple~CloudDocs/Epfl/ADA/ada-2024-project-outliars/data/paths-and-graph/links.tsv"

# Article pairs we want to test
ARTICLE_PAIRS = [
    ("Gas", "Pope_Benedict_XVI"),
    ("President_of_the_United_States", "Sea"),
    ("List_of_vegetable_oils", "Hot_air_balloon"),
    ("Buddha", "Earthquake"),
    ("AIDS", "Tour_de_France"),
    ("Slavery", "Ocean"),
    ("Bobcat", "Timur"),
    ("Bonobo", "Color_blindness"),
    ("Eruption_column", "Where_Did_Our_Love_Go"),
    ("Kenya", "Weapon"),
    ("Tooth_enamel", "Georgetown,_Guyana"),
    ("Thomas_Hobbes", "Trichinosis"),
    ("Gerald_Ford", "IG_Farben_Building"),
    ("Congo_River", "Alzheimer's_disease"),
    ("Asteroid", "Viking"),
    ("Pyramid", "Bean"),
    ("Gazelle", "Death_Valley_National_Park"),
    ("Sassanid_Empire", "Vole")
]

# Maximum path length before counting as fail
MAX_DEPTH = 25

# Output CSV
OUTPUT_FILE = "model_performance.csv"

# Model name to test
MODEL_NAME = "o1-mini"  

# ---------------------------------------------------------------------------
# LOADING THE GRAPH
# ---------------------------------------------------------------------------

# The links.tsv file format:
# linkSource   linkTarget

# We’ll build a dictionary of article -> list of neighbors
graph = {}
if os.path.exists(LINKS_FILE):
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            src, dst = line.split("\t")
            # Decode the article names from URL form
            src_decoded = urllib.parse.unquote(src)
            dst_decoded = urllib.parse.unquote(dst)
            if src_decoded not in graph:
                graph[src_decoded] = []
            graph[src_decoded].append(dst_decoded)
else:
    print(f"Links file '{LINKS_FILE}' not found.")
    exit(1)

# Ensure every node is in graph even if no out-links
for art in list(graph.keys()):
    for neigh in graph[art]:
        if neigh not in graph:
            graph[neigh] = []

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def build_prompt(current_node, goal_node, path_so_far):
    """
    Build the prompt for the model.
    """
    neighbors = graph.get(current_node, [])
    moves = list(neighbors)
    back_option = path_so_far[-2] if len(path_so_far) > 1 else None

    system_msg = "You are simulating a Wikispeedia player. Your goal is to reach the target article by choosing one link at a time. You can also go back to the previous article if you think you made a mistake in the past move. You can give up at any time, but only if you are fatally stuck (avoid if possible). You are only allowed to provide the article name or 'give up'."
    user_msg = f"""
Current state:
- Current article: {current_node}
- Target article: {goal_node}
- Path so far: {' -> '.join(path_so_far)}

You can choose one of the following next steps:
- Click on one of these links: {', '.join(moves)}.
{"- Go back to: " + back_option if back_option else ""}
- Give up (type 'give up' if you want to stop).

You must choose exactly one action. Just provide the chosen article name or 'give up'. YOU ARE STRICTLY FORBIDDEN TO WRITE ANYTHING ELSE. 
"""

    return system_msg, user_msg

def interpret_model_response(response, neighbors, back_option):
    """
    Interpret the model's response.
    """
    choice = response.strip()
    if choice == "give up":
        return None
    if choice in neighbors:
        return choice
    if back_option and choice == back_option:
        return choice
    return None

# ---------------------------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------------------------

# Check if the output file exists to determine if headers should be written
file_exists = os.path.exists(OUTPUT_FILE)

for source, target in ARTICLE_PAIRS:
    path = [source]
    won = False

    for depth in range(MAX_DEPTH + 1):
        # Check if we already reached the target
        if path[-1] == target:
            won = True
            break
        if depth > MAX_DEPTH:
            break

        current_node = path[-1]
        neighbors = graph.get(current_node, [])
        back_option = path[-2] if len(path) > 1 else None

        # Build prompt
        system_msg, user_msg = build_prompt(current_node, target, path)

        # Call LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=50,
            temperature=0.5
        )

        model_answer = response.choices[0].message.content.strip()
        print(f"Model response: {model_answer}")
        
        choice = interpret_model_response(model_answer, neighbors, back_option)

        if choice is None:
            break
        else:
            path.append(choice)

    # Record result
    result = {
        "model_name": MODEL_NAME,
        "source": source,
        "target": target,
        "path": " -> ".join(path),
        "won": won
    }

    # Convert the result to a DataFrame
    result_df = pd.DataFrame([result])

    # Append the result to the CSV file
    result_df.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)

    # After the first write, set file_exists to True to avoid writing headers again
    if not file_exists:
        file_exists = True

    print(f"Results updated: {OUTPUT_FILE}")

print("Simulation complete!")
