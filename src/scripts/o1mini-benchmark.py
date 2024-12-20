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
    moves = ', '.join(neighbors)
    back_option = path_so_far[-2] if len(path_so_far) > 1 else None

    user_msg = f"""
You are simulating a Wikispeedia player. Your goal is to reach the target article by choosing one link at a time. 
Current state:
- Current article: {current_node}
- Target article: {goal_node}
- Path so far: {' -> '.join(path_so_far)}

You can choose one of the following:
- Click on one of these links: {moves}
{"- Go back to: " + back_option if back_option else ""}
- Give up (type 'give up' to stop).
IT IS VERY IMPORTANT TO ONLY Provide the chosen article name in plain text or 'give up'. EVERYTHING ELSE IS STRICTLY FORBIDDEN.
"""
    return user_msg

def interpret_model_response(response, neighbors, back_option):
    """
    Interpret the model's response.
    """
    choice = response.strip().lower()
    normalized_neighbors = [n.lower() for n in neighbors]
    normalized_back_option = back_option.lower() if back_option else None

    if choice == "give up":
        return None
    if choice in normalized_neighbors:
        return neighbors[normalized_neighbors.index(choice)]
    if back_option and choice == normalized_back_option:
        return back_option
    return None

# ---------------------------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------------------------

file_exists = os.path.exists(OUTPUT_FILE)

for source, target in ARTICLE_PAIRS:
    path = [source]
    won = False

    for depth in range(MAX_DEPTH + 1):
        if path[-1] == target:
            won = True
            break
        if depth > MAX_DEPTH:
            break

        current_node = path[-1]
        neighbors = graph.get(current_node, [])
        back_option = path[-2] if len(path) > 1 else None

        # Build prompt
        user_msg = build_prompt(current_node, target, path)

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": user_msg}
                ],
            )
        except Exception as e:
            print(f"Error calling the model: {e}")
            break

        if not response.choices:
            print("No response from the model.")
            break

        model_answer = response.choices[0].message.content.strip()
        print(f"Model response: {model_answer}")
        
        choice = interpret_model_response(model_answer, neighbors, back_option)

        if choice is None:
            break
        else:
            path.append(choice)

    result = {
        "model_name": MODEL_NAME,
        "source": source,
        "target": target,
        "path": " -> ".join(path),
        "won": won
    }

    result_df = pd.DataFrame([result])
    result_df.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)

    if not file_exists:
        file_exists = True

    print(f"Results updated: {OUTPUT_FILE}")

print("Simulation complete!")
