import os
import subprocess
import shutil

from urllib.parse import unquote

import numpy as np
import pandas as pd 

from utils.graph_processing import construct_adjacency_list, export_graph_to_indexed_format


DEF_ARTICLES_PATH = "../data/paths-and-graph/articles.tsv"
DEF_CATEGORIES_PATH = "../data/paths-and-graph/categories.tsv"
DEF_LINKS_PATH = "../data/paths-and-graph/links.tsv"
DEF_FINISHED_PATH = "../data/paths-and-graph/paths_finished.tsv"
DEF_UNFINISHED_PATH = "../data/paths-and-graph/paths_unfinished.tsv"

DEF_EMBEDDINGS_PATH = "../src/data/article_embeddings_smaller.csv"
DEF_UNIQUE_GAMES_PATH = "../data/paths-and-graph/unique_games.tsv"
DEF_ADJ_LIST_PATH = "../data/paths-and-graph/adj_list.txt"
DEF_PAIR_STATS_PATH = "../data/paths-and-graph/pair_data.tsv"
DEF_NODE_STATS_PATH = "../data/paths-and-graph/node_data.tsv"

EXECTABLE_NAME = 'graph_stats'


def load_article_df(path: str = DEF_ARTICLES_PATH):
    articles_df = pd.read_csv(path, sep = "\t", comment = '#', header = None)
    articles_df.columns = ['article_name']

    # Decode names
    articles_df['article_name'] = articles_df['article_name'].apply(unquote) 
    
    print(f"Loaded {len(articles_df)} articles in df of shape {articles_df.shape}")
    
    return articles_df


def load_categories_df(path: str = DEF_CATEGORIES_PATH):
    categories_df = pd.read_csv(path, ep = "\t", comment = '#', header = None)
    categories_df.columns = ['article_name', 'category']

    # Decode article names
    categories_df['article_name'] = categories_df['article_name'].apply(unquote)

    # Split the 'category' column into multiple columns (one for each level of category)
    df_split = categories_df['category'].str.split('.', expand=True).drop(columns=[0])

    # Rename the columns to represent each level
    df_split.columns = ['Level_1', 'Level_2', 'Level_3']

    # Join the new columns with starting dataframe
    categories_df = categories_df.drop(columns = ['category']).join(df_split)
    
    print(f"Loaded {len(categories_df)} categories in df of shape {categories_df.shape}")
    
    return categories_df


def load_links_df(path: str = DEF_LINKS_PATH):
    links_df = pd.read_csv(path, sep = "\t", comment = '#', header = None)
    links_df.columns = ['source', 'target']

    # Decode article names
    links_df = links_df.map(unquote)
    
    print(f"Loaded {len(links_df)} links in df of shape {links_df.shape}")
    
    return links_df


def load_finished_df(path: str = DEF_FINISHED_PATH):
    finished_df = pd.read_csv(path, sep = "\t", comment = '#', header = None)
    finished_df.columns = ['hashIP', 'timestamp', 'duration', 'path', 'difficulty_rating']

    # Decode article names and transform path into list
    finished_df['path'] = finished_df['path'].apply(lambda a: [unquote(art) for art in a.split(";")])

    # Calculate path length
    finished_df['path_length'] = finished_df['path'].apply(len)

    # Calculate number of backward clicks in each path
    finished_df['num_backward'] = finished_df['path'].apply(lambda a: a.count("<"))

    # Convert timestamp to reasonable units 
    finished_df['timestamp'] = pd.to_datetime(finished_df['timestamp'], unit='s')
    
    print(f"Loaded {len(finished_df)} finished paths in df of shape {finished_df.shape}")
    return finished_df


def load_unfinished_df(path: str = DEF_UNFINISHED_PATH):
    unfinished_df = pd.read_csv(path, sep = "\t", comment = '#', header = None)
    unfinished_df.columns = ['hashIP', 'timestamp', 'duration', 'path', 'target_article', 'type_end']

    # Decode article names and transform path to list
    unfinished_df['path'] = unfinished_df['path'].apply(lambda a: [unquote(art) for art in a.split(";")])

    # Decode article names and transform path to list
    unfinished_df['target_article'] = unfinished_df['target_article'].apply(lambda a: unquote(a))

    # Calculate length of unfinished paths
    unfinished_df['path_length'] = unfinished_df['path'].apply(len)

    # Calculate number of backward clicks
    unfinished_df['num_backward'] = unfinished_df['path'].apply(lambda a: a.count("<"))

    # Convert timestampt to reasonable units 
    unfinished_df['timestamp'] = pd.to_datetime(unfinished_df['timestamp'], unit='s')
    
    print(f"Loaded {len(unfinished_df)} unfinished paths in df of shape {unfinished_df.shape}")
    return unfinished_df


def load_embeddings(path: str = DEF_EMBEDDINGS_PATH):
    embeddings_df = pd.read_csv(path)

    embeddings_df['article_name'] = embeddings_df['article_name'].apply(unquote)
    embeddings_df.index = embeddings_df['article_name']
    embeddings_df.drop(columns=['article_name'], inplace=True)
    embeddings_df.sort_index(inplace=True)

    # Turn the embedded value into numpy array
    embeddings_df['embedding'] = embeddings_df['embedding'].apply(lambda x: np.asarray(x.replace('[', '').replace(']', '').split(', '), dtype=np.float32))
    print(f"Loaded {len(embeddings_df)} embeddings in df of shape {embeddings_df.shape}")
    
    return embeddings_df


def preprocess_and_concat_unfinished_and_finished(unfinished_df, finished_df):
    finished_mod = finished_df.copy()
    finished_before = len(finished_mod)
    
    # Extract source and target articles
    finished_mod['source'] = finished_mod.path.apply(lambda a: a[0])
    finished_mod['target'] = finished_mod.path.apply(lambda a: a[-1])

    # Keep only paths that are after the first unfinished path (to avoid bias)
    finished_mod = finished_mod[finished_mod['timestamp'] > unfinished_df['timestamp'].min()]
    other_df = finished_mod[finished_mod['timestamp'] <= unfinished_df['timestamp'].min()]

    # Add a field that will be used to merge with the unfinished paths
    finished_mod['finished'] = True

    finished_mod = finished_mod.reindex(sorted(finished_mod.columns), axis=1)
    print(f"After filtering all paths after {unfinished_df['timestamp'].min()}")
    print(f"we kept {len(finished_mod)} paths out of {finished_before} finished paths")
    print(f"There are {len(unfinished_df)} unfinished paths")
    
    unfinished_mod = unfinished_df.copy()

    # Extract source article
    unfinished_mod['source'] = unfinished_mod.path.apply(lambda a: a[0])
    unfinished_mod.rename(columns={'target_article': 'target'}, inplace=True)

    # Add a field that will be used to merge with the finished paths
    unfinished_mod['finished'] = False

    unfinished_mod = unfinished_mod.reindex(sorted(unfinished_mod.columns), axis=1)
    
    # Concatenate the two dataframes
    all_games_df = pd.concat([finished_mod, unfinished_mod], ignore_index = True)

    return all_games_df, other_df


def prune_invalid_games(all_games_df, articles_df):
    # Keep only games that have valid source and target articles
    print(f"Pruning invalid games. Initially we have {len(all_games_df)} games")
    all_games_df = all_games_df[all_games_df['source'].isin(articles_df['article_name'])]
    all_games_df = all_games_df[all_games_df['target'].isin(articles_df['article_name'])]
    
    print(f"Pruned invalid games. Now we have {len(all_games_df)} valid games")
    
    return all_games_df


def compute_cosine_similarity(all_games_df, embeddings_df):
    all_games_df = all_games_df.join(embeddings_df, on = 'source')
    all_games_df.rename(columns = {'embedding': 'embedding_source'}, inplace = True)

    all_games_df = all_games_df.join(embeddings_df, on = 'target')
    all_games_df.rename(columns = {'embedding': 'embedding_target'}, inplace = True)

    # Embeddings are normalized so so a dot product is a cosine similarity
    all_games_df['cosine_similarity'] = all_games_df.apply(
        lambda x: np.dot(x['embedding_source'], x['embedding_target']),
        axis=1
    )
    
    all_games_df.drop(columns=['embedding_source', 'embedding_target'], inplace=True)
    
    return all_games_df
    
    
def merge_with_node_data(all_games_df, node_stats_df):
    all_games_df = all_games_df.join(node_stats_df, on = 'source')
    all_games_df = all_games_df.join(node_stats_df, on = 'target', rsuffix = '_target')

    all_games_df.rename(
        columns = {
            'degree': 'degree_source', 
            'closeness': 'closeness_source', 
            'betweenness': 'betweenness_source', 
            'pagerank': 'pagerank_source'
        }, 
        inplace = True
    )

    num_games_before = len(all_games_df)
    # Drop games which do not have node statistics
    all_games_df.dropna(subset = ['degree_source'], inplace = True)
    all_games_df.dropna(subset = ['degree_target'], inplace = True)
    if len(all_games_df) < num_games_before:
        print(f"Dropped {num_games_before - len(all_games_df)} games without node statistics")
    
    return all_games_df

    
def dump_unique_source_target_pairs(valid_games_df):
    unique_games = valid_games_df[['source', 'target']].drop_duplicates()
    unique_games.to_csv('../data/paths-and-graph/unique_games.tsv', sep = '\t', index = False, header = False)
    
    print(f"Dumped {len(unique_games)} unique source-target pairs")
    
    return unique_games


def dump_adjacency_list(links_df, articles_df, path: str = DEF_ADJ_LIST_PATH):
    # Get the list of article names
    articles = articles_df['article_name'].tolist()
    
    # Generate the adjacency list
    adj_list = construct_adjacency_list(links_df, articles)
    
    # Dump the adjacency list to a file
    export_graph_to_indexed_format(adj_list, path)
    
    # Count the number of articles and entries in the adjacency list
    num_articles = len(articles)
    num_entries = sum(len(targets) for targets in adj_list.values())
    
    print(f"Dumped adjacency list with {num_articles} articles and {num_entries} entries")
    
    return adj_list


def run_cpp_code_to_compute_unique_source_target_pair_stats(graph_path: str = DEF_ADJ_LIST_PATH, game_path: str = DEF_UNIQUE_GAMES_PATH):
    # Detect whether the necessary files are present
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Adjacency list not found at {graph_path}. Please make sure it is present.")
    
    if not os.path.exists(game_path):
        raise FileNotFoundError(f"Unique source-target pairs not found at {DEF_UNIQUE_GAMES_PATH}. Please run the function dump_unique_source_target_pairs() first.")
    
    # Detect whether CMakeLists.txt is present
    if not os.path.exists("CMakeLists.txt"):
        raise FileNotFoundError("CMakeLists.txt not found. Please ensure the C++ project is set up correctly.")
    print("CMakeLists.txt found")
    
    build_dir = os.path.join(os.getcwd(), 'build')
    executable = 'graph_stats'
    if os.name == 'nt':  # Windows
        executable += '.exe'
    
    executable_path = os.path.join(build_dir, executable)
    
    # Check whether the code has already been compiled
    if os.path.exists(build_dir):
        print("Build directory already exists")
        
        if os.path.exists(executable_path):
            print("Executable already exists")
        else:
            print("Executable not found. Cleaning build directory")
            shutil.rmtree(build_dir)
            os.mkdir(build_dir)
            print("Build directory cleaned and recreated")
    else :
        print("Build directory not found")
        os.mkdir(build_dir)
    
    # Build the C++ code from the build directory
    if not os.path.exists(executable_path):
        print("Compiling C++ code...")
        try:
            subprocess.run(['cmake', '-DCMAKE_BUILD_TYPE=Release', '..'], cwd=build_dir, check=True)
            subprocess.run(['make'], cwd=build_dir, check=True)
            print("C++ code compiled successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Build process failed with error: {e}")
        
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"Executable {executable} not found in the build directory after compilation")
    
    try:
        print("Be patient, the C++ code can execute for about 10 minutes...")
        subprocess.run([executable_path], cwd=build_dir, check=True)
        print("Ran C++ code to compute unique source-target pair stats. Check results in the appropriate folder.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Execution failed with error: {e}")


def load_or_compute_unique_source_target_pair_stats(path: str = DEF_PAIR_STATS_PATH):
    unique_game_stats = None
    
    try:
        unique_game_stats = pd.read_csv(path, sep = '\t', header = None)
        print(f"Loaded {len(unique_game_stats)} unique source-target pair stats")
        
    except FileNotFoundError:
        print("Unique source-target pairs not found. Computing them...")
        
        # Run the C++ code to compute the unique source-target pair stats
        run_cpp_code_to_compute_unique_source_target_pair_stats()
        
        # Try to load the unique source-target pair stats again
        try:
            unique_game_stats = pd.read_csv(path, sep = '\t', header = None)
            print(f"Loaded {len(unique_game_stats)} unique source-target pair stats")
        except FileNotFoundError:
            raise FileNotFoundError("Unique source-target pair stats not found even after running the C++ code")
        
    if unique_game_stats is None:
        raise RuntimeError("Unique source-target pair stats could not be loaded")
        
    unique_game_stats.columns = [
        'source', 'target', 'shortest_path_length', 
        'shortest_path_count', 'max_sp_pagerank', 'max_sp_avg_pagerank', 'avg_sp_avg_pagerank',
        'one_longer_path_count', 'max_ol_pagerank', 'max_ol_avg_pagerank', 'avg_ol_avg_pagerank', 
        'two_longer_pagerank', 'max_tl_pagerank','max_tl_avg_pagerank'
    ]

    unique_game_stats.set_index(['source', 'target'], inplace = True)
        
    return unique_game_stats


def load_or_compute_node_stats(path: str = DEF_NODE_STATS_PATH):
    node_stats = None
    
    try:
        node_stats = pd.read_csv(path, sep = '\t', header = None)
        print(f"Loaded {len(node_stats)} node stats")
        
    except FileNotFoundError:
        print("Node stats not found. Computing them...")
        
        # Run the C++ code to compute the node stats
        run_cpp_code_to_compute_unique_source_target_pair_stats()
        
        # Try to load the node stats again
        try:
            node_stats = pd.read_csv(path, sep = '\t', header = None)
            print(f"Loaded {len(node_stats)} node stats")
        except FileNotFoundError:
            raise FileNotFoundError("Node stats not found even after running the C++ code")
        
    if node_stats is None:
        raise RuntimeError("Node stats could not be loaded")
        
    node_stats.columns = ['article_name', 'degree', 'closeness', 'betweenness', 'pagerank']
    node_stats.index = node_stats['article_name']
    node_stats.drop(columns=['article_name'], inplace=True)
        
    return node_stats
    