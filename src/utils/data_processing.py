import os
import subprocess
import shutil

from urllib.parse import unquote

import numpy as np
import pandas as pd 

from utils.graph_processing import construct_adjacency_list, export_graph_to_indexed_format, construct_adjecency_matrix, from_adjacency_matrix_to_list, generate_inverse_index_mapping, load_pair_data_with_multiindex


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

DEF_LINK_PROB_PATH = "../src/data/link_probabilities.csv"
DEF_FAME_PATH = "../src/data/topic_fame_updated.csv"

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


def find_impossible_games(games_src, articles_df, dead_end, isolated_target, distance_df):
    counter = 0
    typo_src, typo_trg = [], []
    isol_src, isol_trg = [], []
    print("Impossible games because of no existing path:")
    for i, row in games_src.iterrows():
        if not i in articles_df['article_name'].values:
            typo_src.append(i)
            continue
        if not row['target'] in articles_df['article_name'].values:
            typo_trg.append(row['target'])
            continue
        if np.isin(i, dead_end):
            isol_src.append(i)
        if np.isin(row['target'], isolated_target):
            isol_trg.append(row['target'])
        if pd.isna(distance_df.loc[i, row['target']]):
            print(i, "-", row['target'])
            counter += 1
    return counter, list(set(typo_src)), list(set(typo_trg)), list(set(isol_src)), list(set(isol_trg))

def get_manually_added_links():
    return [
        ('Finland', 'Åland'), 
        ('Republic_of_Ireland', 'Éire'), 
        ('Claude_Monet', 'Édouard_Manet'), 
        ('Impressionism', 'Édouard_Manet'), 
        ('Ireland', 'Éire'), 
        ('Francisco_Goya', 'Édouard_Manet')
    ]

def load_links_df(path: str = DEF_LINKS_PATH, articles_df = None):
    links_df = pd.read_csv(path, sep = "\t", comment = '#', header = None)
    links_df.columns = ['source', 'target']

    # Decode article names
    links_df = links_df.map(unquote)
    
    # Print how many were loaded from the file
    print(f"Loaded {len(links_df)} links in df of shape {links_df.shape}")

    
    # Create a dataframe with the links that are not in the articles dataframe
    missing_links = get_manually_added_links()

    missing_links_df = pd.DataFrame(missing_links, columns = ['source', 'target'])
    
    # Concatenate the missing links with the existing links
    links_df = pd.concat([links_df, missing_links_df], ignore_index = True)
    
    # Create a df that contains the links to license information for every article if the articles_df is given
    if articles_df is not None:
        license_links_df = pd.DataFrame(columns = ['source', 'target'])
        license_links_df['source'] = articles_df['article_name']
        license_links_df['target'] = 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'
        
        links_df = pd.concat([links_df, license_links_df], ignore_index = True)

    print(f"After adding missing links, there are {len(links_df)} links in df")
    
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

def load_fame(path: str = DEF_FAME_PATH):
    fame_df = pd.read_csv(path)
    fame_df.drop(columns = ['decoded_article'], inplace = True)
    fame_df['encoded_article'] = fame_df['encoded_article'].apply(unquote)
    fame_df.rename(columns={"encoded_article": "article_name"}, inplace=True)

    fame_df.index = fame_df['article_name']
    fame_df.drop(columns=['article_name'], inplace=True)
    fame_df.sort_index(inplace=True)
    
    return fame_df

def load_pair_data():
    articles_df = load_article_df()
    links_df = load_links_df()
    adj_matrix = construct_adjecency_matrix(links_df, articles_df['article_name'].tolist())
    adj_list = from_adjacency_matrix_to_list(adj_matrix)
    index_mapping = generate_inverse_index_mapping(adj_list)
    pair_data = load_pair_data_with_multiindex('../src/data/pair_stats.txt', index_mapping)
    return pair_data

def load_link_proba(path: str = DEF_LINK_PROB_PATH):
    link_proba_df = pd.read_csv(path)
    link_proba_df.drop(columns = ['decoded_source', 'decoded_target'], inplace = True)
    link_proba_df['encoded_source'] = link_proba_df['encoded_source'].apply(unquote)
    link_proba_df['encoded_target'] = link_proba_df['encoded_target'].apply(unquote)
    link_proba_df.rename(columns={"encoded_source": "article_source", "encoded_target": "article_target"}, inplace=True)
    link_proba_df.set_index(['article_source', 'article_target'], inplace=True)
    #print(link_proba_df.columns)
    #link_proba_df.drop(columns = ['article_source', 'article_target'], inplace = True)
    
    return link_proba_df

def remove_games_with_not_existing_link(games_df, links_df):
    len1 = games_df.shape[0]
    invalid_pairs = []
    def check_consecutive_links(path):
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            if (source == '<') or (target == '<'):
                continue
            if not ((links_df['source'] == source) & (links_df['target'] == target)).any():
                invalid_pairs.append((source, target))
                return False 
        return True  # All pairs are valid
    games_df = games_df[games_df['path'].apply(lambda x: check_consecutive_links(x, links_df))]
    if len(invalid_pairs) > 0:
            print(f"Removed {len1 - len(invalid_pairs)} games that contained non existing links, such as: {invalid_pairs}")
    return games_df

def remove_unexisting_link(games_df, remove = False):
    # after analysis, we saw that all games have a link to 'Wikipedia_Text_of_the_GNU_Free_Documentation_License' but this leads to nowhere so we should remove such games, as well as games containing non existing links
    if remove == True:
        return games_df
    len1 = games_df.shape[0]
    
    def check_consecutive_links(path):
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            if (source == '<') or (target == '<'):
                continue

            if (source, target) in [('Finland', 'Åland'), ('Republic_of_Ireland', 'Éire'), ('Claude_Monet', 'Édouard_Manet'), ('Impressionism', 'Édouard_Manet'), ('Ireland', 'Éire'), ('Francisco_Goya', 'Édouard_Manet')]:
                
                return False 
            if (source == 'Wikipedia_Text_of_the_GNU_Free_Documentation_License') or (target == 'Wikipedia_Text_of_the_GNU_Free_Documentation_License'):
                
                return False
        return True  # All pairs are valid
    games_df = games_df[games_df['path'].apply(lambda x: check_consecutive_links(x))]
    if (len(games_df) < len1) > 0:
            print(f"Removed {len1 - len(games_df)} games that contained non existing links")
    return games_df
    


def preprocess_and_concat_unfinished_and_finished(unfinished_df, finished_df):
    # Keep only the games that happened after the first unfinished game
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

def prune_timeout_games(all_games_df):
    # Remove games that ended with a timeout
    all_games_df = all_games_df[all_games_df['type_end'] != 'timeout']
    print(f"After removing timeouted games, there are {len(all_games_df)} games left")
    return all_games_df


def load_preprocessed_games(path1 = "../data/paths-and-graph/paths_finished.tsv", path2 = "../data/paths-and-graph/paths_unfinished.tsv", path3= DEF_ARTICLES_PATH, remove_timeout = True, remove = False):
    # Loads all games, removes the one before 2011, remove the ones with invalid article names, potentially remove timeouted
    finished_df = load_finished_df(path1)
    unfinished_df = load_unfinished_df(path2)
    gamess, _ = preprocess_and_concat_unfinished_and_finished(unfinished_df, finished_df)
    articles_df = load_article_df(path3)
    new_games = prune_invalid_games(gamess, articles_df)
    new_games = remove_unexisting_link(new_games, remove)
    if remove_timeout:
        neww_games = prune_timeout_games(new_games)
    return neww_games


def compute_cosine_similarity(all_games_df, embeddings_df, pairs = [['source', 'target']]):
    for pair in pairs:
        col1, col2 = pair[0], pair[1]
    
        all_games_df = all_games_df.join(embeddings_df, on=col1)
        all_games_df.rename(columns={'embedding': f'embedding_{col1}'}, inplace=True)
    
        all_games_df = all_games_df.join(embeddings_df, on=col2)
        all_games_df.rename(columns={'embedding': f'embedding_{col2}'}, inplace=True)
    
        sim_column_name = f'cosine_sim_{col1}_{col2}'
        all_games_df[sim_column_name] = all_games_df.apply(
            lambda x: np.dot(x[f'embedding_{col1}'], x[f'embedding_{col2}']),
            axis=1
        )
    
        # Drop the temporary embedding columns for this pair
        all_games_df.drop(columns=[f'embedding_{col1}', f'embedding_{col2}'], inplace=True)


    return all_games_df
    
    
def merge_with_node_data(all_games_df, node_stats_df, columns = ['source', 'target'], data = ['degree', 'closeness', 'betweenness', 'pagerank']):
    all_games_df = all_games_df.copy()
    for i in range(0, len(columns)):
        all_games_df = all_games_df.join(node_stats_df[data], on = columns[i], rsuffix = '_'+columns[i])
    
    rename_dict = {}

    for i in data:
        rename_dict[i] = i + '_' + columns[0]

    all_games_df = all_games_df.rename(
        columns = rename_dict
    )

    num_games_before = len(all_games_df)
    # Drop games which do not have node statistics
    matching_columns = [col for col in all_games_df.columns if any(col.startswith(prefix) for prefix in data)]
    all_games_df.dropna(subset=matching_columns, inplace=True)
    
    if len(all_games_df) < num_games_before:
        print(f"Dropped {num_games_before - len(all_games_df)} games without node statistics")
    
    return all_games_df


def merge_with_fame_data(all_games_df, fame_df, columns = ['source', 'target']):
    all_games_df = all_games_df.copy()
    for i in range(0, len(columns)):
        all_games_df = all_games_df.join(fame_df['fame_score'], on = columns[i], rsuffix = '_'+columns[i])

    rename_dict = {}
    rename_dict['fame_score'] = 'fame_score' + '_' + columns[0]

    all_games_df = all_games_df.rename(
        columns = rename_dict
    )

    num_games_before = len(all_games_df)
    # Drop games which do not have fame statistics
    matching_columns = [col for col in all_games_df.columns if col.startswith('fame_score')]
    all_games_df.dropna(subset=matching_columns, inplace=True)
    
    if len(all_games_df) < num_games_before:
        print(f"Dropped {num_games_before - len(all_games_df)} games without fame statistics")

    return all_games_df

def add_pair_data(all_games_df, pair_data, pairs = [['source', 'target']], names = [""], data = ['shortest_path_length', 'shortest_path_count', 'max_sp_node_degree',
       'max_sp_avg_node_degree', 'avg_sp_avg_node_degree',
       'one_longer_path_count', 'max_ol_node_degree', 'max_ol_avg_node_degree',
       'avg_ol_avg_node_degree', 'two_longer_path_count', 'max_tl_node_degree',
       'max_tl_avg_node_degree']):
    if len(pairs) != len(names):
        raise ValueError("Error: give a column name for each given pair !")
    num_games_before = len(all_games_df)
    c = []
    for i, name in enumerate(names):
        if len(pairs[i]) != 2:
            raise ValueError(f"Each pair must contain exactly 2 column names. Invalid pair: {pair}")
        for d in data:
            all_games_df[d + "_" + name] = all_games_df.apply(
                    lambda row: (
                        0 if ((row[pairs[i][0]] == '<') | (row[pairs[i][1]] == '<'))  # Check for '<' first
                        else (
                            None if ((row[pairs[i][0]], row[pairs[i][1]]) not in pair_data.index)  # Then check if tuple is in index
                            else pair_data[d].loc[(row[pairs[i][0]], row[pairs[i][1]])]  # Otherwise, retrieve value from pair_data
                        )
                    ),
                    axis=1
                )
        c.append(d+"_"+name)
        
    all_games_df.dropna(subset=c, inplace=True)
    
    if len(all_games_df) < num_games_before:
        print(f"Dropped {num_games_before - len(all_games_df)} games without link statistics")

    return all_games_df

def add_link_proba_info(all_games_df, link_proba, pairs, names):
    if len(pairs) != len(names):
        raise ValueError("Error: give a column name for each given pair !")
    num_games_before = len(all_games_df)
    for i, name in enumerate(names):
        if len(pairs[i]) != 2:
            raise ValueError(f"Each pair must contain exactly 2 column names. Invalid pair: {pair}")
        all_games_df[name] = all_games_df.apply(
                        lambda row: (
                            0 if ((row[pairs[i][0]] == '<') | (row[pairs[i][1]] == '<'))  # Check for '<' first
                            else (
                                None if ((row[pairs[i][0]], row[pairs[i][1]]) not in link_proba.index)  # Then check if tuple exists in index
                                else link_proba.loc[(row[pairs[i][0]], row[pairs[i][1]]), 'link_probability']  # Retrieve value if valid
                            )
                        ),
                        axis=1
                    )

        #all_games_df[name] = all_games_df.apply(
        #        lambda row: print(row) or link_proba.loc[row[pairs[i][0]], row[pairs[i][1]]], axis=1
        #    )
    all_games_df.dropna(subset=names, inplace=True)
    
    if len(all_games_df) < num_games_before:
        print(f"Dropped {num_games_before - len(all_games_df)} games without link statistics")

    return all_games_df
        

    
def dump_unique_source_target_pairs(valid_games_df):
    #Extract all pairs of all paths that appear in the dataset
    unique_games = set()
    for path, target in zip(valid_games_df['path'], valid_games_df['target']):
        for i in range(len(path) - 1):
            source = path[i]
            
            if source != '<':
                unique_games.add((source, target))
            
    unique_games = pd.DataFrame(list(unique_games), columns = ['source', 'target'])
            
    # Drop duplicates
    unique_games.drop_duplicates(inplace = True)
    
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
    