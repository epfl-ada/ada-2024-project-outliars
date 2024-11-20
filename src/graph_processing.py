import pandas as pd
import numpy as np

from urllib.parse import unquote


def construct_adjecency_matrix(
        article_links: pd.DataFrame,
        articles: list
) -> pd.DataFrame:
    """
    Takes as input a list of source-destination pairs 

    Args:
        article_links (pd.DataFrame): A DataFrame with columns 'source' and 'target'
        articles (list): A list of all articles in the dataset
        
    Returns: 
        An adjacency matrix representation of the provided links
    """
    
    # Create a mapping from article names to indices
    article_to_index = {article: idx for idx, article in enumerate(articles)}
    n = len(articles)
    
    # Map 'source' and 'target' to indices
    source_indices = article_links['source'].map(article_to_index)
    target_indices = article_links['target'].map(article_to_index)
    
    # Create the underlying matrix, to avoid overflow in addition 16 bits should be enough
    adj_matrix = pd.DataFrame(
        data=np.zeros((n, n), dtype=np.uint16),
        index=articles,
        columns=articles
    )
    
    # Set the corresponding entries to 1
    adj_matrix.values[source_indices, target_indices] = 1
    
    return adj_matrix


def construct_adjacency_list(
        article_links: pd.DataFrame,
        articles: list
) -> dict:
    """
    Takes as input a list of source-destination pairs 

    Args:
        article_links (pd.DataFrame): A DataFrame with columns 'source' and 'target'
        articles (list): A list of all articles in the dataset
        
    Returns: 
        An adjacency list representation of the provided links
    """
    
    # adj_list = {article: [] for article in articles}

    # # Add existing links
    # for _, row in article_links.iterrows():
    #     source, target = row['source'], row['target']
    #     adj_list[source].append(target)
        
    # return adj_list
    
    # Group targets by source
    adj_series = article_links.groupby('source')['target'].apply(list)
    
    # Ensure all articles are included
    adj_list = {article: adj_series.get(article, []) for article in articles}
    
    return adj_list


def from_adjacency_list_to_matrix(adj_list: dict) -> pd.DataFrame:
    """
    Converts an adjacency list to an adjacency matrix
    
    Args:
        adj_list (dict): An adjacency list representation of the provided links
        articles (list): A list of all articles in the dataset
        
    Returns: 
        An adjacency matrix representation of the provided links
    """
    
    # Create a DataFrame from the adjacency list
    edges = [(source, target) for source, targets in adj_list.items() for target in targets]
    edges_df = pd.DataFrame(edges, columns=['source', 'target'])
    
    # Pivot the DataFrame to create the adjacency matrix
    adj_matrix = edges_df.assign(value=1).pivot(index='source', columns='target', values='value').fillna(0)
    
    # Ensure the matrix includes all articles
    articles = list(adj_list.keys())
    adj_matrix = adj_matrix.reindex(index=articles, columns=articles, fill_value=0)
    
    return adj_matrix


def from_adjacency_matrix_to_list(adj_matrix: pd.DataFrame) -> dict:
    """
    Converts an adjacency matrix to an adjacency list
    
    Args:
        adj_matrix (pd.DataFrame): An adjacency matrix representation of the provided links
        
    Returns: 
        An adjacency list representation of the provided links
    """
    
    # Use boolean indexing and apply to avoid nested loops
    adj_list = (adj_matrix != 0).apply(lambda row: row[row].index.tolist(), axis=1).to_dict()
    
    return adj_list


def read_distance_matrix(file_path: str, articles_df: pd.DataFrame, skip_lines: int = 17, delimiter: str = None) -> pd.DataFrame:
    """
    Reads a shortest-path distance matrix from a text file and constructs a DataFrame.

    Args:
        file_path (str): Path to the text file containing the distance matrix.
        articles_df (pd.DataFrame): DataFrame containing 'article_name' column with article names.
        skip_lines (int, optional): Number of lines to skip at the beginning of the file. Defaults to 17.
        delimiter (str, optional): Delimiter used in the file. If None, whitespace is used.

    Returns:
        pd.DataFrame: DataFrame representing the distance matrix with article names as index and columns.
    """
    # Retrieve list of article names
    article_names = articles_df['article_name'].tolist()
    article_names = [unquote(x) for x in article_names] 
    
    with open('data/paths-and-graph/shortest-path-distance-matrix.txt', 'r') as file:
        lines = file.readlines()

    # Skip metadata lines
    lines = lines[17:]
    
    # Ensure the number of articles matches the number of lines
    assert len(lines) == len(article_names) 
    
    # Transform each line into a list of distances
    distances = np.empty((len(lines), len(lines)), dtype=np.float32)
    for i, line in enumerate(lines):
        # Treat each character as a distance
        distances[i] = np.array([np.nan if char == '_' else int(char) for char in line.strip()])

    # Create the distance matrix dataframe
    distance_df = pd.DataFrame(distances, columns=article_names, index=article_names)
    
    return distance_df


def export_graph_to_indexed_format(adj_list: dict, output_file: str):
    """
    Exports the adjacency list to a text file with an index mapping and adjacency lists using indices.

    Args:
        adj_list (dict): The adjacency list of the graph where keys are node names and values are lists of adjacent nodes.
        output_file (str): The path to the output text file.
    """
    # Step 1: Create an index mapping
    # Collect all unique nodes from keys and values
    unique_nodes = set(adj_list.keys())
    for neighbors in adj_list.values():
        unique_nodes.update(neighbors)
    unique_nodes = sorted(unique_nodes)  # Optional: sort nodes for consistency
    
    # Assign a unique ID to each node
    node_to_id = {node: idx for idx, node in enumerate(unique_nodes)}
    
    # Step 2: Write the index mapping and adjacency lists to the output file
    with open(output_file, 'w') as f:
        # Write the index mapping
        f.write('# Node Index Mapping\n')
        for node, idx in node_to_id.items():
            f.write(f'{idx}\t{node}\n')
        f.write('\n')  # Add a newline for readability

        # Write the adjacency lists using indices
        f.write('# Adjacency Lists\n')
        for node in unique_nodes:
            node_id = node_to_id[node]
            neighbors = adj_list.get(node, [])
            neighbor_ids = [node_to_id[neighbor] for neighbor in neighbors]
            # Write the node ID followed by its neighbor IDs
            neighbor_ids_str = ' '.join(map(str, neighbor_ids))
            f.write(f'{node_id}: {neighbor_ids_str}\n')
            