import numpy as np
import pandas as pd
from utils.data_processing import *
from utils.graph_processing import *

# builds dataset from given dataset starting_games with features for predicting outcome based on first click, uses all games not won in 1 click or less
def build_dataset_for_first(starting_games, node_stats_df = None, embeddings_df = None, pair_data = None):
    starting_games = starting_games[starting_games['path_length'] > 1] # remove games of length 1
    condition = (starting_games['path_length'] == 2) & (starting_games['finished'] == True)
    starting_games = starting_games[~condition] # removes games won in one click

    starting_games['first_click'] = starting_games.apply(lambda row: row['path'][1], axis = 1)

    starting_games.drop(columns = ["difficulty_rating", 'duration', 'hashIP', 'num_backward', 'path_length', 'path','type_end', 'timestamp'], inplace = True)

    if node_stats_df is None:
        node_stats_df = load_or_compute_node_stats()
    starting_games = merge_with_node_data(starting_games, node_stats_df, columns = ['target', 'first_click'], data = ['pagerank'])

    if embeddings_df is None:
        embeddings_df = load_embeddings()
    starting_games = compute_cosine_similarity(starting_games, embeddings_df, pairs = [['first_click', 'target'], ['source', 'target']])

    if pair_data is None:
        pair_data = load_pair_data()
    starting_games = add_pair_data(starting_games, pair_data, pairs =[['first_click', 'target']], names = ["first"], data = ['shortest_path_length', 'shortest_path_count'])

    starting_games.drop(columns = ['source', 'target', 'first_click', 'cosine_sim_source_target'],inplace = True)
    starting_games['n'] = 1

    features_1 = ['pagerank_target',
       'pagerank_first_click', 'cosine_sim_first_click_target',
       'shortest_path_length_first', 'shortest_path_count_first']
    return starting_games, features_1


# builds a dataset starting from given dataset starting_games, using data for n-th click (one row -> one row)
def build_dataset(starting_games, n, ind = None, node_stats_df = None, embeddings_df = None, pair_data = None):
    cols = []
    for i in range(n): # extracting article 
        cols.append(f"{i+1}_click")
        starting_games[f"{i+1}_click"] = starting_games.apply(lambda row: row['path'][i+1], axis = 1)
    starting_games['duration'] = (n/(starting_games['path_length']-1))* starting_games['duration'] # extracting avg duration

    starting_games['num_back'] = starting_games.apply(lambda a: (a[cols] == '<').sum()/ (n-1), axis = 1) 
    print(starting_games['num_back'].describe())
    print(starting_games['num_back'].unique())

    # removing < sign
    starting_games['2_click'] = starting_games.apply(lambda row: row['2_click'] if (row['2_click'] != '<') else row['source'], axis = 1)
        
    if n > 2:
        for i in range(3, n+1):
            starting_games[f"{i}_click"] = starting_games.apply(
                    lambda row: row[f"{i}_click"] if (row[f"{i}_click"] != '<') else row[f"{i-2}_click"], axis=1)

    if node_stats_df is None:
        node_stats_df = load_or_compute_node_stats()
    cols.append('source')
    cols.append('target')
    starting_games = merge_with_node_data(starting_games, node_stats_df, columns = cols, data = ['pagerank'])
    cols.remove('target')
    temp = []
    for i in cols:
        temp.append(f"pagerank_{i}")
    starting_games['max_pagerank'] = starting_games.apply(lambda row: row[temp].max(), axis = 1)
    print(temp)

    if embeddings_df is None:
        embeddings_df = load_embeddings()
    starting_games = compute_cosine_similarity(starting_games, embeddings_df, pairs = [['source', 'target'], [f"{n}_click", 'target']])
    starting_games['cos_diff'] = starting_games[f'cosine_sim_{n}_click_target'] - starting_games['cosine_sim_source_target']

    index_before = starting_games.index

    if pair_data is None:
        pair_data = load_pair_data()
    starting_games = add_pair_data(starting_games, pair_data, pairs =[[f'{n}_click', 'target']], names = [f"{n}"], data = ['shortest_path_length', 'shortest_path_count'])

    index_after = starting_games.index

    if ind is not None:
        removed_indices = index_before.difference(index_after)
        ind.difference_update(removed_indices)
    
    for i in range(1, n):
        cols.append(f"pagerank_{i}_click")

    starting_games.rename(columns = {f"pagerank_{n}_click": 'pagerank_n_click', f"cosine_sim_{n}_click_target": 'cosine_sim_n_click_target', f'shortest_path_length_{n}': 'shortest_path_length_n', f'shortest_path_count_{n}': 'shortest_path_count_n'}, inplace = True)

    cols.append("cosine_sim_source_target")
    cols.append("pagerank_source")
    cols.extend(["difficulty_rating", 'hashIP', 'num_backward', 'path_length', 'path','type_end', 'timestamp'])
    cols.append('target')
    starting_games.drop(columns = cols, inplace = True)
    features = starting_games.columns.values.tolist()
    features.remove("finished")
    starting_games['n'] = n
    
    
    return starting_games, features


# creates dataset, for each row creates one row for each click from 2 to maxx, can also return dataset for first click (one row for each row)
# balanced: if one click of one row cannot be generated, remove all clicks from that row
# filter: remove all games that are not at least of length n before creating rows for nth click
# special for each -> return one dataset for each click
def create_from_dataset(datagames, whole_dataset, minn, maxx, filter = True, balanced = False, special_for_each = False, node_stats_df = None, embeddings_df = None, pair_data = None):
    if whole_dataset == None:
        whole_dataset = pd.DataFrame(columns=['duration','num_back','pagerank_n_click','pagerank_target', 'max_pagerank', 'cosine_sim_n_click_target','cos_diff','shortest_path_length_n', 'shortest_path_count_n','finished','n'])
    valid_indices = set(datagames.index)
    final = []
    if minn == 1:
        n_games = datagames.copy()
        if filter == True:
            n_games = n_games[n_games['path_length'] > 1]
            condition = (n_games['path_length'] == 2) & (n_games['finished'] == True)
            n_games = n_games[~condition] # removes games won in n clicks exactly
        data_one, _ = build_dataset_for_first(n_games, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data)
        data_one['n'] = data_one['n'].astype('float64')
        data_one['finished'] = data_one['finished'].astype('int')
    for n in range (2, maxx+1):
        print(f"Adding for click {n}...")
        #print(f"Prediction based on first {n} clicks: ")
        n_games = datagames.copy()
        if filter == True:
            n_games = n_games[n_games['path_length'] > n]
            condition = (n_games['path_length'] == (n+1)) & (n_games['finished'] == True)
            n_games = n_games[~condition] # removes games won in n clicks exactly
    
        #print("Number of games before building dataset: ", n_games.shape)
        
        dataset, features = build_dataset(n_games, n, valid_indices if balanced else None, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data)
        features.append("n")
        final.append(dataset)
        
        print("----------------------------------------------------")
        print()
    if balanced:
        if minn == 1:
            data_one = data_one.loc[list(valid_indices)]
        if special_for_each == True:
            for i in range(len(final)):
                final[i] = final[i].loc[list(valid_indices)]
                final[i]['n'] = final[i]['n'].astype('float64')
                final[i]['finished'] = final[i]['finished'].astype('int')
            whole_dataset = final
        else:
            for i in final:
                #valid_dataframes = [i.loc[list(valid_indices)] for i in final if not i.loc[list(valid_indices)].empty]
                #whole_dataset = pd.concat([whole_dataset] + valid_dataframes, ignore_index=True)
                whole_dataset = pd.concat([whole_dataset, i.loc[list(valid_indices)]], ignore_index=True)
            whole_dataset['n'] = whole_dataset['n'].astype('float64')
            whole_dataset['finished'] = whole_dataset['finished'].astype('int')
    else:
        if special_for_each == True:
            for i in final:
                i['n'] = i['n'].astype('float64')
                i['finished'] = i['finished'].astype('int')
            whole_dataset = final
        else:
            for i in final:
                whole_dataset = pd.concat([whole_dataset, i], ignore_index=True)
            whole_dataset['n'] = whole_dataset['n'].astype('float64')
            whole_dataset['finished'] = whole_dataset['finished'].astype('int')
        
    
    if minn == 1:
        return whole_dataset, data_one
    else:
        return whole_dataset

# from starting dataset, sample num_samples won and lost games, use other games to create train dataset, create test dataset for won and for lost games
# create datasets using first n_min clicks
# filter out games using max path and min path arguments
def create_dataset_and_samples(n_min, start_dataset, num_samples, max_path_length = None, min_path_length = None, special_for_each = False, node_stats_df = None, embeddings_df = None, pair_data = None, max_train_set = n_min):
    dataset = start_dataset.copy()
    if max_path_length is not None:
        if min_path_length is not None:
            true_games = dataset[(dataset['path_length'] > (n_min+1)) & (dataset['finished'] == True) & (dataset['path_length'] <= max_path_length)  & (dataset['path_length'] >= min_path_length)]
            false_games = dataset[(dataset['path_length'] > (n_min+1)) & (dataset['finished'] == False) & (dataset['path_length'] <= max_path_length)  & (dataset['path_length'] >= min_path_length)]
        else:
            true_games = dataset[(dataset['path_length'] > (n_min+1)) & (dataset['finished'] == True) & (dataset['path_length'] <= max_path_length)]
            false_games = dataset[(dataset['path_length'] > (n_min+1)) & (dataset['finished'] == False) & (dataset['path_length'] <= max_path_length)]
        
    else:
        if min_path_length is not None:
            true_games = dataset[(dataset['path_length'] > (n_min+1)) & (dataset['finished'] == True)  & (dataset['path_length'] >= min_path_length)]
            false_games = dataset[(dataset['path_length'] > (n_min+1)) & (dataset['finished'] == False) & (dataset['path_length'] >= min_path_length)]
        else:
            true_games = dataset[(dataset['path_length'] > (n_min+1)) & (dataset['finished'] == True)]
            false_games = dataset[(dataset['path_length'] > (n_min+1)) & (dataset['finished'] == False)]

    sampled_true_games = true_games.sample(n=num_samples, random_state=7)
    sampled_false_games = false_games.sample(n=num_samples, random_state=7)
    dataset = dataset.drop(sampled_true_games.index)
    dataset = dataset.drop(sampled_false_games.index)
    if special_for_each == True:
        max_train_set = n_min
    new_dataset, one_dataset = create_from_dataset(dataset, None, 1, max_train_set, special_for_each = special_for_each, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data)
    sampled_true_games_m, sampled_true_one = create_from_dataset(sampled_true_games, None, 1, n_min, balanced = True, special_for_each = special_for_each, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data)
    sampled_false_games_m, sampled_false_one = create_from_dataset(sampled_false_games, None, 1, n_min, balanced = True, special_for_each = special_for_each, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data)
    return new_dataset, sampled_true_games_m, sampled_false_games_m, one_dataset, sampled_true_one, sampled_false_one


# generate predictions from test sets and calculate accuracy using one model (trained on whole dataset) or models (one for each specific dataset)
def joint_predict_and_print(true_games, false_games, num_samples, n_min, threshold, models, true_one, false_one, model_1, graphs = False, special_for_each = False):
    y_pred_true = model_1.predict_proba(true_one.drop(columns = ['finished', 'n']))
    y_pred_false = model_1.predict_proba(false_one.drop(columns = ['finished', 'n']))
    true_games_stats = (y_pred_true > threshold).astype(int)
    false_games_stats = (y_pred_false > threshold).astype(int)
    temp_true = y_pred_true.copy()
    temp_false = y_pred_false.copy()
    
    real_true = true_one['finished'].values
    real_false = false_one['finished'].values

    
    matches_true = (true_games_stats == real_true)
    matches_false = (false_games_stats == real_false)
    
    accuracies_true_one = matches_true.mean()
    accuracies_false_one = matches_false.mean()

    if special_for_each == False:

        y_pred_true = model.predict_proba(true_games.drop(columns = ['finished', 'n']))
        y_pred_false = model.predict_proba(false_games.drop(columns = ['finished', 'n']))
        
        real_true = true_games['finished'].values.reshape((n_min-1), num_samples - ((num_samples*(n_min-1)-len(y_pred_true))//(n_min-1))).T
        real_false = false_games['finished'].values.reshape((n_min-1), num_samples - ((num_samples*(n_min-1)-len(y_pred_false))//(n_min-1))).T

       
    else:
        num_models = len(models)  
        
        y_pred_true_list = []
        y_pred_false_list = []
        
        for i, model in enumerate(models):
            true_chunk = true_games[i]
            false_chunk = false_games[i]
            
            if not true_chunk.empty:
                y_pred_true = model.predict_proba(true_chunk.drop(columns=['finished', 'n']))
                y_pred_true_list.append(y_pred_true)
            
            if not false_chunk.empty:
                y_pred_false = model.predict_proba(false_chunk.drop(columns=['finished', 'n']))
                y_pred_false_list.append(y_pred_false)
        
        y_pred_true = np.vstack(y_pred_true_list).reshape(-1)
        y_pred_false = np.vstack(y_pred_false_list).reshape(-1)
        real_true  = np.vstack([df['finished'].values for df in true_games]).T
        real_false  = np.vstack([df['finished'].values for df in false_games]).T
        
    print(y_pred_true.shape, y_pred_false.shape)
    y_pred_true_reshaped = y_pred_true.reshape((n_min-1), num_samples - ((num_samples*(n_min-1)-len(y_pred_true))//(n_min-1))).T
    y_pred_false_reshaped = y_pred_false.reshape((n_min-1), num_samples - ((num_samples*(n_min-1)-len(y_pred_false))//(n_min-1))).T


    true_games_stats = (y_pred_true_reshaped > threshold).astype(int)
    false_games_stats = (y_pred_false_reshaped > threshold).astype(int)

    
    matches_true = (true_games_stats == real_true)
    matches_false = (false_games_stats == real_false)
    
    accuracies_true = matches_true.mean(axis=0)
    accuracies_false = matches_false.mean(axis=0)

    print("Accuracies for clicks in finished games:")
    print(f"Accuracy at click 1: {accuracies_true_one:.4f}")
    for idx, acc in enumerate(accuracies_true):
        print(f"Accuracy at index {idx+2}: {acc:.4f}")
    
    print("Accuracies for clicks in unfinished games:")
    print(f"Accuracy at click 1: {accuracies_false_one:.4f}")
    for idx, acc in enumerate(accuracies_false):
        print(f"Accuracy at click {idx+2}: {acc:.4f}")

    return np.concatenate((temp_true, y_pred_true)), np.concatenate((temp_false, y_pred_false))

    