import numpy as np
import pandas as pd
from utils.data_processing import *
from utils.graph_processing import *
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

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
    index_before = starting_games.index
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

    

    pairss = []
    for c in cols:
        pairss.append([c, 'target'])

    if pair_data is None:
        pair_data = load_pair_data()
    #starting_games = add_pair_data(starting_games, pair_data, pairs =[[f'{n}_click', 'target']], names = [f"{n}"], data = ['shortest_path_length', #'shortest_path_count'])
    starting_games = add_pair_data(starting_games, pair_data, pairs =pairss, names = cols, data = ['shortest_path_length', 'shortest_path_count'])


    def calc_zjb(row, cols):
        zjb_factor = 0
        for i, c in enumerate(cols):
            if i == (len(cols) -1):
                continue
            zjb_factor = zjb_factor + (row[f"shortest_path_length_{cols[i+1]}"] - row[f"shortest_path_length_{cols[i]}"] -1)
        return zjb_factor / (len(cols)-1)
        
    starting_games['zjb_factor'] = starting_games.apply(lambda row: calc_zjb(row, cols), axis = 1)

    #print("now")
    #print(starting_games.columns)
    
    cols.remove("source")
    cols.insert(0, "source")
    #print(cols)
    
    for i in range(len(cols)-1):
        starting_games.drop(columns = [f"shortest_path_length_{cols[i]}", f"shortest_path_count_{cols[i]}"], inplace = True)
    


    #print(starting_games.columns)
    index_after = starting_games.index

    if ind is not None:
        removed_indices = index_before.difference(index_after)
        ind.difference_update(removed_indices)

    
    
    for i in range(1, n):
        cols.append(f"pagerank_{i}_click")

    starting_games.rename(columns = {f"pagerank_{n}_click": 'pagerank_n_click', f"cosine_sim_{n}_click_target": 'cosine_sim_n_click_target', f'shortest_path_length_{n}_click': 'shortest_path_length_n', f'shortest_path_count_{n}_click': 'shortest_path_count_n'}, inplace = True)

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
def create_from_dataset(datagames, whole_dataset, minn, maxx, filter = True, balanced = False, special_for_each = False, node_stats_df = None, embeddings_df = None, pair_data = None, all_clicks = False):
    if whole_dataset == None:
        whole_dataset = pd.DataFrame(columns=['duration','num_back','pagerank_n_click','pagerank_target', 'max_pagerank', 'cosine_sim_n_click_target','cos_diff', 'zjb_factor' ,'shortest_path_length_n', 'shortest_path_count_n','finished','n'])
    valid_indices = set(datagames.index)
    removed_indices = set()
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
    if all_clicks == False:
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
    else:
        
        for n in range (2, 100000):
            print(f"Adding for click {n}...")
            #print(f"Prediction based on first {n} clicks: ")
            n_games = datagames.copy()
            if filter == True:
                n_games = n_games[n_games['path_length'] > n]
                condition = (n_games['path_length'] == (n+1)) & (n_games['finished'] == True)
                n_games = n_games[~condition] # removes games won in n clicks exactly
                if n_games.shape[0] == 0:
                    break
        
            #print("Number of games before building dataset: ", n_games.shape)
            
            valid_indices_before = set(n_games.index)
            valid_indices = set(n_games.index)
            dataset, features = build_dataset(n_games, n, valid_indices if balanced else None, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data)
            
            removed_indices_temp = valid_indices_before - valid_indices
            removed_indices.update(removed_indices_temp)
            
            features.append("n")
            final.append(dataset)
            
            print("----------------------------------------------------")
            print()
    if balanced:
        if minn == 1:
            if all_clicks == False:
                data_one = data_one.loc[list(valid_indices)]
            else:
                data_one = data_one.loc[~data_one.index.isin(removed_indices)].reset_index()
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
                if all_clicks == True:
                    filtered_rows = i.loc[~i.index.isin(removed_indices)]
                    whole_dataset = pd.concat([whole_dataset, filtered_rows], ignore_index=True)
                    print(f"for i {i} whole dataset is now {whole_dataset.shape[0]}")
                else:
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
        
    if (all_clicks == True) & (balanced == True):
        whole_dataset = (whole_dataset, list(removed_indices))
        
    if minn == 1:
        return whole_dataset, data_one
    else:
        return whole_dataset

# from starting dataset, sample num_samples won and lost games, use other games to create train dataset, create test dataset for won and for lost games
# create datasets using first n_min clicks
# filter out games using max path and min path arguments
def create_dataset_and_samples(n_min, start_dataset, num_samples, max_path_length = None, min_path_length = None, special_for_each = False, node_stats_df = None, embeddings_df = None, pair_data = None, max_train_set = None, all_clicks = False):
    if max_train_set is None:
        max_train_set = n_min
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

    sampled_true_games = true_games.sample(n=num_samples, random_state=8)
    sampled_false_games = false_games.sample(n=num_samples, random_state=8)
    dataset = dataset.drop(sampled_true_games.index)
    dataset = dataset.drop(sampled_false_games.index)
    if special_for_each == True:
        max_train_set = n_min
    new_dataset, one_dataset = create_from_dataset(dataset, None, 1, max_train_set, special_for_each = special_for_each, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data)
    sampled_true_games_m, sampled_true_one = create_from_dataset(sampled_true_games, None, 1, n_min, balanced = True, special_for_each = special_for_each, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data, all_clicks = all_clicks)
    sampled_false_games_m, sampled_false_one = create_from_dataset(sampled_false_games, None, 1, n_min, balanced = True, special_for_each = special_for_each, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data, all_clicks = all_clicks)
    if all_clicks == False:
        return new_dataset, sampled_true_games_m, sampled_false_games_m, one_dataset, sampled_true_one, sampled_false_one
    else:
        ret_true = sampled_true_games['path_length']
        ret_true = ret_true.loc[~ret_true.index.isin(sampled_true_games_m[1])]
        ret_false = sampled_false_games['path_length']
        print(f"Removed indices are {ret_false.loc[sampled_false_games_m[1]]}")
        ret_false = ret_false.loc[~ret_false.index.isin(sampled_false_games_m[1])]
        #ret_false = sampled_false_games['path_length'].loc[sampled_false_games_m[1]]
        sampled_true_games_m = sampled_true_games_m[0]
        sampled_false_games_m = sampled_false_games_m[0]
        return new_dataset, sampled_true_games_m, sampled_false_games_m, one_dataset, sampled_true_one, sampled_false_one, ret_true, ret_false


# generate predictions from test sets and calculate accuracy using one model (trained on whole dataset) or models (one for each specific dataset)
def joint_predict_and_print(true_games, false_games, num_samples, n_min, threshold, model, true_one, false_one, model_1, graphs = False, special_for_each = False):
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
        num_models = len(model)  
        
        y_pred_true_list = []
        y_pred_false_list = []
        
        for i, model in enumerate(model):
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

    f1_true = f1_score(real_true.flatten(), true_games_stats.flatten(), average='macro')  
    f1_false = f1_score(real_false.flatten(), false_games_stats.flatten(), average='macro')
    
    print("Accuracies and F1 Scores for clicks in finished games:")
    print(f"Accuracy at click 1: {accuracies_true_one:.4f}")
    for idx, acc in enumerate(accuracies_true):
        print(f"Accuracy at index {idx + 2}: {acc:.4f}")
    print(f"Overall F1 score: {f1_true:.4f}")
    
    print("\nAccuracies and F1 Scores for clicks in unfinished games:")
    print(f"Accuracy at click 1: {accuracies_false_one:.4f}")
    for idx, acc in enumerate(accuracies_false):
        print(f"Accuracy at index {idx + 2}: {acc:.4f}")
    print(f"Overall F1 score: {f1_false:.4f}")

    return np.concatenate((temp_true, y_pred_true)), np.concatenate((temp_false, y_pred_false))


def predict_different_ns(true_games, false_games, counts_true, counts_false, threshold, model, true_one, false_one, model_1):
    y_pred_true = model_1.predict_proba(true_one.drop(columns = ['finished', 'n']))
    y_pred_false = model_1.predict_proba(false_one.drop(columns = ['finished', 'n']))
    true_games_stats = (y_pred_true > threshold).astype(int)
    false_games_stats = (y_pred_false > threshold).astype(int)
    temp_true = y_pred_true.copy()
    temp_false = y_pred_false.copy()

    real_true = true_one['finished'].values
    real_false = false_one['finished'].values

    #counts_true = counts_true - 3
    #counts_false = counts_false - 2
    
    matches_true = (true_games_stats == real_true)
    matches_false = (false_games_stats == real_false)
    
    accuracies_true_one = matches_true.mean()
    accuracies_false_one = matches_false.mean()

    y_pred_true = model.predict_proba(true_games.drop(columns = ['finished', 'n']))
    y_pred_false = model.predict_proba(false_games.drop(columns = ['finished', 'n']))
    
    real_true = true_games['finished'].values
    real_false = false_games['finished'].values

    index_sequence_true = []
    indices = list(range(len(counts_true))) 
    counts_true = counts_true.copy()
    print(np.sum(counts_true))
    start_sum =  np.sum(counts_true)
    print("new")
    while len(index_sequence_true) < start_sum:
        for i in indices:
            if counts_true[i] > 0:
                index_sequence_true.append(i)
                counts_true[i] -= 1  
                if len(index_sequence_true) == sum(counts_true): 
                    break
    true_seq = np.array(index_sequence_true)

    print("True sequence is of length", len(true_seq), "while true games is of len", true_games.shape)

    
    index_sequence_false = []
    indices = list(range(len(counts_false))) 
    counts_false = counts_false.copy()
    start_sum =  np.sum(counts_false)
    
    while len(index_sequence_false) < start_sum:
        for i in indices:
            if counts_false[i] > 0:
                index_sequence_false.append(i)
                counts_false[i] -= 1  
                if len(index_sequence_false) == sum(counts_false): 
                    break
    false_seq = np.array(index_sequence_false)

    print("False sequence is of length", len(false_seq), "while false games is of len", false_games.shape)

    real_true = real_true[true_seq]
    real_false = real_false[false_seq]
        

    true_games_stats = (y_pred_true > threshold).astype(int)
    false_games_stats = (y_pred_false > threshold).astype(int)

    
    matches_true = (true_games_stats == real_true)
    matches_false = (false_games_stats == real_false)
    
    accuracies_true = matches_true.mean(axis=0)
    accuracies_false = matches_false.mean(axis=0)

    #f1_true = f1_score(real_true, true_games_stats, average='macro')  
    #f1_false = f1_score(real_false, false_games_stats, average='macro')
    
    f1_true = f1_score(real_true.flatten(), true_games_stats.flatten(), average='macro')  
    f1_false = f1_score(real_false.flatten(), false_games_stats.flatten(), average='macro')
    
    print("Accuracies and F1 Scores for clicks in finished games:")
    print(f"Accuracy at click 1: {accuracies_true_one:.4f}")
    print(f"Accuracy at others: {accuracies_true:.4f}")
    print(f"Overall F1 score: {f1_true:.4f}")
    
    print("\nAccuracies and F1 Scores for clicks in unfinished games:")
    print(f"Accuracy at click 1: {accuracies_false_one:.4f}")
    print(f"Accuracy at others : {accuracies_false:.4f}")
    print(f"Overall F1 score: {f1_false:.4f}")

    return np.concatenate((temp_true, y_pred_true)), np.concatenate((temp_false, y_pred_false)), true_seq, false_seq

def build_inherent_dataset(data):
    data_inh = data[data['n'] == 2]
    data_inh = data_inh[['pagerank_target', 'finished']]
    features_inh = ['pagerank_target']
    return data_inh, features_inh

def predict_inh_for_true_false(true_games, false_games, model_inh):
    true_games_inh = true_games[true_games['n'] == 2][['pagerank_target']]
    false_games_inh = false_games[false_games['n'] == 2][['pagerank_target']]
    pred_inh_true = model_inh.predict_proba(true_games_inh)
    pred_inh_false = model_inh.predict_proba(false_games_inh)
    return pred_inh_true, pred_inh_false

def gen_final_from_inh_clicks(true_pred, false_pred, n_min, pred_inh_true, pred_inh_false):
    x_true = len(pred_inh_true)
    x_false = len(pred_inh_false)
    true_pred_reshaped = true_pred.reshape(6, x_true).T  # Shape (x, 6)
    final_true_pred = np.insert(true_pred_reshaped, 0, pred_inh_true, axis=1)
    false_pred_reshaped = false_pred.reshape(6, x_false).T  # Shape (x, 6)
    final_false_pred = np.insert(false_pred_reshaped, 0, pred_inh_false, axis=1)
    return final_true_pred, final_false_pred


def gen_final_for_n_clicks(true_pred, false_pred, pred_inh_true, pred_inh_false, counts_true, counts_false):
    index_sequence_true = []
    indices = list(range(len(counts_true))) 
    counts_true = counts_true.copy()
    print(np.sum(counts_true))
    start_sum =  np.sum(counts_true)
    print("new")
    while len(index_sequence_true) < start_sum:
        for i in indices:
            if counts_true[i] > 0:
                index_sequence_true.append(i)
                counts_true[i] -= 1  
                if len(index_sequence_true) == sum(counts_true): 
                    break
    true_seq = np.array(index_sequence_true)

    #print("True sequence is of length", len(true_seq), "while true games is of len", true_games.shape)

    
    index_sequence_false = []
    indices = list(range(len(counts_false))) 
    counts_false = counts_false.copy()
    start_sum =  np.sum(counts_false)
    
    while len(index_sequence_false) < start_sum:
        for i in indices:
            if counts_false[i] > 0:
                index_sequence_false.append(i)
                counts_false[i] -= 1  
                if len(index_sequence_false) == sum(counts_false): 
                    break
    false_seq = np.array(index_sequence_false)
    
    unique_values = np.unique(true_seq)
    grouped_arrays_true = []
    
    for value in unique_values:
        indices = np.where(true_seq == value)[0]  
        grouped_arrays_true.append(true_pred[indices])

    result_true = [np.insert(row, 0, val) for row, val in zip(grouped_arrays_true, pred_inh_true)]


    unique_values = np.unique(false_seq)
    grouped_arrays_false = []
    
    for value in unique_values:
        indices = np.where(false_seq == value)[0]  
        grouped_arrays_false.append(false_pred[indices]) 
    result_false = [np.insert(row, 0, val) for row, val in zip(grouped_arrays_false, pred_inh_false)]
        
    #result_false = np.array([np.insert(row, 0, val) for row, val in zip(np.array(grouped_arrays_false), pred_inh_false)])
    
    return result_true, result_false

def plot_mean_of_games(final_true_pred, final_false_pred):
    mean_true_pred = np.mean(final_true_pred, axis=0)
    mean_false_pred = np.mean(final_false_pred, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_true_pred, label="True Array (Mean)", linestyle='-', marker='o', color='green')
    plt.plot(mean_false_pred, label="False Array (Mean)", linestyle='--', marker='x', color='red')
    
    plt.xlabel("Columns")
    plt.ylabel("Average Values")
    plt.title("Mean of Predictions for True and False Arrays")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))  # Adjust legend position
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def assign_labels(array, is_true=True, oneForAll = True):
    labels = []  
    if oneForAll == False:
        truths = is_true.copy()
    
    for i in range(len(array)):
        if oneForAll == False:
            is_true = truths[i]
        row = array[i]
        if is_true: # won games
            if np.max(row) - np.min(row) <= 0.3:
                labels.append(1)
            elif row[0] < 0.45 and np.mean(np.diff(row)) > 0:
                labels.append(2)
            elif np.mean(np.diff(row)) > 0:
                labels.append(3)
            else:
                labels.append(4)
            
        else: # lost games
            if np.max(row) - np.min(row) <= 0.3:
                labels.append(1)
            elif row[0] > 0.45 and np.mean(np.diff(row)) < 0:
                labels.append(2)
            elif np.mean(np.diff(row)) < 0:
                labels.append(3)
            else:
                labels.append(4)
    
    return np.array(labels)


def plot_average_for_labels(array, labels, is_true=True, changeable = False, max_elements = None):
    if is_true:
        label_names = {1: "Stationary start Win", 2: "Clutch Win", 3: "Good start Win", 4: "How did you win this?"}
        color = "green"
    else:
        label_names = {1: "Stationary start Loss", 2: "Unexpected Loss", 3: "Bad start Loss", 4: "How did you lose this?"}
        color = "red"
        
    if changeable == False:
        for label in [1, 2, 3, 4]:
            label_indices = np.where(labels == label)[0]
            
            # Calculate the average for each label
            if len(label_indices) > 0:
                avg_array = np.mean(array[label_indices], axis=0)
                
                plt.figure(figsize=(10, 6))
                plt.plot(avg_array, marker='o', label=f"Average for {label_names[label]}", color=color)
                
                plt.title(f"Average of All Games for Label {label} ({label_names[label]})")
                plt.xlabel("Columns")
                plt.ylabel("Values")
                plt.legend(loc='upper right')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
    else:
        for label in [1, 2, 3, 4]:
            
            label_indices = np.where(labels == label)[0]
            selected_rows = [array[idx] for idx in label_indices]  # Select rows for the label
            
            if len(selected_rows) == 0:
                continue
            
            max_length = max(len(row) for row in selected_rows)

            if max_elements is not None:
                max_length = min(max_length, max_elements)
            
           
            column_sums = np.zeros(max_length)
            column_counts = np.zeros(max_length)
            
           
            for row in selected_rows:
                length = min(len(row), max_length)  # Truncate row to max_length
                column_sums[:length] += row[:length]
                #column_sums[:length] += row
                column_counts[:length] += 1  
            
            
            avg_array = column_sums / column_counts
            
            
            plt.figure(figsize=(10, 6))
            plt.plot(avg_array, marker='o', label=f"Average for {label_names[label]}")
            
            plt.title(f"Average of All Games for Label {label} ({label_names[label]})")
            plt.xlabel("Columns")
            plt.ylabel("Values")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()




def plot_random_games_for_labels(array, labels, is_true=True):
    if is_true:
        label_names = {1: "Stationary start Win", 2: "Clutch Win", 3: "Good start Win", 4: "How did you win this?"}
        color = "green"
    else:
        label_names = {1: "Stationary start Loss", 2: "Unexpected Loss", 3: "Bad start Loss", 4: "How did you lose this?"}
        color = "red"

    for label in [1, 2, 3, 4]:  # Now including label 4
        label_indices = np.where(labels == label)[0]
        
        # Randomly select 3 indices for this label
        if len(label_indices) > 0:
            random_indices = np.random.choice(label_indices, size=min(3, len(label_indices)), replace=False)

            for i, idx in enumerate(random_indices):
                plt.figure(figsize=(10, 6))
                plt.plot(array[idx], marker='o', label=f"Game {i+1}", color=color)
                
                plt.title(f"Random Game {i+1} for Label {label} ({label_names[label]})")
                plt.xlabel("Columns")
                plt.ylabel("Values")
                plt.legend(loc='upper right')
                plt.grid(True)
                plt.tight_layout()
                plt.show()


def check_overlaps(group):
    group = group.copy()
    group["overlap"] = (
        group["timestamp"] < group["timestamp"].shift(1) + pd.to_timedelta(group["duration"].shift(1), unit="s")
    )
    return group
    
def get_non_overlapping_games(games):
    overlap = games.copy().sort_values(by=["hashIP", "timestamp"]).reset_index(drop=True)
    overlap = overlap.groupby("hashIP", group_keys=False).apply(check_overlaps)
    players_with_overlaps = overlap.loc[overlap["overlap"], "hashIP"].unique()
    all_games_with_overlaps = overlap[overlap["hashIP"].isin(players_with_overlaps)]
    columns_to_display = ["hashIP", "path", "finished", "timestamp", "duration", "type_end", "overlap"]
    non_overlap_rows = games.copy()[~games["hashIP"].isin(players_with_overlaps)]
    
    # for rows where hashIP IS in players_with_overlaps, keep only the row with the smallest timestamp
    overlap_rows = (
        games.copy()[games["hashIP"].isin(players_with_overlaps)]
        .sort_values(by=["hashIP", "timestamp"])
        .groupby("hashIP", group_keys=False)
        .head(1)
    )
    
    filtered_games = pd.concat([non_overlap_rows, overlap_rows]).sort_index()
    filtered_games = filtered_games.reset_index(drop=True)
    return filtered_games

# we assume games are filtered by timestamp
def calculate_elos_for_player(games_clicks, labels, win, C= 40):
    """
    Calculate Elo ratings for a player based on their game performance.
    
    Parameters:
    - games_clicks: 2D numpy array (each row represents a game, first column is inherent difficulty)
    - labels: 1D numpy array of game labels (1 to 4)
    - win: 1D numpy array indicating if the game was won (1) or lost (0)
    - K: Constant controlling the magnitude of Elo adjustments
    
    Returns:
    - elos: 1D numpy array of player's Elo after each game
    """
    d = 10
    T = 30
    num_games = len(games_clicks)
    elos = np.zeros(num_games+1)  
    player_rating = 1000  # initial player rating
    elos[0] = 1000
    score_game = []
    num_until_now = 0
    
    for i in range(num_games):
        # Calculate game score
        #inherent_difficulty = games_clicks[i, 0] # for partial games
        inherent_difficulty = games_clicks[i][0] 
        game_score = 1000 + inherent_difficulty * 2000  # game rating
        score_game.append(int(game_score))

        K = C / (1 + np.exp((num_until_now - T) / d))
        
        expected_performance = 1 / (1 + 10 ** ((game_score - player_rating) / 400))
      
        if win[i] == 1:  # Game was won
            if labels[i] == 1:  # Stationary start win
                actual_performance = 0.6
            elif labels[i] == 2:  # Clutch win
                actual_performance = 1
            elif labels[i] == 3:  # Good start win
                actual_performance = 0.8
            elif labels[i] == 4:  # How did you win this
                actual_performance = 0.5
            else:
                print(labels[i])
        else:  # Game was lost
            if labels[i] == 1:  # Stationary game loss
                actual_performance = -0.1
            elif labels[i] == 2:  # Unexpected loss
                actual_performance = -1
            elif labels[i] == 3:  # Bad start loss
                actual_performance = -0.5
            elif labels[i] == 4:  # How did you lose this
                actual_performance = -0.3
            else:
                print(labels[i])
        
        new_rating = player_rating + K * (actual_performance - expected_performance)
        elos[i+1] = int(new_rating)
        
        player_rating = new_rating
        num_until_now += 1
    
    return elos, score_game


# send new games !
def build_dataset_for_players(train_dataset, filtered_dataset, player_ips, n_min = 4, node_stats_df = None, embeddings_df = None, pair_data = None, all_clicks = False):
    player_datasets = []
    for h in player_ips:
        player_datasets.append(filtered_dataset.copy()[(filtered_dataset['path_length'] > 5) & (filtered_dataset['hashIP'] == h)])
    train_dataset = train_dataset.copy()[~train_dataset['hashIP'].isin(player_ips)]
    new_dataset, one_dataset = create_from_dataset(train_dataset, None, 1, n_min, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data)
    player_games_sets, player_one_sets = [], []
    for d in player_datasets:
        pg, pgo = create_from_dataset(d, None, 1, n_min, node_stats_df = node_stats_df, embeddings_df = embeddings_df, pair_data = pair_data, all_clicks = all_clicks)
        if all_clicks == True:
            player_games_sets.append((pg, d[['path_length', 'finished']]))
        else:
            player_games_sets.append(pg)
        player_one_sets.append(pgo)
    return new_dataset, one_dataset, player_games_sets, player_one_sets


def calc_clicks_and_labels(model_inh, model_1, model, player_games_sets, player_one_sets, all_clicks = False):
    clicks = []
    labels = []
    wins = []

    for i in range(len(player_games_sets)):

        if all_clicks == False:
            proba_2 = model.predict_proba(player_games_sets[i].drop(columns = ['finished', 'n']))
            proba_1 = model_1.predict_proba(player_one_sets[i].drop(columns = ['finished', 'n']))
            inh_d = player_games_sets[i][player_games_sets[i]['n'] == 2]
            proba_0 = model_inh.predict_proba(inh_d[['pagerank_target']])
           
            num_clicks = int(player_games_sets[i]['n'].max())
            
            num_games = player_games_sets[i].shape[0] // (num_clicks-1)
            
            proba_2_reshaped = proba_2.reshape(num_games, num_clicks - 1)
            result = np.hstack([
                proba_0.reshape(-1, 1), 
                proba_1.reshape(-1, 1),  
                proba_2_reshaped         
            ])
            won = player_games_sets[i][player_games_sets[i]['n'] == 2]['finished'].values
        else:
            pg, info = player_games_sets[i]
            print(player_one_sets[i].columns)
            proba_2 = model.predict_proba(pg.drop(columns = ['finished', 'n']))
            proba_1 = model_1.predict_proba(player_one_sets[i].drop(columns = ['finished', 'n']))
            inh_d = pg[pg['n'] == 2]
            proba_0 = model_inh.predict_proba(inh_d[['pagerank_target']])
            #print(player_games_sets[i]['n'].max())
            info['click_count'] = info.apply(lambda row: (row['path_length'] - 3) if (row['finished'] == True) else (row['path_length'] - 2), axis = 1)
            cc = info['click_count'].values
            index_sequence = []
            indices = list(range(len(cc))) 
            cc = cc.copy()
            start_sum =  np.sum(cc)
            
            while len(index_sequence) < start_sum:
                for i in indices:
                    if cc[i] > 0:
                        index_sequence.append(i)
                        cc[i] -= 1  
                        if len(index_sequence) == start_sum: 
                            break
            seq = np.array(index_sequence)
            
            unique_values = np.unique(seq)
            grouped_arrays = []
            
            for value in unique_values:
                indices = np.where(seq == value)[0]  
                grouped_arrays.append(proba_2[indices])
        
            result = [np.insert(row, 0, val) for row, val in zip(grouped_arrays, proba_1)]
            result = [np.insert(row, 0, val) for row, val in zip(grouped_arrays, proba_0)]
            won = pg[pg['n'] == 2]['finished'].values
            
            
        clicks.append(result)

        
        labels.append(assign_labels(result, is_true=won, oneForAll = False)) 
        wins.append(won)

    return clicks, labels, wins
    

def plot_player_games(clicks, wins, elos=None):
    """
    Plots games for each player. When Elo ratings are provided, plots each game separately 
    with Elo ratings and game scores included in the plot titles.
    
    Args:
        clicks (list of 2D arrays): Each element is a 2D array where rows represent games for a player.
        wins (list of 1D arrays): Each element is a 1D binary array indicating win (1) or loss (0) for each game.
        elos (list of tuples, optional): If provided, each element is a tuple (player_elos, game_scores).
            - player_elos: 1D array of the player's Elo scores before each game.
            - game_scores: 1D array of the game scores (Elo of the game) for each game.
    """
    num_players = len(clicks)

    for player_idx in range(num_players):
        player_games = clicks[player_idx]  # 2d array: rows are games
        player_wins = wins[player_idx]     # 1d array: 1 = win, 0 = loss
        
        if elos is None:
            plt.figure(figsize=(10, 6))
            for game_idx, game in enumerate(player_games):
                if player_wins[game_idx] == 1:  # Won game
                    plt.plot(game, marker='o', color='green', label='Win' if game_idx == 0 else "") 
                else:  # Lost game
                    plt.plot(game, marker='o', color='red', label='Loss' if game_idx == 0 else "")
            
            plt.title(f"Player {player_idx + 1}: Games")
            plt.xlabel("Clicks")
            plt.ylabel("Values")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            player_elos, game_scores = elos[player_idx]  # Tuple of 1D arrays
            num_games = len(player_games)
            
            for game_idx, game in enumerate(player_games):
                plt.figure(figsize=(10, 6))
                
                color = 'green' if player_wins[game_idx] == 1 else 'red'
                plt.plot(game, marker='o', color=color)
                
                elo_before = player_elos[game_idx]
                elo_after = player_elos[game_idx + 1] if game_idx + 1 < len(player_elos) else "N/A"
                game_score = game_scores[game_idx]
                
                plt.title(
                    f"Player {player_idx + 1}: Game {game_idx + 1}\n"
                    f"Elo Before: {elo_before}, Elo After: {elo_after}\n"
                    f"Game Score: {game_score}"
                )
                
                plt.xlabel("Clicks")
                plt.ylabel("Values")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

        


    