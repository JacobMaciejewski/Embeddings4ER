import os
import sys
import json
import pandas as pd
from collections import defaultdict
import time
from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils
from utils import cases


def to_path(path : str):
    return os.path.expanduser(path)

def values_given(configuration: dict, parameter: str) -> bool:
    """Values for requested parameters have been supplied by the user in the configuration file

    Args:
        configuration (dict): Configuration File
        parameter (str): Requested parameter name

    Returns:
        bool: Values for requested parameter supplied
    """
    return (parameter in configuration) and (isinstance(configuration[parameter], list)) and (len(configuration[parameter]) > 0)

def get_multiples(num : int, n : int) -> list:
    """Returns a list of multiples of the requested number up to n * number

    Args:
        num (int): Number
        n (int): Multiplier

    Returns:
        list: Multiplies of num up to n * num 
    """
    multiples = []
    for i in range(1, n+1):
        multiples.append(num * i)
    return multiples

def necessary_dfs_supplied(configuration : dict) -> bool:
    """Configuration file contains values for source, target and ground truth dataframes

    Args:
        configuration (dict): Configuration file

    Raises:
        ValueError: Zero values supplied for one or more paths

    Returns:
        bool: Source, target and ground truth dataframes paths supplied within configuration dict
    """
    for path in ['source_dataset_path', 'target_dataset_path', 'ground_truth_path']:
        if(not values_given(configuration, path)):
            raise ValueError(f"{path}: No values given")
        
    return len(configuration['source_dataset_path']) == len(configuration['target_dataset_path']) == len(configuration['ground_truth_path'])

def clear_json_file(path : str):
    if os.path.exists(path):
        if os.path.getsize(path) > 0:
            open(path, 'w').close()

def purge_id_column(columns : list):
    """Return column names without the identifier column

    Args:
        columns (list): List of column names

    Returns:
        list: List of column names except the identifier column
    """
    non_id_columns : list = []
    for column in columns:
        if(column != 'id'):
            non_id_columns.append(column)
    
    return non_id_columns

def get_deepblocker_candidates(source_dataset : pd.Dataframe,
                               target_dataset : pd.Dataframe,
                               nearest_neighbors : int = 5,
                               columns_to_block : list = ["aggregate value"]
                               ) -> pd.Dataframe:
    """Applies DeepBlocker matching and retrieves the nearest neighbors 
       for each entity of the source dataset and their corresponding scores
       in a dataframe

    Args:
        columns (list): List of column names

    Returns:
        list: List of column names except the identifier column
    """        
    tuple_embedding_model = AutoEncoderTupleEmbedding()
    topK_vector_pairing_model = ExactTopKVectorPairing(K=nearest_neighbors)
    db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
    candidate_pairs = db.block_datasets(source_dataset, target_dataset, columns_to_block)
    return candidate_pairs

def iteration_normalized(value, iterations):
    return float(value) / float(iterations)

def update_workflow_statistics(statistics : dict,
                                candidates : pd.DataFrame,
                                ground_truth : pd.DataFrame,
                                iterations : int,
                                duplicate_of : dict
                               ) -> None:
    """Sorts the candidate pairs in descending score order globally.
       Iterates over those pairs within specified budget, 
       and updates the statistics (e.x. AUC score) for the given experiment

    Args:
        statistics (dict): Dictionary storing the statistics of the given experiment
        candidates (pd.DataFrame): Candidate pairs with their scores
        iterations (int): The number of times current workflow will be executed 
        duplicate_of (dict): Mapping from source dataset entity ID to target dataset true positive entity ID 
    """
    candidates : pd.DataFrame = candidates.sort_values(by='similarity', ascending=False)
    
    is_true_positive = candidates.apply(lambda row: row['rtable_id'] in duplicate_of[row['ltable_id']], axis=1)
    is_true_positive : list = is_true_positive.tolist()
    total_tps : int = len(ground_truth) 
    tps_found : int = 0
    total_emissions : int = 0
    recall_axis : list = []  
    tp_indices : list = []      
    statistics['total_candidates'] += iteration_normalized(value=min(len(is_true_positive), statistics[budget]),
                                                           iterations=iterations)

    for emission, tp_emission in enumerate(is_true_positive):
        if(emission >= budget or tps_found >= total_tps):
            total_emissions = emission 
            break
        if(tp_emission):
            recall = (tps_found + 1.0) / total_tps
            tps_found += 1
            tp_indices.append(emission)
        recall_axis.append(recall)
     
    statistics['total_emissions'] += iteration_normalized(value=total_emissions,
                                                          iterations=iterations)
    auc : float = sum(recall_axis) / (total_emissions + 1.0)     
    statistics['auc'] += iteration_normalized(value=auc, iterations=iterations) 
    statistics['recall'] += iteration_normalized(value=recall_axis[-1], iterations=iterations)
    statistics['tp_indices'] = tp_indices
    
def gt_to_df(ground_truth : pd.DataFrame):  
    duplicate_of = defaultdict(set)
    for _, row in ground_truth.iterrows():
        id1, id2 = (row[0], row[1])
        if id1 in duplicate_of: duplicate_of[id1].add(id2)
        else: duplicate_of[id1] = {id2}
    return duplicate_of