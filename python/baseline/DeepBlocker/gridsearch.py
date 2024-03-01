import time
import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
from itertools import product
from gridsearch_utils import (
    to_path,
    values_given,
    get_multiples,
    necessary_dfs_supplied,
    clear_json_file,
    purge_id_column,
    get_deepblocker_candidates,
    update_workflow_statistics,
    statistics_to_dataframe,
    gt_to_df,
    iteration_normalized
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        dest='config_name',
                        type=str,
                        default="gt_experiments",
                        help='Name of the experiments config file within script-configs folder (without .json extension)')
    parser.add_argument('--dataset',
                        dest='dataset',
                        type=str,
                        default="d1",
                        help='Which dataset should the gridsearch be conducted for')
    args = parser.parse_args()
    
    #-EDIT-START-#
    # parameters native to the Deepblocker Workflow
    # don't edit, unless new parameters were added to the Workflow
    EXECUTION_PATH = '~/baseline_pyjedai/Embeddings4ER/python/baseline/DeepBlocker/'
    VALID_WORKFLOW_PARAMETERS = ["number_of_nearest_neighbors"]
    # path of the configuration file
    CONFIG_FILE_PATH = to_path(EXECUTION_PATH + 'grid_config/' + args.config_name + '.json')
    # which configuration from the json file should be used in current experiment  
    EXPERIMENT_NAME = args.config_name + '_' + args.dataset
    # path at which the results will be stored within a json file
    RESULTS_STORE_PATH = to_path(EXECUTION_PATH + 'results/' + EXPERIMENT_NAME + '.csv')
    # path at which the top workflows for specified argument values are stored
    BEST_WORKFLOWS_STORE_PATH = to_path(RESULTS_STORE_PATH + '_best.json')
    # results should be stored in the predefined path
    STORE_RESULTS = True
    # AUC calculation and ROC visualization after execution
    VISUALIZE_RESULTS = True
    # workflow arguments and execution info should be printed in terminal once executed
    PRINT_WORKFLOWS = True
    # identifier column names for source and target datasets
    D1_ID = 'id'
    D2_ID = 'id'  
    #-EDIT-END-#          
                                    
    config = config[args.dataset]
    if(not necessary_dfs_supplied(config)):
        raise ValueError("Different number of source, target dataset and ground truth paths!")

    datasets_info = list(zip(config['source_dataset_path'], config['target_dataset_path'], config['ground_truth_path']))
    iterations = config['iterations'][0] if(values_given(config, 'iterations')) else 1
    execution_count : int = 0
    
    workflows_dataframe_columns = ['budget','dataset','total_candidates','total_emissions','time','name','auc','recall','tp_indices']
    workflows_dataframe_columns = VALID_WORKFLOW_PARAMETERS + VALID_WORKFLOW_PARAMETERS
    workflows_dataframe = pd.DataFrame(columns=workflows_dataframe_columns)
    
    if(STORE_RESULTS):
        clear_json_file(path=JSON_STORE_PATH)

    for id, dataset_info in enumerate(datasets_info):
        dataset_id = id + 1
        d1_path, d2_path, gt_path = dataset_info
        dataset_name = config['dataset_name'][id] if(values_given(config, 'dataset_name') and len(config['dataset_name']) > id) else ("D" + str(dataset_id))
        
        sep = config['separator'][id] if values_given(config, 'separator') else '|'
        d1 : pd.DataFrame = pd.read_csv(to_path(d1_path), sep=sep, engine='python', na_filter=False).astype(str)
        d2 : pd.DataFrame = pd.read_csv(to_path(d2_path), sep=sep, engine='python', na_filter=False).astype(str)
        gt : pd.DataFrame = pd.read_csv(to_path(gt_path), sep=sep, engine='python')
        gt.columns = ['ltable_id', 'rtable_id']
        duplicate_of : dict = gt_to_df(ground_truth=gt)
        true_positives_number : int = len(gt)

        workflow_config : dict = {k: v for k, v in config.items() if(values_given(config, k) and k in VALID_WORKFLOW_PARAMETERS)}
        workflow_config['budget'] = config['budget'] if values_given(config, 'budget') else get_multiples(true_positives_number, 10)
        parameter_names : list = workflow_config.keys() 
        argument_combinations : list = product(*(workflow_config.values()))    
        total_workflows : int = len(argument_combinations) * len(datasets_info) * iterations
    
        for argument_combination in argument_combinations:
            workflow_arguments = dict(zip(parameter_names, argument_combination))
               
            workflow_statistics = defaultdict(float) 
            workflow_statistics['number_of_nearest_neighbors'] = workflow_arguments['number_of_nearest_neighbors']
            workflow_statistics['budget'] = workflow_arguments['budget']
            workflow_statistics['dataset'] = dataset_name
                    
            for iteration in range(iterations):
                execution_count += 1
                print(f"#### WORKFLOW {execution_count}/{total_workflows} ####")
                start_time = time.time()
                
                candidates : pd.Dataframe = get_deepblocker_candidates(source_dataset=d1,
                                                                       target_dataset=d2,
                                                                       nearest_neighbors=workflow_arguments['number_of_nearest_neighbors'])
                update_workflow_statistics(statistics=workflow_statistics,
                                           candidates=candidates,
                                           iterations=iterations,
                                           duplicate_of=duplicate_of)                        
                workflow_statistics['time'] += ((time.time() - start_time) / iterations)

            dataframe.append(workflow_statistics, ignore_index=True)    
    if(STORE_RESULTS):
        workflows_dataframe.to_csv(RESULTS_STORE_PATH, index=False)
    
    
    
                                      
