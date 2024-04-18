from __future__ import print_function

import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from wsi_datasets.classification import WSI_Classification_Dataset
from data.cls_default import label_dicts
from utils.process_args import define_args

# pytorch imports
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import pandas as pd
import numpy as np
import json

def merge_dict(main_dict, new_dict):
    """
    Merges two dictionaries

    Args:
        main_dict (dict): The dictionary to which values will be appended.
        new_dict (dict): The dictionary from which values will be retrieved and added 
            to main_dict.

    Returns:
        dict: The merged dictionary with values appended or added as necessary.
    """
    for k, v in new_dict.items():
        if k not in main_dict:
            main_dict[k] = []
        main_dict[k].append(v)
    return main_dict

def main(args):
    """
    Main function

    Args:
        args (Namespace): Experiment arguments

    Returns:
        None
    """
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    folds = np.arange(start, end)
    dataset_kwargs = dict(data_source = args.data_source, 
                         label_map = args.label_map,
                         label_map_race = args.label_map_race,
                         target_col = args.target_col,
                         study = args.task,
                         )
    
    all_val_results = {}

    for fold_idx in folds:
        seed_torch(args.seed)
        splits = read_splits(args.split_dir, fold_idx=fold_idx)
        print('successfully read splits for: ', list(splits.keys()))

        datasets =  build_datasets(splits, **dataset_kwargs)
        val_dump, val_results = train(datasets, fold_idx, args)
        all_val_results = merge_dict(all_val_results, val_results)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(fold_idx))
        save_pkl(filename, val_dump)

    final_dict = {'folds': folds}
    final_dict.update({k + '_val':v for k,v in all_val_results.items()})
    final_df = pd.DataFrame(final_dict)

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

def seed_torch(seed=7):
    """
    Set deterministic seed

    Args:
        seed (int): The seed to be set
        
    Returns:
        None
    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_splits(split_dir, fold_idx = None):
    """
    Read the splits dataframes

    Args:
        split_dir (str): Path to splits directory
        fold_idx (int): Read splits for which fold. If None, then not added to name

    Returns:
        splits: dictionary with splits
    """
    splits = {}
    for split in ['train', 'val']:
        if fold_idx is not None:
            split_path = os.path.join(split_dir, split + f'_{fold_idx}.csv')
        else:
            split_path = os.path.join(split_dir, split + '.csv')
        if os.path.isfile(split_path):
            df = pd.read_csv(split_path)
            splits[split] = df
    return splits

def build_datasets(splits, **kwargs):
    """
    Build dataloaders for each split

    Args:
        splits (dict): dictionary with splits dfs

    Returns:
        splits: dataloaders for splits
    """
    for k, df in splits.items():
        dataset = WSI_Classification_Dataset(df, **kwargs)
        
        if k == "train":
            if args.mitigation == "imp_weighing":
                weights = dataset.weights
                dataloader = DataLoader(dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), num_workers=2)
            else:
                # no mitigation strategy
                dataloader = DataLoader(dataset, batch_size=1, shuffle = True, num_workers=2)
        else:
            dataloader = DataLoader(dataset, batch_size=1, shuffle = False, num_workers=2)

        splits[k] = dataloader
        print(f'split: {k}, n: {len(dataloader)}')
    return splits

if __name__ == "__main__":
    
    args = define_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.label_map = label_dicts[args.task]
    
    args.label_map_race = label_dicts["race_map"]
    args.n_classes_race = len(set(list(args.label_map_race.values())))
    print('label map race: ', args.label_map_race)
    
    print('task: ', args.task)
    print('label map: ', args.label_map)
    args.n_classes = len(set(list(args.label_map.values()))) 
    
    print('split_dir: ', args.split_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n################### Settings ###################")
    for key, val in vars(args).items():
        print("{}:  {}".format(key, val))       
    with open(os.path.join(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
        
    results = main(args)
    print("finished!")
 
