from __future__ import print_function, division
import os
import torch

from torch.utils.data import Dataset
import h5py

class WSI_Classification_Dataset(Dataset):
    """
    Dataset class for WSI classification tasks.

    Args:
        df (pandas.DataFrame): The dataframe containing the metadata.
        data_source (str): Path to the directory containing the features.
        target_transform (composed transforms, optional): Composed transforms to apply.
        index_col (str, optional): Name of the column containing slide IDs.
        target_col (str, optional): Name of the column containing labels.
        use_h5 (bool, optional): Whether to features are stored as HDF5 files.
        label_map (dict, optional): Mapping of original labels to encoded values.
        label_map_race (dict, optional): Mapping of races to encoded values.
        study (str, optional): Name of the study.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        get_ids(ids): Returns slide IDs for the specified indices.
        get_labels(ids): Returns labels for the specified indices.
        get_caseID(ids): Returns case IDs for the specified indices.
        get_race(ids): Returns race information for the specified indices.
        __getitem__(idx): Returns WSI features, label, case id, and race for specified index

    """
    def __init__(self, 
                 df, 
                 data_source, 
                 target_transform = None,
                 index_col = 'slide_id',
                 target_col = 'label', 
                 use_h5 = False,
                 label_map = None,
                 label_map_race = None,
                 study = None
                 ):
        
        self.label_map = label_map
        self.label_map_race = label_map_race
        self.data_source = data_source
        self.index_col = index_col
        self.target_col = target_col
        self.target_transform = target_transform
        self.data = df
        self.data.fillna('N', inplace=True)
        self.use_h5 = use_h5
        self.study = study

        if "race" in self.data.columns:
            self._prep_instance_weights()
        else:
            self.weights = None

    def _prep_instance_weights(self):

        race_counts = dict(self.data["race"].value_counts()) # counts per race
        N = sum(dict(self.data["race"].value_counts()).values()) # total count
        weight_per_race = {}
        
        # assign a weight to race inversely proportional to its count
        for race in race_counts:
            race_count = race_counts[race]
            weight = N / race_count
            weight_per_race[race] = weight

        self.weights = [0] * int(N)  

        for idx in range(N):   
            y = self.data.loc[idx, "race"]                 
            self.weights[idx] = weight_per_race[y]

        self.weights = torch.DoubleTensor(self.weights)


    def __len__(self):
        return len(self.data)

    def get_ids(self, ids):
        return self.data.loc[ids, self.index_col]

    def get_labels(self, ids):
        return self.data.loc[ids, self.target_col]
    
    def get_caseID(self, ids):
        return self.data.loc[ids, "case_id"]
    
    def get_race(self, ids):        
        if "race" in self.data.columns:
            return self.label_map_race[self.data.loc[ids, "race"]]
        else:
            return -1
    
    def __getitem__(self, idx):
        
        slide_id = self.get_ids(idx)
        label = self.get_labels(idx)
        case_id = self.get_caseID(idx)
        race = self.get_race(idx)

        if self.label_map is not None:
            label = self.label_map[label]
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.use_h5:
            feat_path = os.path.join(self.data_source, 'h5_files', slide_id + '.h5')
            with h5py.File(feat_path, 'r') as f:
                features = torch.from_numpy(f['features'][:])
        else:
            feat_path = os.path.join(self.data_source, 'pt_files', slide_id + '.pt')
            features = torch.load(feat_path)

        return {'img': features, 'label': label, "case_id": case_id, "race": race}