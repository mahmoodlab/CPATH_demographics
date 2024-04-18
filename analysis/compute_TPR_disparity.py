import os
from os.path import join as j_

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from tqdm import tqdm
import pandas as pd

def calc_metrics_binary(y_label, y_prob, cutoff=None, return_pred=False):
    """
    Computing (almost all) binary calculation metrics.
    
    Args:
        y_label (np.array): (n,)-dim np.array containing ground-truth predictions.
        y_prob (np.array): (n,)-dim np.array containing probability scores (for y=1).
        cutoff (int): Whether to use a Yolan's J cutoff (calculated from a model)
        return_pred (np.array): (n,)-dim np.array containing predictions using Yolan's J.
    Return:
        results (list): List of binary classification metrics.
    """
    ### AUC
    auc = roc_auc_score(y_label, y_prob)
    
    ### Yolans J
    if cutoff == None:
        fpr, tpr, thresholds = roc_curve(y_label, y_prob)
        J = tpr - fpr
        cutoff = thresholds[np.argmax(J)]
    y_pred = np.array(y_prob > cutoff).astype(int)
    
    ### Classification Report
    out = classification_report(y_label, y_pred, output_dict=True, zero_division=0)
    if return_pred:
        return y_pred, [auc, cutoff, out['accuracy'],
            out['0.0']['precision'], out['0.0']['recall'], out['0.0']['f1-score'], out['0.0']['support'],
            out['1.0']['precision'], out['1.0']['recall'], out['1.0']['f1-score'], out['1.0']['support'],
            out['macro avg']['precision'], out['macro avg']['recall'], out['macro avg']['f1-score'],
            out['weighted avg']['precision'], out['weighted avg']['recall'], out['weighted avg']['f1-score'],
           ]
    else:
        return [auc, cutoff, out['accuracy'],
                out['0.0']['precision'], out['0.0']['recall'], out['0.0']['f1-score'], out['0.0']['support'],
                out['1.0']['precision'], out['1.0']['recall'], out['1.0']['f1-score'], out['1.0']['support'],
                out['macro avg']['precision'], out['macro avg']['recall'], out['macro avg']['f1-score'],
                out['weighted avg']['precision'], out['weighted avg']['recall'], out['weighted avg']['f1-score'],
               ]

def join_dfs(results_df, test_df):
    """
    Join two DataFrames based on slide_id.

    Args:
        results_df (DataFrame): DataFrame containing results.
        test_df (DataFrame): DataFrame containing test data.

    Returns:
        DataFrame: Joined DataFrame.
    """
    results_df = results_df.reset_index()
    test_df = test_df.reset_index()
    
    Y_map = dict(zip(results_df["slide_id"], results_df["Y"]))
    p_0_map = dict(zip(results_df["slide_id"], results_df["p_0"]))
    p_1_map = dict(zip(results_df["slide_id"], results_df["p_1"]))
    
    test_df["Y"] = test_df["slide_id"].apply(lambda x : Y_map.get(x, None))
    test_df["p_0"] = test_df["slide_id"].apply(lambda x : p_0_map.get(x, None))
    test_df["p_1"] = test_df["slide_id"].apply(lambda x : p_1_map.get(x, None))
    test_df.dropna(inplace=True)
    test_df.set_index("slide_id", inplace=True)
    
    return test_df

def get_cv_metrics(eval_path, test_df=None, col=None, label=None,
                   val_metrics=None, seed=False):
    """
    Computes Cross-Validated Classification Metrics for race-stratified / overall test population.
    
    Args:
        eval_path (str): Path to 'fold_{i}.csv' where i in {1...10}, which contains predicted probabiltiy scores for each label.
        test_df (DataFrame): DataFrame containing slide_ids with matched patient-level / slide-level information (e.g. - oncotree_code, race)
        col (str, Optional): Column in test_df (containing categorical values) to subset the predictions by. 
        label (str, Optional): Label to subset the test_df by.
        val_metrics (DataFrame): DataFrame that contains the Yolan's J for each fold.
        Seed (None or int): Bootstrap seed for whether or not to resample the dataframe (used in bootstrap for loop).
        
    Return:
        cv_metrics (DataFrame): DataFrame containing classification metrics for each fold.
    """
    cv_metrics = []
    for i in range(NUM_FOLDS):

        if os.path.isfile(os.path.join(eval_path, 's_%d_checkpoint_results.pkl' % i)):
            results_df = pd_read_pickle(j_(eval_path, 's_%d_checkpoint_results.pkl' % i))  
        else:
            continue      
        
        if label is not None:
            results_df = join_dfs(results_df=results_df, test_df=test_df)
            results_df.dropna(inplace=True)
            results_df = results_df[results_df[col].str.contains(label)]
        
        if seed is not None:
            
            bootstrap = results_df.sample(n=results_df.shape[0], replace=True, random_state=seed).copy()
            collision = 0
            ### In case resampled df contains predictions of only one value.
            
            collision = 1
            while collision:
                
                bootstrap = results_df.sample(n=results_df.shape[0], replace=True, random_state=seed+1000+collision)
                y_label = np.array(bootstrap['Y'])
                y_prob = np.array(bootstrap['p_1'])
                
                ### Test for Collision (Errors when y_label or y_pred are all of one class)
                fpr, tpr, thresholds = roc_curve(y_label, y_prob)
                J = tpr - fpr
                cutoff = thresholds[np.argmax(J)]
                y_pred = np.array(y_prob > cutoff).astype(int)
                
                if (y_label.sum() == 0) or (y_label.sum() == bootstrap.shape[0]) or (y_pred.sum() == 0) or (y_pred.sum() == bootstrap.shape[0]):
                    collision += 1
                else:
                    collision = 0
            
        else:
            y_label = np.array(results_df['Y'])
            y_prob = np.array(results_df['p_1'])
        
        if val_metrics is None:
            cv_metrics.append(calc_metrics_binary(y_label, y_prob, None))
        else:
            cv_metrics.append(calc_metrics_binary(y_label, y_prob, val_metrics['Cutoff'][i]))
           
    
    cv_metrics = pd.DataFrame(cv_metrics)
    cv_metrics.columns = METRIC_COLS
    cv_metrics.index.name = 'Folds'
    return cv_metrics

def get_metrics_stratified_boot(test_df, val_metrics=None, eval_path='', num_boot=10):
    """
    Get stratified and overall metrics
    
    Args:
        test_df (DataFrame): DataFrame containing slide_ids with matched patient-level / slide-level information (e.g. - oncotree_code, race)
        val_metrics (DataFrame): DataFrame that contains the Yolan's J for each fold
        eval_path (str): root eval path
        num_boot (int): Number of bootsrap iterations
        
    Return:
        TPRs (DataFrame): TPRs by race
    """
    
    race_cv_metrics_by_label = {}
    race_cv_TPRs_by_label = {}
    race_boot_metrics_by_label = {}
    
    ### 1. Race-Stratified Evaluation
    pbar_labels = tqdm(LABELS, position=0, leave=True)
    for label in pbar_labels:
        pbar_labels.set_description('Bootstrap - Race %s' % label)
        
        ### 1.1. Mean of the Cross-Validated Metrics (for each race group)
        race_cv_metrics = get_cv_metrics(eval_path, test_df=test_df, col=COL, 
                                    label=label, val_metrics=val_metrics, seed=None)
        race_cv_metrics = race_cv_metrics.drop(['Cutoff'], axis=1)
        race_cv_metrics_by_label[label] = race_cv_metrics
        race_cv_TPRs_by_label[label] = race_cv_metrics[['Y=0 R', 'Y=1 R']] # specifically recall for TPR disparity calculation
        
        ### 1.2. 95% Confidence Interval of the Cross-Validated Metrics (for each race group)
        race_boot_metrics = []
        for seed in tqdm(range(num_boot), total=num_boot, position=0, leave=True):
            race_boot_metrics.append(get_cv_metrics(eval_path, test_df=test_df, col=COL, 
                                                    label=label, val_metrics=val_metrics, seed=seed).mean())
        race_boot_metrics = pd.concat(race_boot_metrics, axis=1).T
        race_boot_metrics.index.name = 'Runs'
        race_boot_metrics = race_boot_metrics.drop(['Cutoff'], axis=1)
        race_boot_metrics_by_label[label] = race_boot_metrics
        
    ### 2.1 Overall Evaluation
    overall_metrics = get_cv_metrics(eval_path, test_df=None, col=None, 
                                     label=None, val_metrics=val_metrics, seed=None)
    overall_metrics = overall_metrics.drop(['Cutoff'], axis=1).mean()
    
    ### 2.2 95% Confidence Interval of Overall Population Metrics
    overall_boot_metrics = []
    for seed in tqdm(range(num_boot), total=num_boot, position=0, leave=True):
        overall_boot_metrics.append(get_cv_metrics(eval_path, test_df=None, col=None,
                                                   label=None, val_metrics=val_metrics, seed=seed).mean())
    overall_boot_metrics = pd.concat(overall_boot_metrics, axis=1).T
    overall_boot_metrics.index.name = 'Runs'
    overall_boot_metrics = overall_boot_metrics.drop(['Cutoff'], axis=1)
        
    ### 3. Summary
    metrics_ci = pd.DataFrame([race_cv_metrics_by_label[label].mean().map('{:.3f}'.format).astype(str) + ' ' + \
                               race_boot_metrics_by_label[label].apply(CI_pm)
                               for label in LABELS])
    metrics_ci = pd.concat([metrics_ci, 
                            pd.DataFrame(overall_metrics.map('{:.3f}'.format).astype(str) + ' ' + \
                            overall_boot_metrics.apply(CI_pm)).T
                            ])
    metrics_ci.index = LABELS + ['Overall']
        
    return race_cv_TPRs_by_label, metrics_ci

def CI_pm(data):
    """
    Calculate the CI.

    Args:
        data (np.array): Input data.

    Returns:
        str: Formatted string representing the confidence interval.
    """
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(data, p)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(data, p)
    return '(%0.3f, %0.3f)' % (lower, upper)

def stratify_summarize(eval_path, val_metrics): 
    """
    Summarize stratified evaluation results.

    Args:
        eval_path (str): Path to evaluation data.
        val_metrics (list): List of validation metrics.

    Returns:
        df_tpr_disparity (DataFrame): TPR disparities
        metrics_ci (DataFrame): stratified metrics with CI
    """
    
    # load the test_df and make sure labels are in right format
    test_df = pd.read_csv(PATH_TO_TEST_DF, index_col=2)
    test_df[COL] = test_df[COL].astype(str)
    
    # Calculate Results
    TPRs, metrics_ci = get_metrics_stratified_boot(
        test_df, 
        val_metrics=val_metrics, 
        eval_path=eval_path, 
        num_boot=NUM_BOOT
    )
    
    ### get TPR Disparity in clean format
    df_tpr_disparity = get_TPR_disparity(TPRs)
    
    return df_tpr_disparity, metrics_ci

def get_TPR_disparity(TPRs):
    """Calculate TPR disparity.

    Args:
        TPRs (dict): Dictionary containing true positive rates.

    Returns:
        DataFrame: DataFrame containing TPR disparity.
    """
    
    y0_df = pd.concat([pd.concat([pd.Series([label]*NUM_FOLDS), TPRs[label]['Y=0 R']], axis=1) for label in LABELS], axis=0)
    y1_df = pd.concat([pd.concat([pd.Series([label]*NUM_FOLDS), TPRs[label]['Y=1 R']], axis=1) for label in LABELS], axis=0)
    y0_df.columns = ['Race', 'TPR']
    y0_df['TPR'] = y0_df['TPR'] - y0_df['TPR'].median() # according to user's needs, can be replaced by mean or any other version of center
    y0_df.insert(0, 'Class', Y_NAMES[0])
    y1_df.columns = ['Race', 'TPR']
    y1_df.insert(0, 'Class', Y_NAMES[1])
    y1_df['TPR'] = y1_df['TPR'] - y1_df['TPR'].median() # according to user's needs, can be replaced by mean or any other version of center
    combined_df = pd.concat([y0_df, y1_df])
    
    return combined_df


def pd_read_pickle(path):
    """
    Read data from a pickle file and convert it into a DataFrame.

    Args:
        path (str): Path to the pickle file.

    Returns:
        DataFrame: DataFrame containing the loaded data.
    """
    results = pd.read_pickle(path)
    y_label = np.array(results['labels']).astype(float)
    
    results_df = pd.DataFrame({'Y': y_label, 
                               'p_0': np.array(results['probs'][:,0]),
                               'p_1': np.array(results['probs'][:,1])})
    results_df.index = results['slide_ids']
    results_df.index.name = 'slide_id'
    return results_df

def acquire_cutoffs(model):
    """
    Acquire cutoffs from validation folds

    Args:
        model (str): Name of model.

    Returns:
        DataFrame: Validation metrics and cutoff
    """
    eval_path_train = j_(DATAROOT_TRAIN, model)
    metrics = []
    y_label_all, y_prob_all = [], []
        
    for i in range(NUM_FOLDS):
        if os.path.isfile(os.path.join(eval_path_train, 'split_%d_results.pkl' % i)):
            results_df = pd_read_pickle(j_(eval_path_train, 'split_%d_results.pkl' % i))
        else:
            continue

        y_label = np.array(results_df['Y'])
        y_prob = np.array(results_df['p_1'])
        y_label_all.append(y_label)
        y_prob_all.append(y_prob)
        metrics.append(calc_metrics_binary(y_label, y_prob, cutoff=None))

    y_label_all = np.hstack(y_label_all)
    y_prob_all = np.hstack(y_prob_all)
    metrics.append(calc_metrics_binary(y_label_all, y_prob_all, cutoff=None))
    metrics = pd.DataFrame(metrics)
    metrics.columns = METRIC_COLS
    
    return metrics

if __name__ == "__main__":

    ALL_MODELS = ["model name"]
    DATAROOT_TRAIN = "path/to/train/results"
    DATAROOT_EVAL = 'path/to/independent/test/results'
    PATH_TO_TEST_DF = "path/to/test/df"
    NUM_FOLDS = 20
    METRIC_COLS = ['AUC', 'Cutoff', 'Acc',
        'Y=0 P', 'Y=0 R', 'Y=0 F1', 'Y=0 Support',
        'Y=1 P', 'Y=1 R', 'Y=1 F1', 'Y=1 Support',
        'Macro Avg P', 'Macro Avg R', 'Macro Avg F1',
        'Weight Avg P', 'Weight Avg R', 'Weight Avg F1'
        ]
    
    # example for IDH1 mutation prediction
    STUDY = "IDH1"
    Y_NAMES = ['idhwt', 'idhmut']
    LABELS = ["W", "A", "B"]
    COL = "race"
    NUM_BOOT = 1000
    LIM = [-0.7, 0.3]
    STEP_SIZE =  0.2
    BOUNDS = 0.001

    for model in ALL_MODELS:
        
        print()
        print(f"Going for model {model}...")
        print()
        
        metrics = acquire_cutoffs(model)    

        print("Acquired the cutoffs...")
        print()

        eval_path = j_(DATAROOT_EVAL, model)

        TPR_disparity, metrics_ci = stratify_summarize(
            eval_path=eval_path, 
            val_metrics=metrics.copy(),
        )