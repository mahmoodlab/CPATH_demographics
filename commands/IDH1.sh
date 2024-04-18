#!/bin/bash

#########################################################################################################################################
########################################### Summary of key arguments for train and test #################################################
#########################################################################################################################################

# key files
# main.py : training and validation script on Ebrains 
# eval.py : independent test script on TCGA-GBMLGG

# key arguments
# --data_source : path to the WSI features stored as .pt files 
# --in_dim : dimension of features
# --exp_code : name of your experiment
# --target_col : name of the label column in metadata csv
# --task : what task are you running
# --k : number of folds
# --max_epochs : maximum number of epochs
# --early_stopping : perform early stopping
# --es_min_epochs : minimum epochs after which early stopping is started
# --es_patience : patience for early stopping
# --split_dir : where are the splits stored
# --model_type : name of model
# --no_inst_cluster : when provided with clam_sb, ABMIL model type is selected
# --results_dir : folder to store results in
# --ckpt_path : path to folder containing the checkpoint

# for a detailed summary of all arguments, refer to utils/process_args.py


#########################################################################################################################################
############################################### IDH1 mutation prediction + ResNet + ABMIL ###############################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features_resnet \
--in_dim 1024 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--max_epochs 20 \
--early_stopping \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type clam_sb \
--no_inst_cluster \
--results_dir train_results_folder

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="train_results_folder/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features_resnet \
--in_dim 1024 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type clam_sb \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder

#########################################################################################################################################
############################################# IDH1 mutation prediction + CTransPath + ABMIL #############################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features_ctranspath \
--in_dim 768 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--max_epochs 20 \
--early_stopping \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type clam_sb \
--no_inst_cluster \
--results_dir train_results_folder

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="train_results_folder/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features_ctranspath \
--in_dim 768 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type clam_sb \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder

#########################################################################################################################################
################################################ IDH1 mutation prediction + UNI + ABMIL #################################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features_uni \
--in_dim 1024 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--max_epochs 20 \
--early_stopping \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type clam_sb \
--no_inst_cluster \
--results_dir train_results_folder

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="train_results_folder/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features_uni \
--in_dim 1024 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type clam_sb \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder

#########################################################################################################################################
############################################### IDH1 mutation prediction + ResNet + CLAM ################################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features_resnet \
--in_dim 1024 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--max_epochs 20 \
--early_stopping \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type clam_sb \
--results_dir train_results_folder

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="train_results_folder/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features_resnet \
--in_dim 1024 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type clam_sb \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder

#########################################################################################################################################
############################################# IDH1 mutation prediction + CTransPath + CLAM ##############################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features_ctranspath \
--in_dim 768 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--max_epochs 20 \
--early_stopping \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type clam_sb \
--results_dir train_results_folder

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="train_results_folder/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features_ctranspath \
--in_dim 768 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type clam_sb \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder

#########################################################################################################################################
################################################ IDH1 mutation prediction + UNI + CLAM ##################################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features_uni \
--in_dim 1024 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--max_epochs 20 \
--early_stopping \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type clam_sb \
--results_dir train_results_folder

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="train_results_folder/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features_uni \
--in_dim 1024 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type clam_sb \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder

#########################################################################################################################################
############################################## IDH1 mutation prediction + ResNet + TransMIL #############################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features_resnet \
--in_dim 1024 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--lr 0.0002 \
--max_epochs 20 \
--early_stopping \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type transmil \
--no_inst_cluster \
--results_dir train_results_folder

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="train_results_folder/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features_resnet \
--in_dim 1024 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type transmil \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder

#########################################################################################################################################
############################################ IDH1 mutation prediction + CTransPath + TransMIL ###########################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features_ctranspath \
--in_dim 768 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--lr 0.0002 \
--max_epochs 20 \
--early_stopping \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type transmil \
--no_inst_cluster \
--results_dir train_results_folder

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="train_results_folder/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features_ctranspath \
--in_dim 768 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type transmil \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder

#########################################################################################################################################
############################################### IDH1 mutation prediction + UNI + TransMIL ###############################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features_uni \
--in_dim 1024 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--lr 0.0002 \
--max_epochs 20 \
--early_stopping \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type transmil \
--no_inst_cluster \
--results_dir train_results_folder

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="train_results_folder/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features_uni \
--in_dim 1024 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type transmil \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder