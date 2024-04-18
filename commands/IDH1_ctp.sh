#########################################################################################################################################
###################################### IDH1 mutation prediction + CTransPath + ABMIL (base config) ######################################
#########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source /path/to/train/features \
--in_dim 768 \
--exp_code experiment_name \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--k 20 \
--max_epochs 20 \
--es_min_epochs 10 \
--es_patience 5 \
--split_dir splits_MonteCarlo/Ebrains_IDH1Mutation/ \
--model_type clam_sb \
--no_inst_cluster \
--results_dir train_results_folder \
--early_stopping

EXP_CODE="experiment_name_{seed}"
CKPT_PATH="results_reproduce/$EXP_CODE"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source /path/to/test/features \
--in_dim 768 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col idh_status \
--task Ebrains_IDH1Mutation \
--model_type clam_sb \
--split_dir splits_MonteCarlo/TCGA_GBMLGG/ \
--results_dir test_results_folder