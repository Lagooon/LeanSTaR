# ssh yikangshen@ibm-wtf -i ~/.ssh/ccc_id_rsa

DATA_DIR=/nobackup/users/yikangs/zhiqings/math

git config --global credential.helper store
huggingface-cli login
# python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ScalableMath/MATH', repo_type='dataset', local_dir='$DATA_DIR', local_dir_use_symlinks=False)"
# python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ScalableMath/GSM8K', repo_type='dataset', local_dir='$DATA_DIR', local_dir_use_symlinks=False)"

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ScalableMath/metamath_level1-3', repo_type='dataset', local_dir='$DATA_DIR', local_dir_use_symlinks=False)"

# python -u data_scripts/prepare_easy_math.py \
#     --metamath_math_data $DATA_DIR/MATH_all_data_155588.jsonl \
#     --metamath_gsm8k_data $DATA_DIR/GSM8K_all_data_126568.jsonl \
#     --output_dir $DATA_DIR \
#     --train_portion 0.5

# python -u data_scripts/post_process_math_train.py

# Navigate to the desired directory
# cd /workspace/zhiqings/output3/data/

# # Copy each file to the current directory
# cp $(readlink -f GSM8K_all_data_126568.jsonl) .
# cp $(readlink -f MATH_all_data_155588.jsonl) .
