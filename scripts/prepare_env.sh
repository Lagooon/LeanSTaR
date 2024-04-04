# ssh yikangshen@ibm-wtf -i ~/.ssh/ccc_id_rsa

# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.35.2
pip install fire
pip install datasets
pip install einops
pip install wandb
pip install sentencepiece

pip install peft==0.4.0
pip install bitsandbytes==0.41.0
pip install deepspeed==0.9.3
pip install "fschat[model_worker,webui]"
pip install openai anthropic GPUtil typing-inspect==0.8.0 typing_extensions==4.5.0 ray==2.7.1
pip install vllm torch==2.2.2

# (L1072) vi /home/ray/workspace/.local/lib/python3.9/site-packages/peft/tuners/lora.py

# git config --global credential.helper store
# huggingface-cli login
# python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ScalableMath/MATH', repo_type='dataset', local_dir='$DATA_DIR')"
# python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ScalableMath/GSM8K', repo_type='dataset', local_dir='$DATA_DIR')"

# python -u data_scripts/prepare_easy_math.py \
#     --metamath_math_data $DATA_DIR/MATH_all_data_155588.jsonl \
#     --metamath_gsm8k_data $DATA_DIR/GSM8K_all_data_126568.jsonl \
#     --output_dir $DATA_DIR

# Navigate to the desired directory
# cd /workspace/zhiqings/output3/data/

# # Copy each file to the current directory
# cp $(readlink -f GSM8K_all_data_126568.jsonl) .
# cp $(readlink -f MATH_all_data_155588.jsonl) .
