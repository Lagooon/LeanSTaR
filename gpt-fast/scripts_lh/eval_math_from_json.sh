module load cuda/11.8
# TMPDIR='/home/wliu/longhui/llms-all/alpaca-lora' pip install -r /home/wliu/longhui/llms-all/ScalableMath_sun-main/ScalableMath-1222/gpt-fast/re.txt
export PATH=/home/wliu/anaconda3/envs/openrlhf/bin:$PATH
SAVE_PATH='/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma-7b_metamath-hf'
python eval_math_vllm.py --model $SAVE_PATH --tensor_parallel_size 8 \
--data_file /home/wliu/longhui/llms-all/ScalableMath_sun-main/data/metamath_math_level1-3_20193_mapped.json \
--end 400
echo $SAVE_PATH

