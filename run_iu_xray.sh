python main.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/updated_annotation.json \
--dataset_name iu_xray \
--max_seq_length 80 \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--save_dir ../autodl-tmp/results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--zeta 0.0 \
--method r2gen \
--eval_period 1 

# --method r2gen / pretrained \
# d_model: 1B: 2048 3B: 3072 
# --pretrained_model_path ../autodl-tmp \
# --pretrained_model_name meta-llama/Llama-3.2-1B \