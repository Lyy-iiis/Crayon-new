python main.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/updated_annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 1 \
--epochs 100 \
--save_dir ../autodl-tmp/results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--zeta 2000.0 \
--pretrained_model_path ./llama