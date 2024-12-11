python gen_report.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/updated_annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 1 \
--seed 9223 \
--resume ../autodl-tmp/results/iu_xray/current_checkpoint.pth \
--pretrained_model_path ../autodl-tmp

cd ./external/CheXbert/src

python label.py -d=../../../generated_reports.csv -o=../../../ -c=../chexbert.pth