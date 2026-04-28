# Counting
## convnext
python main.py --dataset_file Leaf --epochs 300 --batch_size 4 \
  --model_name convnext_small \
  --pretrained_path dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth

## vit
python main.py --dataset_file Leaf --epochs 300 --batch_size 4 \
  --model_name vit_small \
  --pretrained_path dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# Segmentation
## convnext
python main_segment.py --dataset_file Segment --epochs 300 --batch_size 4 \
  --model_name convnext_small \
  --pretrained_path dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth


## vit
python main_segment.py --dataset_file Segment --epochs 300 --batch_size 4 \
  --model_name vit_small \
  --pretrained_path dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# Classification
## convnext
python main_classify.py --dataset_file Classify --epochs 300 --batch_size 4 \
  --model_name vit_small \
  --pretrained_path dinov3_vits16_pretrain_lvd1689m-08c60483.pth
