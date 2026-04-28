import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .count_dataset import build as build_count
from .class_dataset import build as build_classify
from .detect_dataset import build as build_detect

data_path = {
    'Leaf': './count_dataset',
    'SHA': './data',
    'Classify': './class_disease/classification_dataset',
    'Detect': './detect_dataset',
}

def build_dataset(image_set, args):
    if args.dataset_file in data_path:
        args.data_path = data_path[args.dataset_file]

    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    if args.dataset_file == 'Leaf':
        return build_count(image_set, args)
    if args.dataset_file == 'Classify':
        return build_classify(image_set, args)
    if args.dataset_file == 'Detect':
        return build_detect(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
