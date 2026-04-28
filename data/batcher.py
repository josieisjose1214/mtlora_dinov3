import torch
import torch.nn as nn
import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader,BatchSampler,Sampler
from torchvision import datasets,transforms
import numpy as np
import random
import glob
import xml.etree.ElementTree as ET

# load every single task dataset
# 1.disease classification
class ClassifyDisease(Dataset):
    def __init__(
        self,
        path,
        split='train',
        transform=None,
        task_id=0,
    ):
        self.root=path #./datasets/class_disease/classification_dataset
        self.split_dir= os.path.join(self.root, split) #train/val/test
        self.transform=transform
        self._task_id=task_id

        # 1. 查找所有类别名称 (子文件夹名)
        # os.listdir 遍历 split_dir 下的所有文件和文件夹，过滤出文件夹
        class_names = [d.name for d in os.scandir(self.split_dir) if d.is_dir()]
        self.class_to_idx = {name: i for i, name in enumerate(sorted(class_names))}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

        # 2. 收集所有图像文件的路径和对应的标签
        self.image_list = []
        self.label_list = []

        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(self.split_dir, class_name)
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.jfif', '*.webp'):
                for img_path in glob.glob(os.path.join(class_path, ext)):
                    #匹配类别文件夹下所有后缀为.jpg...的图片路径
                    self.image_list.append(img_path)
                    self.label_list.append(class_idx)


    def get_task_id(self):
        return self._task_id

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.label_list[idx]

        image= np.array(Image.open(img_path).convert(
            'RGB')).astype(float) #numpy数组形式的图像

        sample={"image": image, "text": label, "uid": idx}

        if self.transform:
            sample = self.transform(sample)
        
        return {
            "task": self._task_id, #分类任务的编号
            "sample": sample, #图像，类别号，图像的编号
        }

#2. detection dataset
class DetectWheat(Dataset):
    def __init__(
        self, 
        path,
        split='train',
        transform=None,
        task_id=1,
    ):
        self.root=path 
        #./datasets/class_disease/classification_dataset
        img_root=os.path.join(os.path.join(self.root,"images"),split) #train/val/test
        label_root=os.path.join(os.path.join(self.root,"labels"),split)
        self.transform=transform
        self._task_id=task_id

        self.image_list = []
        self.label_list = []
        imgs=sorted(os.listdir(img_root))

        for f in imgs:
            img_path = os.path.join(img_root, f)
            label_path = os.path.join(label_root, f.replace(".png", ".txt"))
            self.image_list.append(img_path)
            self.label_list.append(label_path)


    def get_task_id(self):
        return self._task_id

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label_path = self.label_list[idx]

        image=np.array(Image.open(img_path).convert('RGB')).astype(float)
        label=self._load_box(label_path) #np.array形式的标签，相对坐标

        sample={"image":image,"bbox":label,"uid":idx}

        if self.transform:
            sample = self.transform(sample)
        
        return {
            "task": self._task_id, #分类任务的编号
            "sample": sample, #图像，类别号，图像的编号
        }

    def _load_box(self,label_path):
        try:
            with open(label_path, "r") as f:
                lines = f.read().splitlines()
                if len(lines) == 0:
                    # 如果文件为空，返回空的2D数组 [0, 5]
                    return np.empty((0, 5), dtype=np.float32)
                
                # 解析每一行，每行应该是 "class x y w h" 格式
                bbox_list = []
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    parts = line.split()
                    if len(parts) >= 5:  # 确保至少有5个值
                        try:
                            bbox = [float(parts[i]) for i in range(5)]
                            # 验证bbox的有效性
                            class_id, x, y, w, h = bbox
                            
                            # 检查坐标范围（相对坐标应该在0-1范围内）
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                                # 如果坐标超出范围，可能是绝对坐标，尝试归一化（假设图像大小为448）
                                # 或者跳过这个无效的bbox
                                # 这里我们选择跳过，因为数据应该是相对坐标
                                continue
                            
                            # 检查类别索引（应该 >= 0）
                            if class_id < 0:
                                continue
                            
                            bbox_list.append(bbox)
                        except (ValueError, IndexError) as e:
                            # 如果解析失败，跳过这一行
                            continue
                    elif len(parts) > 0:
                        # 如果行不为空但格式不对，可能是格式错误
                        # 跳过这一行，不返回错误
                        continue
                
                if len(bbox_list) == 0:
                    # 如果没有有效的bbox，返回空数组
                    return np.empty((0, 5), dtype=np.float32)
                
                # 转换为numpy数组，确保是2D [N, 5]
                l = np.array(bbox_list, dtype=np.float32)
                
                # 确保是2D数组 [N, 5]
                if len(l.shape) == 1:
                    # 如果只有一行，reshape为 [1, 5]
                    if l.shape[0] == 5:
                        l = l.reshape(1, 5)
                    else:
                        # 如果长度不是5，可能是格式错误，返回空数组
                        return np.empty((0, 5), dtype=np.float32)
                elif len(l.shape) == 2:
                    # 确保第二维是5
                    if l.shape[1] != 5:
                        # 如果第二维不是5，尝试修复或返回空数组
                        if l.shape[0] > 0 and l.shape[1] > 5:
                            # 如果有多余的列，只取前5列
                            l = l[:, :5]
                        else:
                            return np.empty((0, 5), dtype=np.float32)
                
                #l=[class, x, y, w, h] 均为相对坐标
                return l
        except FileNotFoundError:
            # 如果文件不存在，返回空数组
            return np.empty((0, 5), dtype=np.float32)
        except Exception as e:
            # 其他错误，返回空数组（避免程序崩溃）
            # 在生产环境中，可以考虑记录错误日志
            return np.empty((0, 5), dtype=np.float32)

# 3. counting dataset (point annotations, PET format)
class CountWheat(Dataset):
    """
    计数任务数据集。目录结构（与 single_task/count_dataset.py 一致）：
    count_dataset/
      images/train/ xxx.jpg
      images/val/   xxx.jpg
      annotations/train/ xxx.xml
      annotations/val/   xxx.xml
    XML 中读取每个 object 的 bndbox，使用框中心作为计数点，格式为 [y, x]。
    """
    def __init__(
        self,
        path,
        split='train',
        transform=None,
        task_id=3,
    ):
        self.root = path
        img_dir = os.path.join(self.root, 'images', split)
        ann_dir = os.path.join(self.root, 'annotations', split)
        self.transform = transform
        self._task_id = task_id

        self.image_list = []
        self.ann_list = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.jfif', '*.webp', '*.JPG'):
            for img_path in sorted(glob.glob(os.path.join(img_dir, ext))):
                base = os.path.splitext(os.path.basename(img_path))[0]
                ann_path = os.path.join(ann_dir, base + '.xml')
                if os.path.isfile(ann_path):
                    self.image_list.append(img_path)
                    self.ann_list.append(ann_path)

    def get_task_id(self):
        return self._task_id

    def __len__(self):
        return len(self.image_list)

    def _load_points(self, ann_path):
        """Parse XML annotation and return points in [y, x] format."""
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
        except Exception:
            return np.empty((0, 2), dtype=np.float32)

        points = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            try:
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
            except Exception:
                continue
            cx = np.round((xmin + xmax) / 2.0).astype(np.int32)
            cy = np.round((ymin + ymax) / 2.0).astype(np.int32)
            points.append([cy, cx])  # [y, x]

        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def compute_density(points):
        if points.shape[0] <= 1:
            return np.array([999.0], dtype=np.float32)
        dist = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        density = np.min(dist, axis=1).mean()
        return np.array([density], dtype=np.float32)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        ann_path = self.ann_list[idx]
        image = np.array(Image.open(img_path).convert('RGB')).astype(float)
        points = self._load_points(ann_path)
        density = self.compute_density(points) if points.shape[0] > 0 else np.array([999.0], dtype=np.float32)
        sample = {"image": image, "points": points, "density": density, "uid": idx}
        if self.transform:
            sample = self.transform(sample)
        return {
            "task": self._task_id,
            "sample": sample,
        }


# 4. wheat segmentation
class SegmentWheat(Dataset):
    def __init__(
        self,
        path,
        split='train',
        transform=None,
        task_id=2,
    ):
        self.root=path #./datasets/class_disease/classification_dataset
        img_root=os.path.join(os.path.join(self.root,split),"images") #train/val/test
        label_root=os.path.join(os.path.join(self.root,split),"class_id")
        self.transform=transform
        self._task_id=task_id

        self.image_list = []
        self.label_list = []

        imgs=sorted(os.listdir(img_root))
        for f in imgs:
            img_path=os.path.join(img_root,f)
            label_path=os.path.join(label_root,f)

            self.image_list.append(img_path)
            self.label_list.append(label_path)

    def __len__(self):
        return len(self.image_list)

    def get_task_id(self):
        return self._task_id

    def __getitem__(self,idx):
        img_path=self.image_list[idx]
        label_path=self.label_list[idx]
        img=np.array(Image.open(img_path).convert("RGB")).astype(float)
        label=self._load_semseg(label_path)

        sample={"image":img, "semseg":label, "uid":idx}

        if self.transform is not None:
            sample=self.transform(sample)

        return {
            "task_id": self._task_id,
            "sample": sample,
        }

    def _load_semseg(self,label_path):
        # Note: 和其他工作一样，直接忽视背景类
        # 3类
        _semseg = np.array(Image.open(label_path)).astype(float)
        _semseg[_semseg == 0] = 256
        _semseg = _semseg - 1
        return _semseg


# 组合成一个多任务数据集
class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {} #建立task id 到dataset的映射
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, (
                "Duplicate task_id %s" % task_id
            )
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets) #多任务数据集的数据量

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]

# 定义batch的方式
class MultiTaskBatchSampler(BatchSampler):
    def __init__(
        self,
        datasets,
        batch_size,
        mix_opt, #控制混合任务的情况
        extra_task_ratio, #控制辅助任务的比例，为负数表示全量使用所有辅助任务
        bin_on=False, #开启分箱，应对数据不平衡
        bin_size=4, 
        bin_grow_ratio=0.5,
    ):
        self._datasets=datasets #多任务数据集
        self._batch_size=batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        train_data_list=[]
        for dataset in self._datasets:
            if bin_on:
                train_data_list.append(
                    self._get_shuffled_index_batches_bin(
                        dataset,
                        batch_size,
                        bin_size=bin_size,
                        bin_grow_ratio=bin_grow_ratio
                    )
                )
            else:
                train_data_list.append(
                    self._get_shuffled_index_batches(len(dataset),batch_size)
                )
        self._train_data_list=train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len,batch_size):
        #最普通的数据分batch方式
        index_batches=[
            list(range(i,min(i+batch_size,dataset_len)))
            for i in range(0,dataset_len,batch_size)
        ]
        #按batch_size，生成批次列表，e.g.,index_batches=[[0,1,2],[3,4,5],...]
        random.shuffle(index_batches)
        return index_batches

    @staticmethod
    def _get_shuffled_index_batches_bin(dataset,batch_size,bin_size,bin_grow_ratio):
        pass

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list) #整个多任务数据集的总batch数

    def __iter__(self):
        all_iters=[iter(item) for item in self._train_data_list]
        #为每个子任务样本集创建一个迭代器，组合成迭代器列表
        all_indices=self._gen_task_indices(
            self._train_data_list, self._mix_opt, self._extra_task_ratio
        )
        for local_task_idx in all_indices:
            task_id=self._datasets[local_task_idx].get_task_id()
            batch=next(all_iters[local_task_idx]) #调用一次迭代器
            yield [(task_id, sample_id) for sample_id in batch]
            #返回的是(task_id，当前batch中样本id)

    @staticmethod
    def _gen_task_indices(train_data_list,mix_opt,extra_task_ratio):
        all_indices=[]
        if len(train_data_list)>1 and extra_task_ratio >0:
            #将多任务划分成主任务0，与多个辅助任务的情况
            main_indices=[0]*len(train_data_list[0])
            extra_indices=[]
            for i in range(1,len(train_data_list)):
                extra_indices+=[i]*len(train_data_list[i])
            random_picks=int(
                min(len(train_data_list[0])*extra_task_ratio,len(extra_indices))
            )
            extra_indices=np.random.choice(extra_indices,random_picks,replace=False)
            #从所有辅助任务样本中随机抽取指定数量的样本
            if mix_opt>0:
                #辅助任务在前，主任务在后，乱序
                extra_indices=extra_indices.totist()
                random.shuffle(extra_indices) #辅助任务内部乱序
                all_indices=extra_indices+main_indices
            else:
                #非乱序
                all_indices=main_indices+extra_indices.tolist()
        else:
            for i in range(1,len(train_data_list)):
                all_indices+=[i]*len(train_data_list[i])
            if mix_opt>0:
                random.shuffle(all_indices)
            all_indices+=[0]*len(train_data_list[0])
        if mix_opt<1:
            random.shuffle(all_indices)
            return all_indices





