# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
#
# Original file:
# License: Attribution-NonCommercial 4.0 International (https://github.com/facebookresearch/astmt/)
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------


import numpy.random as random
import numpy as np
import torch
import cv2
import math
import data.helpers as helpers
import torchvision


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """

    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False, flagvals=None):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg
        self.flagvals = flagvals

    def __call__(self, sample):
        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                # bbox格式: [N, 5] = [class, x, y, w, h] (相对坐标)
                # 旋转和缩放时，需要计算变换后的坐标
                tmp = sample[elem]
                if isinstance(tmp, np.ndarray) and len(tmp.shape) == 2 and tmp.shape[1] == 5:
                    if tmp.shape[0] > 0:
                        # 获取图像尺寸（从sample中的image获取）
                        if 'image' in sample:
                            img = sample['image']
                            if isinstance(img, np.ndarray):
                                h, w = img.shape[:2]
                            elif isinstance(img, torch.Tensor):
                                if len(img.shape) == 3:
                                    h, w = img.shape[1], img.shape[2]
                                else:
                                    h, w = 448, 448  # 默认值
                            else:
                                h, w = 448, 448  # 默认值
                        else:
                            h, w = 448, 448  # 默认值
                        
                        # 将相对坐标转换为绝对坐标
                        boxes = tmp.copy()
                        x_center = boxes[:, 1] * w  # x (相对 -> 绝对)
                        y_center = boxes[:, 2] * h  # y (相对 -> 绝对)
                        width = boxes[:, 3] * w     # w (相对 -> 绝对)
                        height = boxes[:, 4] * h    # h (相对 -> 绝对)
                        
                        # 计算旋转和缩放后的坐标
                        center = (w / 2, h / 2)
                        rot_rad = rot * math.pi / 180.0
                        cos_r = math.cos(rot_rad)
                        sin_r = math.sin(rot_rad)
                        
                        # 对每个框的中心点进行旋转和缩放
                        x_center_new = (x_center - center[0]) * cos_r - (y_center - center[1]) * sin_r + center[0]
                        y_center_new = (x_center - center[0]) * sin_r + (y_center - center[1]) * cos_r + center[1]
                        
                        # 缩放
                        x_center_new = x_center_new * sc
                        y_center_new = y_center_new * sc
                        width_new = width * sc
                        height_new = height * sc
                        
                        # 转换回相对坐标
                        boxes[:, 1] = np.clip(x_center_new / w, 0.0, 1.0)
                        boxes[:, 2] = np.clip(y_center_new / h, 0.0, 1.0)
                        boxes[:, 3] = np.clip(width_new / w, 0.0, 1.0)
                        boxes[:, 4] = np.clip(height_new / h, 0.0, 1.0)
                        
                        # 过滤掉无效的框（宽度或高度太小）
                        # 放宽过滤条件：允许更小的框（0.005而不是0.01），避免过度过滤
                        # 同时确保框的中心点在图像范围内（允许稍微超出，因为后续会clip）
                        # 但不要限制宽高的最大值，允许超过1.0的框（部分超出图像）
                        valid_mask = (boxes[:, 3] > 0.005) & (boxes[:, 4] > 0.005) & \
                                     (boxes[:, 1] >= -0.1) & (boxes[:, 1] <= 1.1) & \
                                     (boxes[:, 2] >= -0.1) & (boxes[:, 2] <= 1.1)
                        
                        if valid_mask.sum() > 0:
                            # 裁剪框的坐标到有效范围
                            filtered_boxes = boxes[valid_mask].copy()
                            filtered_boxes[:, 1] = np.clip(filtered_boxes[:, 1], 0.0, 1.0)  # x (中心x限制在0-1)
                            filtered_boxes[:, 2] = np.clip(filtered_boxes[:, 2], 0.0, 1.0)  # y (中心y限制在0-1)
                            # 不要限制宽高到max=1.0，允许超过1.0的框
                            filtered_boxes[:, 3] = np.clip(filtered_boxes[:, 3], 0.005, None)  # w (只确保为正)
                            filtered_boxes[:, 4] = np.clip(filtered_boxes[:, 4], 0.005, None)  # h (只确保为正)
                            sample[elem] = filtered_boxes
                        else:
                            # 如果没有有效框，创建空数组
                            # 注意：这会导致loss为0，但比保留无效框更好
                            sample[elem] = np.empty((0, 5), dtype=np.float32)
                elif isinstance(tmp, torch.Tensor) and len(tmp.shape) == 2 and tmp.shape[1] == 5:
                    # Tensor格式，转换为numpy处理
                    tmp_np = tmp.numpy() if tmp.requires_grad is False else tmp.detach().cpu().numpy()
                    # 使用相同的处理逻辑
                    if tmp_np.shape[0] > 0:
                        # ... (类似上面的处理)
                        # 简化处理：对于Tensor，暂时跳过复杂的旋转（因为需要图像尺寸）
                        pass
                # 如果不是bbox格式，跳过
                continue

            tmp = sample[elem]

            # 只对图像类数据（2D或3D数组）进行warpAffine
            if isinstance(tmp, np.ndarray) and len(tmp.shape) >= 2:
                h, w = tmp.shape[:2]
                center = (w / 2, h / 2)
                assert (center != 0)  # Strange behaviour warpAffine
                M = cv2.getRotationMatrix2D(center, rot, sc)
                if self.flagvals is None:
                    if ((tmp == 0) | (tmp == 1)).all():
                        flagval = cv2.INTER_NEAREST
                    elif 'gt' in elem and self.semseg:
                        flagval = cv2.INTER_NEAREST
                    else:
                        flagval = cv2.INTER_CUBIC
                else:
                    flagval = self.flagvals[elem]

                if elem == 'normals':
                    # Rotate Normals properly
                    in_plane = np.arctan2(tmp[:, :, 0], tmp[:, :, 1])
                    nrm_0 = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2)
                    rot_rad = rot * 2 * math.pi / 360
                    tmp[:, :, 0] = np.sin(in_plane + rot_rad) * nrm_0
                    tmp[:, :, 1] = np.cos(in_plane + rot_rad) * nrm_0
                tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

                if elem == 'depth':
                    tmp = tmp / sc

                sample[elem] = tmp
            elif isinstance(tmp, torch.Tensor) and len(tmp.shape) >= 2:
                # Tensor格式，需要转换为numpy处理
                tmp_np = tmp.numpy() if tmp.requires_grad is False else tmp.detach().cpu().numpy()
                h, w = tmp_np.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, rot, sc)
                tmp_np = cv2.warpAffine(tmp_np, M, (w, h), flags=cv2.INTER_CUBIC)
                sample[elem] = torch.from_numpy(tmp_np)

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """

    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None and self.resolutions is not None:
            # Allow extra resolution-only keys (e.g. image_<task>) for task-specific resize.
            missing_flag_keys = [k for k in self.flagvals.keys() if k not in self.resolutions]
            assert len(missing_flag_keys) == 0, (
                f"FixedResize missing resolutions for keys: {missing_flag_keys}"
            )

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())

        # Infer current task from sample keys (Wheat heterogeneous batches).
        current_task = None
        if 'semseg' in sample:
            current_task = 'semseg'
        elif 'bbox' in sample:
            current_task = 'detect'
        elif 'points' in sample:
            current_task = 'count'
        elif 'text' in sample:
            current_task = 'classify'
        for elem in elems:
            if 'meta' in elem:
                continue

            if 'text' in elem:
                continue

            if 'bbox' in elem:
                # 边界框（l=[class, x, y, w, h]）已经是相对坐标，无需进行resize操作。
                if elem in self.resolutions and self.resolutions[elem] is not None:
                    pass # 不需要执行任何操作，直接进入循环末尾返回 sample
                continue

            if elem == 'points':
                # 计数任务点标注 (N, 2) 像素坐标，随图像 resize 按比例缩放
                if 'image' in sample and elem in self.resolutions and self.resolutions.get(elem) is not None:
                    img = sample['image']
                    h_old, w_old = img.shape[0], img.shape[1]
                    target = self.resolutions[elem]
                    if isinstance(target, (list, tuple)):
                        h_new, w_new = target[0], target[1]
                    else:
                        h_new = w_new = target
                    pts = sample[elem]
                    if isinstance(pts, np.ndarray) and pts.size > 0 and pts.ndim == 2:
                        scale_y, scale_x = h_new / float(h_old), w_new / float(w_old)
                        sample[elem] = pts.copy()
                        sample[elem][:, 0] *= scale_y
                        sample[elem][:, 1] *= scale_x
                continue

            if elem == 'density':
                continue

            if elem in self.resolutions:
                #if 'image' in self.resolutions
                target_resolution = self.resolutions[elem]
                if elem == 'image' and current_task is not None:
                    # Allow per-task image resolution via image_<task> key.
                    per_task_key = f'image_{current_task}'
                    if per_task_key in self.resolutions:
                        target_resolution = self.resolutions[per_task_key]
                if target_resolution is None:
                    continue
                if isinstance(sample[elem], list):
                    #对原图进行resize处理
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(target_resolution, [
                                                3, len(sample[elem])])
                    else:
                        output_size = np.append(
                            target_resolution, len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=float)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(
                                crop, target_resolution)
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(
                                crop, target_resolution, flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(
                            sample[elem], target_resolution)
                    else:
                        sample[elem] = helpers.fixed_resize(
                            sample[elem], target_resolution, flagval=self.flagvals[elem])

                    if elem == 'normals':
                        N1, N2, N3 = sample[elem][:, :,
                                                  0], sample[elem][:, :, 1], sample[elem][:, :, 2]
                        Nn = np.sqrt(N1 ** 2 + N2 ** 2 + N3 **
                                     2) + np.finfo(float).eps
                        sample[elem][:, :, 0], sample[elem][:, :,
                                                            1], sample[elem][:, :, 2] = N1/Nn, N2/Nn, N3/Nn
            else:
                del sample[elem]

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class FixedResizeRatio(object):
    """Fixed resize for the image and the ground truth to specified scale.
    Args:
        scales (float): the scale
    """

    def __init__(self, scale=None, flagvals=None):
        self.scale = scale
        self.flagvals = flagvals

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            if elem in self.flagvals:
                if self.flagvals[elem] is None:
                    continue

                tmp = sample[elem]
                tmp = cv2.resize(tmp, None, fx=self.scale,
                                 fy=self.scale, interpolation=self.flagvals[elem])

                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'FixedResizeRatio: '+str(self.scale)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem or 'text' in elem:
                    continue
                elif elem == 'points':
                    # 点坐标 (N, 2)，通常为 (y, x) 像素坐标，翻转时 x_new = W - x_old
                    tmp = sample[elem]
                    if isinstance(tmp, np.ndarray) and tmp.ndim == 2 and tmp.shape[1] >= 2 and tmp.size > 0:
                        if 'image' in sample:
                            w = sample['image'].shape[1]
                            tmp = tmp.copy()
                            tmp[:, 1] = w - 1 - tmp[:, 1]
                            sample[elem] = tmp
                    elif isinstance(tmp, torch.Tensor) and tmp.ndim == 2 and tmp.shape[1] >= 2 and tmp.numel() > 0:
                        if 'image' in sample:
                            w = sample['image'].shape[2]
                            tmp = tmp.clone()
                            tmp[:, 1] = w - 1 - tmp[:, 1]
                            sample[elem] = tmp
                elif 'bbox' in elem:
                    # bbox格式: [N, 5] = [class, x, y, w, h] (相对坐标)
                    # 水平翻转时，只需要改变x坐标: x_new = 1.0 - x_old
                    tmp = sample[elem]
                    if isinstance(tmp, np.ndarray) and len(tmp.shape) == 2 and tmp.shape[1] == 5:
                        # 确保是bbox格式 [N, 5]
                        if tmp.shape[0] > 0:
                            # x坐标是第1列（索引1），需要翻转: x_new = 1.0 - x_old
                            tmp[:, 1] = 1.0 - tmp[:, 1]
                            sample[elem] = tmp
                    elif isinstance(tmp, torch.Tensor) and len(tmp.shape) == 2 and tmp.shape[1] == 5:
                        # Tensor格式
                        if tmp.shape[0] > 0:
                            tmp[:, 1] = 1.0 - tmp[:, 1]
                            sample[elem] = tmp
                    # 如果不是bbox格式，跳过（可能是其他数据）
                else:
                    tmp = sample[elem]
                    # 只对图像类数据（2D或3D数组）进行flip
                    if isinstance(tmp, np.ndarray) and len(tmp.shape) >= 2:
                        tmp = cv2.flip(tmp, flipCode=1)
                        sample[elem] = tmp
                    elif isinstance(tmp, torch.Tensor) and len(tmp.shape) >= 2:
                        # Tensor格式，需要转换为numpy进行flip，再转回
                        tmp_np = tmp.numpy() if tmp.requires_grad is False else tmp.detach().cpu().numpy()
                        tmp_np = cv2.flip(tmp_np, flipCode=1)
                        sample[elem] = torch.from_numpy(tmp_np)

                if elem == 'normals':
                    sample[elem][:, :, 0] *= -1

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class NormalizeImage(object):
    """
    Return the given elements between 0 and 1
    """

    def __init__(self, norm_elem='image', clip=False):
        self.norm_elem = norm_elem
        self.clip = clip

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                if np.max(sample[elem]) > 1:
                    sample[elem] /= 255.0
        else:
            if self.clip:
                sample[self.norm_elem] = np.clip(
                    sample[self.norm_elem], 0, 255)
            if np.max(sample[self.norm_elem]) > 1:
                sample[self.norm_elem] /= 255.0
        return sample

    def __str__(self):
        return 'NormalizeImage'


class ToImage(object):
    """
    Return the given elements between 0 and 255
    """

    def __init__(self, norm_elem='image', custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * \
                    (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * \
                (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'


class AddIgnoreRegions(object):
    """Add Ignore Regions"""

    def __call__(self, sample):

        for elem in sample.keys():
            tmp = sample[elem]

            if elem == 'normals':
                # Check areas with norm 0
                Nn = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1]
                             ** 2 + tmp[:, :, 2] ** 2)

                tmp[Nn == 0, :] = 255.
                sample[elem] = tmp

            elif elem == 'human_parts':
                # Check for images without human part annotations
                if (tmp == 0).all():
                    tmp = 255 * np.ones(tmp.shape, dtype=tmp.dtype)
                    sample[elem] = tmp

            elif elem == 'depth':
                tmp[tmp == 0] = 255.
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'AddIgnoreRegions'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                # 确保bbox是numpy数组
                if isinstance(tmp, np.ndarray):
                    # 确保是2D数组 [N, 5]
                    if len(tmp.shape) == 1:
                        # 如果是1D数组，检查长度
                        if tmp.shape[0] == 5:
                            # 单个bbox，reshape为 [1, 5]
                            tmp = tmp.reshape(1, 5)
                        else:
                            # 空数组或其他格式，转换为 [0, 5]
                            tmp = np.empty((0, 5), dtype=np.float32)
                    elif len(tmp.shape) == 2:
                        # 确保第二维是5
                        if tmp.shape[1] != 5:
                            if tmp.shape[0] > 0 and tmp.shape[1] > 5:
                                # 如果有多余的列，只取前5列
                                tmp = tmp[:, :5]
                            else:
                                # 格式错误，返回空数组
                                tmp = np.empty((0, 5), dtype=np.float32)
                    # 转换为tensor
                    sample[elem] = torch.from_numpy(tmp)
                elif isinstance(tmp, torch.Tensor):
                    # 如果已经是tensor，确保格式正确
                    if len(tmp.shape) == 1:
                        if tmp.shape[0] == 5:
                            tmp = tmp.unsqueeze(0)
                        else:
                            tmp = torch.empty(0, 5, dtype=tmp.dtype)
                    elif len(tmp.shape) == 2:
                        if tmp.shape[1] != 5:
                            if tmp.shape[0] > 0 and tmp.shape[1] > 5:
                                tmp = tmp[:, :5]
                            else:
                                tmp = torch.empty(0, 5, dtype=tmp.dtype)
                    sample[elem] = tmp
                else:
                    # 其他类型，尝试转换或创建空数组
                    try:
                        tmp = np.array(tmp, dtype=np.float32)
                        if len(tmp.shape) == 1 and tmp.shape[0] == 5:
                            tmp = tmp.reshape(1, 5)
                        elif len(tmp.shape) != 2 or tmp.shape[1] != 5:
                            tmp = np.empty((0, 5), dtype=np.float32)
                        sample[elem] = torch.from_numpy(tmp)
                    except:
                        sample[elem] = torch.empty(0, 5, dtype=torch.float32)
                continue
            elif 'text' in elem:
                tmp = sample[elem]
            
                # 1. 检查并确保转换为 NumPy 数组
                # 必须进行这一步，因为 torch.from_numpy 只能接受 NumPy 数组
                if not isinstance(tmp, np.ndarray):
                    # 将 Python int 转换为 NumPy 数组。
                    tmp = np.array(tmp) 
                
                # 2. 转换为 PyTorch LongTensor
                # 分类标签需要 LongTensor (int64) 类型
                    sample[elem] = torch.from_numpy(tmp).long()
                continue
            elif elem == 'points':
                tmp = sample[elem]
                if isinstance(tmp, np.ndarray):
                    sample[elem] = torch.from_numpy(tmp.astype(np.float32))
                elif not isinstance(tmp, torch.Tensor):
                    sample[elem] = torch.tensor(tmp, dtype=torch.float32)
                continue
            elif elem == 'density':
                tmp = sample[elem]
                if isinstance(tmp, np.ndarray):
                    sample[elem] = torch.from_numpy(tmp.astype(np.float32)).squeeze()
                elif not isinstance(tmp, torch.Tensor):
                    sample[elem] = torch.tensor(tmp, dtype=torch.float32).squeeze()
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            if elem == 'image':
                # Between 0 .. 255 so cast as uint8 to ensure compatible w/ imagenet weight
                sample[elem] = self.to_tensor(tmp.astype(np.uint8))

            else:
                tmp = tmp.transpose((2, 0, 1))
                sample[elem] = torch.from_numpy(tmp.astype(float))

        return sample

    def __str__(self):
        return 'ToTensor'


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = torchvision.transforms.Normalize(self.mean, self.std)

    def __call__(self, sample):
        sample['image'] = self.normalize(sample['image'])
        return sample

    def __str__(self):
        return 'Normalize([%.3f,%.3f,%.3f],[%.3f,%.3f,%.3f])' % (self.mean[0], self.mean[1], self.mean[2], self.std[0], self.std[1], self.std[2])
