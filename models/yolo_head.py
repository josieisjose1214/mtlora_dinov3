# Neck+Head
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import C2f,Conv,DWConv,DFL
from .tal import dist2bbox,make_anchors

class Detect(nn.Module):
    """
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.
    """
    dynamic=False
    export=False
    format=None
    max_det=300
    shape=None
    anchors=torch.empty(0) #init
    strides=torch.empty(0) #init
    legacy=False
    xyxy=False #xyxy or xywh format

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=(32, 64, 128)):
        super(Detect, self).__init__()
        """
        Parameters:
        nc (int): Number of classes
        reg_max (int): maximum number of DFL channels
        end2end (bool): whether to use end2end NMS-free detection
        ch (tuple): channel size of feature maps.
        """
        self.nc = nc
        self.nl=len(ch) #Detect的个数
        self.reg_max=reg_max # ch[0]// 16=4/8/12对应yolo n/s/m
        self.no= nc+self.reg_max*4 #Detect总的输出通道数
        self.stride=torch.zeros(self.nl)
        c2,c3=max((16,ch[0]//4,self.reg_max*4)), max(ch[0],min(self.nc, 100)) #回归头和分类头的输出通道
        self.cv2=nn.ModuleList(
            nn.Sequential(Conv(x, c2,3), Conv(c2,c2,3),nn.Conv2d(c2,4*self.reg_max,1)) for x in ch
        )
        self.cv3=(
            nn.ModuleList(nn.Sequential(Conv(x,c3,3), Conv(c3,c3,3),nn.Conv2d(c3,self.nc,1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x,x,3), Conv(x,c3,1)),
                    nn.Sequential(DWConv(c3,c3,3), Conv(c3,c3,1)),
                    nn.Conv2d(c3,self.nc,1),
                )
                for x in ch
            )
        )
        self.dfl=DFL(self.reg_max) if self.reg_max>1 else nn.Identity()

        if end2end:
            self.one2one_cv2=copy.deepcopy(self.cv2)
            self.one2one_cv3=copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility.
        传统YOLO风格
        """
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components.
        RT-DETR风格
        """
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one for v5/v5/v8/v9/11 backward compatibility."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self._end2end = value

    def forward_head(
            self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        #变成1维anchor序列
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        #[B, 4*reg_max, 14*14+28*28+56*56]
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        #[B, nc, 14*14+28*28+56*56]
        return dict(boxes=boxes, scores=scores, feats=x) #boxes: [B, 64, 4116], scores: [B, 1, 4116], [x_detect0,x_detect1,x_detect2]

    def forward(self, x):
        preds=self.forward_head(x,**self.one2many)
        if self.end2end:
            x_detach=[xi.detach() for xi in x]
            one2one=self.forward_head(x_detach, **self.one2one)
            preds={"one2many": preds, "one2one": one2one}
        if self.training:
            return preds #raw prediction，包含boxes,scores,x
        # 测试阶段进行box的解码
        y=self._inference(preds["one2one"] if self.end2end else preds) #解码后的bbox与scores拼接所得
        if self.end2end:
            #NMS-free，即不做IoU NMS, 取score最大的TOP-K
            y=self.postprocess(y.permute(0,2,1))
            #y=[B, max_det, 6]最后的6维表示[x, y, w, h, max_class_prob, class_index]
        return y if self.export else (y,preds)

    def _inference(self,x):
        """
        x可以简单理解为上面求到的preds，包含box,scores和features
        """
        dbox=self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1) #[B, 4+nc, 4116]

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW, hw=14*14
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5)) #为每个feature point建anchor
            self.shape = shape

        #self.dfl返回[B,4,N]的期望距离
        #乘以strides映射回原图尺寸
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox #[B,4,N] 4表示xywh


    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None

class YOLOHead(nn.Module):
    def __init__(self, num_outputs):
        super(YOLOHead, self).__init__()
        """
        :param num_outputs: number of classes
        """
        self.num_outputs = num_outputs #类别数目
        self.c2f_23_1=C2f(c1=216,c2=128,shortcut=False)
        self.c2f_23_2 = C2f(c1=128, c2=128, shortcut=False)
        self.c2f_23_3 = C2f(c1=128, c2=128, shortcut=False)

        self.c2f_12_1=C2f(c1=164, c2=64, shortcut=False)
        self.c2f_12_2 = C2f(c1=64, c2=64, shortcut=False)
        self.c2f_12_3 = C2f(c1=64, c2=64, shortcut=False)

        self.c2f_01_1=C2f(c1=82, c2=32, shortcut=False)
        self.c2f_01_2 = C2f(c1=32, c2=32, shortcut=False)
        self.c2f_01_3 = C2f(c1=32, c2=32, shortcut=False)

        self.down1=nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3,stride=2,padding=1)
        self.c2f_d1_1=C2f(c1=96, c2=64, shortcut=False)
        self.c2f_d1_2 = C2f(c1=64, c2=64, shortcut=False)
        self.c2f_d1_3 = C2f(c1=64, c2=64, shortcut=False)

        self.down2=nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3,stride=2,padding=1)
        self.c2f_d2_1 = C2f(c1=192, c2=128, shortcut=False)
        self.c2f_d2_2 = C2f(c1=128, c2=128, shortcut=False)
        self.c2f_d2_3 = C2f(c1=128, c2=128, shortcut=False)

        self.detect=Detect(nc=1,reg_max=16,ch=(32,64,128))
        self.detect.stride = torch.tensor([8., 16., 32.])

    def forward(self,x):
        # x[0]: 56*56*18
        # x[1]: 28*28*36
        # x[2]: 14*14*72
        # x[3]:14*14*144 最后一层输出
        x0_h, x0_w= x[0].size(2), x[0].size(3)
        x1_h, x1_w= x[1].size(2), x[1].size(3)

        # x[3]&x[2]: concat+c2f
        x_23=torch.concat((x[3],x[2]),dim=1) #x_23: 14*14*216
        x_23=self.c2f_23_3(self.c2f_23_2(self.c2f_23_1(x_23))) #x_23: 14*14*128

        # x_23&x[1]: upsample+ concat+c2f
        x_23_up=F.interpolate(x_23, (x1_h, x1_w), mode='bilinear')
        x_12=torch.concat((x_23_up,x[1]),dim=1) #x_12: 28*28*164
        x_12=self.c2f_12_3(self.c2f_12_2(self.c2f_12_1(x_12))) #x_12: 28*28*64

        # x_12&x[0]: upsample+ concat+ c2f
        x_12_up=F.interpolate(x_12, (x0_h, x0_w), mode='bilinear')
        x_01=torch.concat((x_12_up, x[0]),dim=1) #x_01: 56*56*82
        x_01=self.c2f_01_3(self.c2f_01_2(self.c2f_01_1(x_01))) # x_01: 56*56*32
        #第一个Detect使用的特征图
        x_detect0=x_01

        #反向
        x0_down=self.down1(x_01) #x0_down: 28*28*32
        x0=torch.cat((x0_down,x_12),dim=1) #x0: 28*28*96
        x0=self.c2f_d1_3(self.c2f_d1_2(self.c2f_d1_1(x0))) #x0: 28*28*64
        #第二个Detect使用的特征图
        x_detect1=x0

        x1_down=self.down2(x0) # x0: 14*14*64
        x1=torch.cat((x1_down,x_23),dim=1) #x1: 14*14*192
        x1=self.c2f_d2_3(self.c2f_d2_2(self.c2f_d2_1(x1))) #x1: 14*14*128
        #第三个Detect使用的特征图
        x_detect2=x1

        feat_list=[x_detect0,x_detect1,x_detect2]
        preds=self.detect(feat_list)
        #train阶段：end2end模式preds={"one2many":preds, "one2one":one2one}，否则返回Preds
        #one2many用于计算传统的YOLO loss, one2one用于匈牙利匹配
        #eval阶段：返回(y, preds),preds和训练的内容一样，end2end时y为[B, K, 6] K小于等于300 ，否则y为[B, xywh+nc, 4116]，y还没有NMS
        #y用于计算指标
        return preds





