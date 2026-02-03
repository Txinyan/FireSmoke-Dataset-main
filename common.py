# # This file contains modules common to various models
# import math

# import torch
# import torch.nn as nn


# def autopad(k, p=None):  # kernel, padding
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# def DWConv(c1, c2, k=1, s=1, act=True):
#     # Depthwise convolution
#     return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def fuseforward(self, x):
#         return self.act(self.conv(x))


# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super(Bottleneck, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class BottleneckCSP(nn.Module):
#     # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(BottleneckCSP, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

#     def forward(self, x):
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


# class SPP(nn.Module):
#     # Spatial pyramid pooling layer used in YOLOv3-SPP
#     def __init__(self, c1, c2, k=(5, 9, 13)):
#         super(SPP, self).__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

#     def forward(self, x):
#         x = self.cv1(x)
#         return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# class Focus(nn.Module):
#     # Focus wh information into c-space
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Focus, self).__init__()
#         self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

#     def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
#         return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


# class Concat(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, dimension=1):
#         super(Concat, self).__init__()
#         self.d = dimension

#     def forward(self, x):
#         return torch.cat(x, self.d)


# class Flatten(nn.Module):
#     # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
#     @staticmethod
#     def forward(x):
#         return x.view(x.size(0), -1)


# class Classify(nn.Module):
#     # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Classify, self).__init__()
#         self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
#         self.flat = Flatten()

#     def forward(self, x):
#         z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
#         return self.flat(self.conv(z))  # flatten to x(b,c2)


# # *******添加注意力机制
# # This file contains modules common to various models
# import math

# import torch
# import torch.nn as nn


# def autopad(k, p=None):  # kernel, padding
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# def DWConv(c1, c2, k=1, s=1, act=True):
#     # Depthwise convolution
#     return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def fuseforward(self, x):
#         return self.act(self.conv(x))


# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super(Bottleneck, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class BottleneckCSP(nn.Module):
#     # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(BottleneckCSP, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

#     def forward(self, x):
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


# class SPP(nn.Module):
#     # Spatial pyramid pooling layer used in YOLOv3-SPP
#     def __init__(self, c1, c2, k=(5, 9, 13)):
#         super(SPP, self).__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

#     def forward(self, x):
#         x = self.cv1(x)
#         return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# class Focus(nn.Module):
#     # Focus wh information into c-space
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Focus, self).__init__()
#         self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

#     def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
#         return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


# class Concat(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, dimension=1):
#         super(Concat, self).__init__()
#         self.d = dimension

#     def forward(self, x):
#         return torch.cat(x, self.d)


# class Flatten(nn.Module):
#     # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
#     @staticmethod
#     def forward(x):
#         return x.view(x.size(0), -1)


# class Classify(nn.Module):
#     # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Classify, self).__init__()
#         self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
#         self.flat = Flatten()

#     def forward(self, x):
#         z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
#         return self.flat(self.conv(z))  # flatten to x(b,c2)

# # ==================== 新增的DS-Bottleneck优化模块 ====================

# class DS_Attention_Light(nn.Module):
#     """极轻量DS注意力模块"""
#     def __init__(self, channels):
#         super(DS_Attention_Light, self).__init__()
#         # 深度卷积提取空间特征
#         self.depth_conv = nn.Conv2d(channels, channels, 3,
#                                    padding=1, groups=channels, bias=False)
#         self.bn = nn.BatchNorm2d(channels)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 生成空间注意力权重
#         attention = self.sigmoid(self.bn(self.depth_conv(x)))
#         return x * attention  # 特征重标定


# class FireSmoke_Bottleneck_DS(nn.Module):
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.4):
#         super(FireSmoke_Bottleneck_DS, self).__init__()
#         c_ = int(c2 * e)

#         self.cv1 = Conv(c1, c_, 1, 1)

#         # 修复：深度可分离卷积结构过于复杂，建议简化
#         self.cv2 = nn.Sequential(
#             # 深度卷积
#             nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
#             nn.BatchNorm2d(c_),
#             nn.LeakyReLU(0.1, inplace=True),
#             # 点卷积 (修复：移除多余的卷积层)
#             nn.Conv2d(c_, c2, 1, 1, bias=False),
#             nn.BatchNorm2d(c2),
#         )

#         self.use_attention = c2 >= 128
#         if self.use_attention:
#             self.ds_attention = DS_Attention_Light(c2)

#         self.add = shortcut and c1 == c2
#         self.act = nn.LeakyReLU(0.1, inplace=True)  # 统一的激活函数

#     def forward(self, x):
#         out = self.cv2(self.cv1(x))
#         if self.use_attention:
#             out = self.ds_attention(out)
#         result = x + out if self.add else out
#         return self.act(result)  # 确保最后有激活函数


# class C3_DS(nn.Module):
#     """DS-Bottleneck优化的C3模块 - 与标准C3接口兼容"""
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # 使用标准e=0.5
#         super(C3_DS, self).__init__()
#         # 确保 n 是整数
#         n = int(n) if isinstance(n, float) else n
#         c_ = int(c2 * e)  # 隐藏通道数

#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # 输出融合卷积

#         # 使用DS-Bottleneck替换普通Bottleneck
#         self.m = nn.Sequential(*[FireSmoke_Bottleneck_DS(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# class C3_DS_Enhanced(nn.Module):
#     """增强版DS-C3模块 - 与标准C3接口兼容"""
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super(C3_DS_Enhanced, self).__init__()
#         n = int(n) if isinstance(n, float) else n
#         c_ = int(c2 * e)

#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)

#         self.m = nn.Sequential(*[FireSmoke_Bottleneck_DS(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#         self.output_attention = DS_Attention_Light(c2)

#     def forward(self, x):
#         out = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
#         return self.output_attention(out)

# # ==================== 其他现有模块 ====================

# class SPPF(nn.Module):
#     # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
#     def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5,9,13))
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         x = self.cv1(x)
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# # 添加缺失的模块定义
# class Contract(nn.Module):
#     # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
#     def __init__(self, gain=2):
#         super().__init__()
#         self.gain = gain

#     def forward(self, x):
#         b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
#         s = self.gain
#         x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
#         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
#         return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)

# class Expand(nn.Module):
#     # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
#     def __init__(self, gain=2):
#         super().__init__()
#         self.gain = gain

#     def forward(self, x):
#         b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
#         s = self.gain
#         x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
#         x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
#         return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)

# # 检测头模块（简化版）
# class Detect(nn.Module):
#     def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
#         super(Detect, self).__init__()
#         self.stride = None  # strides computed during build
#         self.nc = nc  # number of classes
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         a = torch.tensor(anchors).float().view(self.nl, -1, 2)
#         self.register_buffer('anchors', a)  # shape(nl,na,2)
#         self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
#         self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

#     def forward(self, x):
#         z = []  # inference output
#         for i in range(self.nl):
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#         return x

# # 标准C3模块（确保存在）
# class C3(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super(C3, self).__init__()
#         c_ = int(c2 * e)
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# # 导出列表 - 包含所有模块（修正）
# __all__ = [
#     'Conv', 'Bottleneck', 'BottleneckCSP', 'C3', 'SPP', 'SPPF', 'Concat',
#     'Detect', 'Contract', 'Expand', 'Flatten', 'Classify', 'Focus',
#     'C3_DS', 'C3_DS_Enhanced', 'FireSmoke_Bottleneck_DS', 'DS_Attention_Light'
# ]


# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def autopad(k, p=None):  # kernel, padding
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p

# # --------------- 深度可分离卷积 (DWConv) ---------------
# class DWConv(nn.Module):
#     def __init__(self, c1, c2, k=3, s=1, act=True):
#         super(DWConv, self).__init__()
#         # 深度卷积保持输入输出通道相同
#         self.depthwise = nn.Conv2d(c1, c1, k, s, k//2, groups=c1, bias=False)
#         # 点卷积进行通道变换
#         self.pointwise = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.act(x)

# # --------------- 轻量化注意力机制 (ECA Attention) ---------------
# class DS_Attention_Light(nn.Module):
#     """极轻量DS注意力模块"""
#     def __init__(self, channels):
#         super(DS_Attention_Light, self).__init__()
#         # 深度卷积提取空间特征
#         self.depth_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
#         self.bn = nn.BatchNorm2d(channels)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 生成空间注意力权重
#         attention = self.sigmoid(self.bn(self.depth_conv(x)))
#         return x * attention  # 特征重标定

# # --------------- 常规卷积层 (Conv) ---------------
# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def fuseforward(self, x):
#         return self.act(self.conv(x))

# # --------------- 标准瓶颈层 (Bottleneck) ---------------
# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
#         super(Bottleneck, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# # --------------- CSP Bottleneck (BottleneckCSP) ---------------
# class BottleneckCSP(nn.Module):
#     # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super(BottleneckCSP, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

#     def forward(self, x):
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

# # --------------- 空间金字塔池化 (SPP) ---------------
# class SPP(nn.Module):
#     # Spatial pyramid pooling layer used in YOLOv3-SPP
#     def __init__(self, c1, c2, k=(5, 9, 13)):
#         super(SPP, self).__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

#     def forward(self, x):
#         x = self.cv1(x)
#         return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

# # --------------- 特征图拼接 (Concat) ---------------
# class Concat(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, dimension=1):
#         super(Concat, self).__init__()
#         self.d = dimension

#     def forward(self, x):
#         return torch.cat(x, self.d)

# # --------------- 展平 (Flatten) ---------------
# class Flatten(nn.Module):
#     # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
#     @staticmethod
#     def forward(x):
#         return x.view(x.size(0), -1)

# # --------------- 分类头 (Classify) ---------------
# class Classify(nn.Module):
#     # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
#         super(Classify, self).__init__()
#         self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
#         self.flat = Flatten()

#     def forward(self, x):
#         z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
#         return self.flat(self.conv(z))

# # --------------- 轻量化瓶颈层 (FireSmoke_Bottleneck_DS) ---------------
# class FireSmoke_Bottleneck_DS(nn.Module):
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.4):
#         super(FireSmoke_Bottleneck_DS, self).__init__()
#         c_ = int(c2 * e)

#         self.cv1 = Conv(c1, c_, 1, 1)

#         # 修复：深度可分离卷积结构过于复杂，建议简化
#         self.cv2 = nn.Sequential(
#             # 深度卷积
#             nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
#             nn.BatchNorm2d(c_),
#             nn.LeakyReLU(0.1, inplace=True),
#             # 点卷积 (修复：移除多余的卷积层)
#             nn.Conv2d(c_, c2, 1, 1, bias=False),
#             nn.BatchNorm2d(c2),
#         )

#         self.use_attention = c2 >= 128
#         if self.use_attention:
#             self.ds_attention = DS_Attention_Light(c2)

#         self.add = shortcut and c1 == c2
#         self.act = nn.LeakyReLU(0.1, inplace=True)  # 统一的激活函数

#     def forward(self, x):
#         out = self.cv2(self.cv1(x))
#         if self.use_attention:
#             out = self.ds_attention(out)
#         result = x + out if self.add else out
#         return self.act(result)  # 确保最后有激活函数

# # --------------- C3 模块 (C3_DS) ---------------
# class C3_DS(nn.Module):
#     """DS-Bottleneck优化的C3模块 - 与标准C3接口兼容"""
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # 使用标准e=0.5
#         super(C3_DS, self).__init__()
#         n = int(n) if isinstance(n, float) else n
#         c_ = int(c2 * e)  # 隐藏通道数

#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # 输出融合卷积

#         # 使用DS-Bottleneck替换普通Bottleneck
#         self.m = nn.Sequential(*[FireSmoke_Bottleneck_DS(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#         self.output_attention = DS_Attention_Light(c2)

#     def forward(self, x):
#         out = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
#         return self.output_attention(out)


# # ==================== 其他现有模块 ====================

# class SPPF(nn.Module):
#     # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
#     def __init__(self, c1, c2, k=5):
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         x = self.cv1(x)
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         y3 = self.m(y2)
#         return self.cv2(torch.cat((x, y1, y2, y3), 1))

# class Contract(nn.Module):
#     # Contract width-height into channels
#     def __init__(self, gain=2):
#         super().__init__()
#         self.gain = gain

#     def forward(self, x):
#         b, c, h, w = x.size()
#         s = self.gain
#         x = x.view(b, c, h // s, s, w // s, s)
#         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
#         return x.view(b, c * s * s, h // s, w // s)

# class Expand(nn.Module):
#     # Expand channels into width-height
#     def __init__(self, gain=2):
#         super().__init__()
#         self.gain = gain

#     def forward(self, x):
#         b, c, h, w = x.size()
#         s = self.gain
#         x = x.view(b, s, s, c // s ** 2, h, w)
#         x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
#         return x.view(b, c // s ** 2, h * s, w * s)

# class Detect(nn.Module):
#     def __init__(self, nc=80, anchors=(), ch=()):
#         super(Detect, self).__init__()
#         self.stride = None
#         self.nc = nc
#         self.no = nc + 5
#         self.nl = len(anchors)
#         self.na = len(anchors[0]) // 2
#         self.grid = [torch.zeros(1)] * self.nl
#         a = torch.tensor(anchors).float().view(self.nl, -1, 2)
#         self.register_buffer('anchors', a)
#         self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
#         self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

#     def forward(self, x):
#         z = []
#         for i in range(self.nl):
#             x[i] = self.m[i](x[i])
#             bs, _, ny, nx = x[i].shape
#             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#         return x

# class C3(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super(C3, self).__init__()
#         c_ = int(c2 * e)
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# class Focus(nn.Module):
#     # Focus wh information into c-space
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         super(Focus, self).__init__()
#         self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

#     def forward(self, x):
#         return self.conv(torch.cat([
#             x[..., ::2, ::2],
#             x[..., 1::2, ::2],
#             x[..., ::2, 1::2],
#             x[..., 1::2, 1::2]
#         ], 1))

# # --------------- 导出的模块 ---------------
# __all__ = [
#     'Conv', 'Bottleneck', 'BottleneckCSP', 'C3', 'SPP', 'SPPF', 'Concat',
#     'Detect', 'Contract', 'Expand', 'Flatten', 'Classify', 'Focus',
#     'C3_DS', 'FireSmoke_Bottleneck_DS', 'DS_Attention_Light'
# ]


# This file contains modules common to various models
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    # Pad to 'same' shape, accounting for dilation
    if d != 1:
        k = d * (k - 1) + 1  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, d=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


# ==================== 修复后的注意力机制 ====================


class MultiScaleAttention(nn.Module):
    """多尺度注意力机制"""

    def __init__(self, channels, reduction=16):
        super(MultiScaleAttention, self).__init__()
        self.channels = channels

        # 多尺度卷积核
        self.conv3 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.conv5 = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=False
        )
        self.conv7 = nn.Conv2d(
            channels, channels, 7, padding=3, groups=channels, bias=False
        )

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # 多尺度特征提取
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        feat7 = self.conv7(x)

        # 通道注意力
        feat_cat = torch.cat(
            [feat3.unsqueeze(2), feat5.unsqueeze(2), feat7.unsqueeze(2)], dim=2
        )
        feat_pool = torch.mean(feat_cat, dim=2)
        channel_weights = self.channel_attention(feat_pool)

        # 特征融合
        weighted_feat = (feat3 + feat5 + feat7) / 3
        weighted_feat = weighted_feat * channel_weights
        weighted_feat = self.bn(weighted_feat)

        return x + weighted_feat


class DS_Attention_Light(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 8), 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 8, 8), channels, 1, bias=False),
            # 移除Sigmoid，让权重可以大于1
        )

        # 添加可学习的scale参数
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = x

        # 空间特征
        spatial_feat = self.depth_conv(x)
        spatial_feat = self.bn1(spatial_feat)

        # 通道注意力 - 移除激活，保持线性
        channel_weights = self.channel_attention(x)

        # 计算注意力权重 - 使用softmax或sigmoid+残差
        attention_weights = torch.sigmoid(spatial_feat) * torch.sigmoid(channel_weights)

        # 关键修复：让注意力可以增强特征
        attention_weights = 1 + self.scale * (attention_weights - 0.5)

        # 应用注意力
        out = x * attention_weights

        # 残差连接
        return residual + out


class EnhancedBottleneck(nn.Module):
    """修复的增强Bottleneck模块"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(EnhancedBottleneck, self).__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)

        # 多分支卷积 - 修复dilation参数问题
        self.conv_branches = nn.ModuleList(
            [
                Conv(c_, c_ // 4, 3, 1, g=g),  # 标准卷积
                Conv(c_, c_ // 4, 3, 1, d=2, g=g),  # 扩张卷积
                Conv(c_, c_ // 4, 1, 1),  # 点卷积
                Conv(c_, c_ - 3 * (c_ // 4), 3, 1, g=g),  # 剩余通道
            ]
        )

        self.cv2 = Conv(c_, c2, 1, 1)
        self.attention = DS_Attention_Light(c2)  # 使用轻量注意力
        self.add = shortcut and c1 == c2

    def forward(self, x):
        residual = x

        out = self.cv1(x)

        # 多分支特征提取
        branch_outs = []
        for conv in self.conv_branches:
            branch_outs.append(conv(out))
        out = torch.cat(branch_outs, dim=1)

        out = self.cv2(out)
        out = self.attention(out)

        if self.add:
            out = out + residual

        return out


class FireSmoke_Bottleneck_DS(nn.Module):
    """修复后的FireSmoke_Bottleneck_DS模块"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(FireSmoke_Bottleneck_DS, self).__init__()
        c_ = int(c2 * e)

        # 输入投影
        self.cv1 = Conv(c1, c_, 1, 1)

        # 深度可分离卷积 + 扩张卷积
        self.dw_conv1 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        self.dw_conv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 2, dilation=2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # 点卷积
        self.pw_conv = nn.Sequential(
            nn.Conv2d(c_ * 2, c2, 1, 1, bias=False), nn.BatchNorm2d(c2)
        )

        self.add = shortcut and c1 == c2
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.cv1(x)

        # 多尺度深度卷积
        out1 = self.dw_conv1(out)
        out2 = self.dw_conv2(out)

        # 特征融合
        out = torch.cat([out1, out2], dim=1)
        out = self.pw_conv(out)

        # 移除注意力机制

        if self.add:
            out = out + residual

        return self.act(out)


# # **********加入注意力机制
# class FireSmoke_Bottleneck_DS(nn.Module):
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.4):
#         super(FireSmoke_Bottleneck_DS, self).__init__()
#         c_ = int(c2 * e)

#         self.cv1 = Conv(c1, c_, 1, 1)

#         # 修复：深度可分离卷积结构过于复杂，建议简化
#         self.cv2 = nn.Sequential(
#             # 深度卷积
#             nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
#             nn.BatchNorm2d(c_),
#             nn.LeakyReLU(0.1, inplace=True),
#             # 点卷积 (修复：移除多余的卷积层)
#             nn.Conv2d(c_, c2, 1, 1, bias=False),
#             nn.BatchNorm2d(c2),
#         )

#         self.use_attention = c2 >= 128
#         if self.use_attention:
#             self.ds_attention = DS_Attention_Light(c2)

#         self.add = shortcut and c1 == c2
#         self.act = nn.LeakyReLU(0.1, inplace=True)  # 统一的激活函数

#     def forward(self, x):
#         out = self.cv2(self.cv1(x))
#         if self.use_attention:
#             out = self.ds_attention(out)
#         result = x + out if self.add else out
#         return self.act(result)  # 确保最后有激活函数


class C3_DS(nn.Module):
    """修复后的C3_DS模块 - 更稳定的设计"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_DS, self).__init__()
        n = int(n) if isinstance(n, float) else n
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # 使用稳定的FireSmoke_Bottleneck_DS
        self.m = nn.Sequential(
            *[FireSmoke_Bottleneck_DS(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

        # 移除输出注意力机制
        self.output_attention = None

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        out = self.cv3(torch.cat((y1, y2), dim=1))

        # 移除注意力处理
        # if self.output_attention is not None:
        #     out = self.output_attention(out)

        return out


class C3_DS_Enhanced(nn.Module):
    """修复后的增强版C3_DS模块"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_DS_Enhanced, self).__init__()
        n = int(n) if isinstance(n, float) else n
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # 使用EnhancedBottleneck
        self.m = nn.Sequential(
            *[EnhancedBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

        # 输出注意力
        self.output_attention = MultiScaleAttention(c2)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        out = self.cv3(torch.cat((y1, y2), dim=1))
        out = self.output_attention(out)
        return out


# ==================== 其他必要模块 ====================


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                1,
            )
        )


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Contract(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class Expand(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // s**2, h * s, w * s)


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        super(Detect, self).__init__()
        self.stride = None
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
        return x


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


__all__ = [
    "Conv",
    "Bottleneck",
    "BottleneckCSP",
    "C3",
    "SPP",
    "SPPF",
    "Concat",
    "Detect",
    "Contract",
    "Expand",
    "Flatten",
    "Classify",
    "Focus",
    "C3_DS",
    "C3_DS_Enhanced",
    "FireSmoke_Bottleneck_DS",
    "DS_Attention_Light",
]
