import torch
from torch import nn
from einops import rearrange
from skimage import io, color
import skimage
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2

class SRMConv2d_simple(nn.Module):

    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, hidden_size=768, dropout_rate=0.1, head_size=8):
        super(CrossAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, cache=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        # mask = mask.bool()

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        # print("qshape",q.shape)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
            # print("kshape", k.shape)
            # print("vshape", v.shape)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        # print("qshape", q.shape)
        # print("kshape", k.shape)
        # print("vshape", v.shape)
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len] mask.unsqueeze(1).shape 1,1,77,77
        # print("xshape",x.shape)
        # x.masked_fill_(mask.unsqueeze(1), -1e9) #masked_fill_() 是 PyTorch 张量的原地操作之一，用于根据给定的掩码（mask）张量，在指定位置上将元素替换为指定的值。用value填充tensor中与mask中值为1位置相对应的元素
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        # print(x.shape)

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
def texture_diversity(patches):
    # 计算每个patch的GLCM纹理特征
    homogeneity_scores = []
    to_grayscale = transforms.Grayscale(num_output_channels=1)

    # 将彩色图像张量转换为灰度图

    for patch in patches:
        # 如果Tensor在GPU上，先将其移动到CPU，然后转换为NumPy数组
        patch = to_grayscale(patch)
        patch = patch.squeeze()
        #print(patch.shape)
        patch = patch.cpu().numpy() if patch.is_cuda else patch.numpy()

        # 然后，使用.astype(np.uint8)来改变数据类型
        #patch_np_uint8 = (patch_np * 255).astype(np.uint8)
        # 现在，你可以将`patch_np_uint8`用于`greycomatrix`函数
       # glcm = greycomatrix(patch_np_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        # 计算GLCM
        glcm = skimage.feature.graycomatrix((patch * 255).astype(np.uint8), distances=[1], angles=[0], levels=256,
                                            symmetric=True,
                                            normed=True)
       # print("glcm",glcm)
        # 计算同质性
        homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0]
       # print(" homogeneity",  homogeneity)
        homogeneity_scores.append(homogeneity)
    # flatten_patches = patches.reshape(patches.size(0), -1)
    # std_dev = torch.std(flatten_patches, dim=1)
    return homogeneity_scores
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print("x1.shape",x1.shape) #1024,14,14
        # print("x2.shape", x2.shape)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512, bilinear=False)
        self.up2 = Up(512, 256, bilinear=False)
        self.up3 = Up(256, 128, bilinear=False)
        self.up4 = Up(128, 64, bilinear=False)
        self.outc = OutConv(64, n_classes)

    def encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # 512,14,14
        x5 = self.down4(x4)  # 1024,7,7
        # print("x5.shape",x5.shape)

        return x5, x4, x3, x2, x1

    def decoder(self,x5, x4, x3, x2, x1):
        #print(x4.shape)

        x = self.up1(x5, x4)
        #print(x.shape)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
# Assuming image_tensor is your 3x224x224 input image tensor
# image_tensor = torch.rand(3, 224, 224) # For example, random noise

# Function to divide the image into 32x32 patches
def get_patches(img_tensor, patch_size=112):
    c, h, w = img_tensor.size()
    img_tensor = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    img_tensor = img_tensor.contiguous().view(c, -1, patch_size, patch_size)
    img_tensor = img_tensor.permute(1, 0, 2, 3)
    return img_tensor

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size=768, dropout_rate=0.1, head_size=8):
        super(CrossAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, cache=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        # mask = mask.bool()

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        # print("qshape",q.shape)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
            # print("kshape", k.shape)
            # print("vshape", v.shape)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        # print("qshape", q.shape)
        # print("kshape", k.shape)
        # print("vshape", v.shape)
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len] mask.unsqueeze(1).shape 1,1,77,77
        # print("xshape",x.shape)
        # x.masked_fill_(mask.unsqueeze(1), -1e9) #masked_fill_() 是 PyTorch 张量的原地操作之一，用于根据给定的掩码（mask）张量，在指定位置上将元素替换为指定的值。用value填充tensor中与mask中值为1位置相对应的元素
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        # print(x.shape)

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class MPVE(nn.Module):
    def __init__(self, image_size=224, patch_size=7, num_classes=10, channels=512,
                 dim=1024, depth=6, heads=8, mlp_dim=2048):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features_SRM= nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)



        )

        self.features_edge =  nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ue = UNet(n_channels=3, n_classes=3)

        num_patches = (7 // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        patch_dim_srm = 28*28*64

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(32, 1, dim))
        self.pos_embedding_srm = nn.Parameter(torch.randn(32, 1, dim))
        self.pos_embedding_edg = nn.Parameter(torch.randn(32, 1, dim))


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_rm = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_edg = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.transformer_srm = Transformer(dim, 3, heads, mlp_dim)
        self.transformer_edg = Transformer(dim, 3, heads, mlp_dim)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.patchsrm_to_embedding = nn.Linear(patch_dim_srm, dim)
        self.patchedg_to_embedding = nn.Linear(patch_dim, dim)
        # self.patchedg_to_embedding = nn.Linear(patch_dim, dim)

        self.crossatt = CrossAttention(hidden_size=dim, dropout_rate=0.1, head_size=heads)

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )
        self.SRM = SRMConv2d_simple()

    def forward(self, img, mask=None):
        p = self.patch_size
        numpy_array = img.detach().cpu().numpy()
        rich_patches = []
        rgb_images = []
        sobel_images = []
        # 对数组进行迭代，逐个取出并转换为灰度图像
        for i in range(numpy_array.shape[0]):
            image = numpy_array[i]
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
            # 计算梯度的幅值，即特征图
            image_fre = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            # 计算通道中的最大值
            # max_value = np.amax(image_fre)
            # min_value = np.amin(image_fre)
            # 最小-最大缩放
            image_fre = image_fre / 255
            sobel_images.append(image_fre)

            image_rgb = cv2.resize(image, (224, 224))
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
            arr_transposed = image_rgb.transpose((2, 0, 1))
            # 然后，使用 np.expand_dims 增加批次维度
            arr_expanded = np.expand_dims(arr_transposed, axis=0)
            # 最后，将NumPy数组转换为PyTorch张量
            image_tensor = torch.tensor(arr_expanded).float() / 255
            image_tensor = image_tensor.squeeze(0)
            # image_tensor =train_transforms(img)
            # Get the patches from the image
            patches = get_patches(image_tensor)
            # Calculate the texture diversity for each patch
            diversities = texture_diversity(patches)
            # # Find the patch with the smallest texture diversity
            max_homogeneity_index = np.argmin(diversities)
            rich_patch = patches[max_homogeneity_index]
            rich_patch = rich_patch.unsqueeze(0).cuda()
            out = self.SRM(rich_patch).squeeze(0) # tensor 1,3,32,32
            # print(simplest_patch )
            rich_patches.append(out.detach().cpu().numpy())
            rgb_images.append(image_rgb)

            # 将灰度图像列表转换为张量，并增加维度为 (b, 1, 224, 224)
        img_rich_patches = torch.tensor(rich_patches).float().cuda()
        img_rgb_t = torch.tensor(rgb_images).cuda()
        img_rgb_t = img_rgb_t.permute(0, 3, 2, 1).float()
        img_fre_t = torch.tensor(sobel_images).unsqueeze(1).float().cuda()
        # print(img_fre_t.shape)

        x_rgb = self.features(img_rgb_t)
        #print(img_rich_patches.shape)
        x_srm = self.features_SRM(img_rich_patches)
        #print(x_srm.shape)
        x_edg = self.features_edge(img_fre_t)

        x_fre_fea, x4, x3, x2, x1 = self.ue.encoder(img_rich_patches)
        # print(x4.shape)
        noise_reco = self.ue.decoder(x_fre_fea, x4, x3, x2, x1)
        # print(x_edg.shape)

        y_srm = rearrange( x_srm, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=28, p2=28)
        y_edg = rearrange(x_edg, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        #print(y_srm.shape)
        y = rearrange(x_rgb, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        y = self.patch_to_embedding(y)

        y_srm = self.patchsrm_to_embedding(y_srm)
        #print(y_srm.shape)
        y_edg = self.patchedg_to_embedding(y_edg)

        cls_tokens = self.cls_token.expand(x_rgb.shape[0], -1, -1)
        cls_tokens_srm = self.cls_token_rm.expand(x_srm.shape[0], -1, -1)
        cls_tokens_edg  = self.cls_token_edg.expand(x_edg.shape[0], -1, -1)

        x_srm = torch.cat((cls_tokens_srm, y_srm), 1)
        x = torch.cat((cls_tokens, y), 1)
        x_edg = torch.cat((cls_tokens_edg, y_edg), 1)

        shape = x.shape[0]
        x_srm += self.pos_embedding_srm[0:shape]
        x += self.pos_embedding[0:shape]
        x_edg += self.pos_embedding[0:shape]
        x_srm = self.transformer_srm(x_srm, mask)
        x = self.transformer(x, mask)
        x_edg = self.transformer_edg(x_edg, mask)
        x_cls = self.to_cls_token(x[:, 0])
       # print(x_cls.unsqueeze(1).shape)
        x_srm_cls = self.to_cls_token(x_srm[:, 0])
        x_edg_cls = self.to_cls_token(x_edg[:, 0])
        a_n = self.crossatt(x_cls.unsqueeze(1),x_srm,x_srm)
        #print(a_n.shape)
        a_n_f = a_n +  x_cls.unsqueeze(1)
        a_e = self.crossatt(x_cls.unsqueeze(1),  x_edg,  x_edg)
        a_e_f = a_e + x_cls.unsqueeze(1)
        y= a_n_f+a_e_f
       # print(y.shape)
        y=y.squeeze(1)







       # y = x + x_srm+x_edg
        #print("y.shape",y.shape)

        return y#,img_rich_patches,noise_reco





# import torch
# from torch import nn
# from einops import rearrange
# from skimage import io, color
# import skimage
# from torchvision import transforms
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from parsing_extractor import CViT_parsing
#
# def initialize_weight(x):
#     nn.init.xavier_uniform_(x.weight)
#     if x.bias is not None:
#         nn.init.constant_(x.bias, 0)
# class SRMConv2d_simple(nn.Module):
#
#     def __init__(self, inc=3, learnable=False):
#         super(SRMConv2d_simple, self).__init__()
#         self.truc = nn.Hardtanh(-3, 3)
#         kernel = self._build_kernel(inc)  # (3,3,5,5)
#         self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
#         # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)
#
#     def forward(self, x):
#         '''
#         x: imgs (Batch, H, W, 3)
#         '''
#         out = F.conv2d(x, self.kernel, stride=1, padding=2)
#         out = self.truc(out)
#
#         return out
#
#     def _build_kernel(self, inc):
#         # filter1: KB
#         filter1 = [[0, 0, 0, 0, 0],
#                    [0, -1, 2, -1, 0],
#                    [0, 2, -4, 2, 0],
#                    [0, -1, 2, -1, 0],
#                    [0, 0, 0, 0, 0]]
#         # filter2：KV
#         filter2 = [[-1, 2, -2, 2, -1],
#                    [2, -6, 8, -6, 2],
#                    [-2, 8, -12, 8, -2],
#                    [2, -6, 8, -6, 2],
#                    [-1, 2, -2, 2, -1]]
#         # filter3：hor 2rd
#         filter3 = [[0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0],
#                    [0, 1, -2, 1, 0],
#                    [0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0]]
#
#         filter1 = np.asarray(filter1, dtype=float) / 4.
#         filter2 = np.asarray(filter2, dtype=float) / 12.
#         filter3 = np.asarray(filter3, dtype=float) / 2.
#         # statck the filters
#         filters = [[filter1],  # , filter1, filter1],
#                    [filter2],  # , filter2, filter2],
#                    [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
#         filters = np.array(filters)
#         filters = np.repeat(filters, inc, axis=1)
#         filters = torch.FloatTensor(filters)  # (3,3,5,5)
#         return filters
#
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x
#
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
#
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads=8):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim ** -0.5
#
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.to_out = nn.Linear(dim, dim)
#
#     def forward(self, x, mask=None):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x)
#         q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
#
#         dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#
#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value=True)
#             assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
#             mask = mask[:, None, :] * mask[:, :, None]
#             dots.masked_fill_(~mask, float('-inf'))
#             del mask
#
#         attn = dots.softmax(dim=-1)
#
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out
#
#
# class CrossAttention(nn.Module):
#     def __init__(self, dim=768, dropout_rate=0.1, head_size=8):
#         super(CrossAttention, self).__init__()
#
#         self.head_size = head_size
#         hidden_size = dim
#
#
#         self.att_size = att_size = hidden_size // head_size
#         self.scale = att_size ** -0.5
#
#         self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
#         self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
#         self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
#         initialize_weight(self.linear_q)
#         initialize_weight(self.linear_k)
#         initialize_weight(self.linear_v)
#
#         self.att_dropout = nn.Dropout(dropout_rate)
#
#         self.output_layer = nn.Linear(head_size * att_size, hidden_size,
#                                       bias=False)
#         initialize_weight(self.output_layer)
#
#     def forward(self, q, k, v, cache=None):
#         orig_q_size = q.size()
#         d_k = self.att_size
#         d_v = self.att_size
#         batch_size = q.size(0)
#         # mask = mask.bool()
#
#         # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
#         q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
#         # print("qshape",q.shape)
#         if cache is not None and 'encdec_k' in cache:
#             k, v = cache['encdec_k'], cache['encdec_v']
#         else:
#             k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
#             v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
#             # print("kshape", k.shape)
#             # print("vshape", v.shape)
#
#             if cache is not None:
#                 cache['encdec_k'], cache['encdec_v'] = k, v
#
#         q = q.transpose(1, 2)  # [b, h, q_len, d_k]
#         v = v.transpose(1, 2)  # [b, h, v_len, d_v]
#         k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
#         # print("qshape", q.shape)
#         # print("kshape", k.shape)
#         # print("vshape", v.shape)
#         # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
#         q.mul_(self.scale)
#         x = torch.matmul(q, k)  # [b, h, q_len, k_len] mask.unsqueeze(1).shape 1,1,77,77
#         # print("xshape",x.shape)
#         # x.masked_fill_(mask.unsqueeze(1), -1e9) #masked_fill_() 是 PyTorch 张量的原地操作之一，用于根据给定的掩码（mask）张量，在指定位置上将元素替换为指定的值。用value填充tensor中与mask中值为1位置相对应的元素
#         x = torch.softmax(x, dim=3)
#         x = self.att_dropout(x)
#         x = x.matmul(v)  # [b, h, q_len, attn]
#         # print(x.shape)
#
#         x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
#         x = x.view(batch_size, -1, self.head_size * d_v)
#
#         x = self.output_layer(x)
#
#         assert x.size() == orig_q_size
#         return x
#
#
# class Transformer_appnoise(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Residual(PreNorm(dim, Attention(dim, heads=heads))),
#                 CrossAttention(dim, head_size=heads),
#                 Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
#             ]))
#
#     def forward(self, x, noise, mask=None):
#         for attn, crossatt, ff in self.layers:
#             x = attn(x, mask=mask)
#             x_cross = crossatt(x, noise, noise,cache=None)
#             x=x+x_cross
#             x = ff(x)
#         return x
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim):
#
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Residual(PreNorm(dim, Attention(dim, heads=heads))),
#                 Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
#             ]))
#
#     def forward(self, x, mask=None):
#         for attn, ff in self.layers:
#             x = attn(x, mask=mask)
#             x = ff(x)
#         return x
#
# def texture_diversity(patches):
#     # 计算每个patch的GLCM纹理特征
#     homogeneity_scores = []
#     to_grayscale = transforms.Grayscale(num_output_channels=1)
#
#     # 将彩色图像张量转换为灰度图
#
#     for patch in patches:
#         # 如果Tensor在GPU上，先将其移动到CPU，然后转换为NumPy数组
#         patch = to_grayscale(patch)
#         patch = patch.squeeze()
#         #print(patch.shape)
#         patch = patch.cpu().numpy() if patch.is_cuda else patch.numpy()
#
#         # 然后，使用.astype(np.uint8)来改变数据类型
#         #patch_np_uint8 = (patch_np * 255).astype(np.uint8)
#         # 现在，你可以将`patch_np_uint8`用于`greycomatrix`函数
#        # glcm = greycomatrix(patch_np_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
#         # 计算GLCM
#         glcm = skimage.feature.graycomatrix((patch * 255).astype(np.uint8), distances=[1], angles=[0], levels=256,
#                                             symmetric=True,
#                                             normed=True)
#        # print("glcm",glcm)
#         # 计算同质性
#         homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0]
#        # print(" homogeneity",  homogeneity)
#         homogeneity_scores.append(homogeneity)
#     # flatten_patches = patches.reshape(patches.size(0), -1)
#     # std_dev = torch.std(flatten_patches, dim=1)
#     return homogeneity_scores
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)
#
#
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
#
# class Up(nn.Module):
#     """Upscaling then double conv"""
#
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#
#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#
#         self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # print("x1.shape",x1.shape) #1024,14,14
#         # print("x2.shape", x2.shape)
#         # input is CHW
#         diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#         diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
#
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
#
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         # self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024)
#
#         self.up1 = Up(1024, 512, bilinear=False)
#         self.up2 = Up(512, 256, bilinear=False)
#         self.up3 = Up(256, 128, bilinear=False)
#         self.up4 = Up(128, 64, bilinear=False)
#         self.outc = OutConv(64, n_classes)
#
#     def encoder(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3) # 512,14,14
#         x5 = self.down4(x4)  # 1024,7,7
#         # print("x5.shape",x5.shape)
#
#         return x5, x4, x3, x2, x1
#
#     def decoder(self,x5, x4, x3, x2, x1):
#         #print(x4.shape)
#
#         x = self.up1(x5, x4)
#         #print(x.shape)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#
#         return logits
# # Assuming image_tensor is your 3x224x224 input image tensor
# # image_tensor = torch.rand(3, 224, 224) # For example, random noise
#
# # Function to divide the image into 32x32 patches
# def get_patches(img_tensor, patch_size=112):
#     c, h, w = img_tensor.size()
#     img_tensor = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
#     img_tensor = img_tensor.contiguous().view(c, -1, patch_size, patch_size)
#     img_tensor = img_tensor.permute(1, 0, 2, 3)
#     return img_tensor
#
# class CViT(nn.Module):
#     def __init__(self, image_size=224, patch_size=7, num_classes=10, channels=512,
#                  dim=1024, depth=6, heads=8, mlp_dim=2048):
#         super().__init__()
#         assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
#
#         self.features = nn.Sequential(
#
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.features_SRM= nn.Sequential(
#
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#
#
#
#         )
#
#         self.features_edge =  nn.Sequential(
#
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(num_features=512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.ue = UNet(n_channels=3, n_classes=3)
#
#         num_patches = (7 // patch_size) ** 2
#         patch_dim = channels * patch_size ** 2
#         patch_dim_srm = 28*28*64
#
#         self.patch_size = patch_size
#
#         self.pos_embedding = nn.Parameter(torch.randn(32, 1, dim))
#         self.pos_embedding_srm = nn.Parameter(torch.randn(32, 1, dim))
#         self.pos_embedding_edg = nn.Parameter(torch.randn(32, 1, dim))
#
#
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.cls_token_rm = nn.Parameter(torch.randn(1, 1, dim))
#         self.cls_token_edg = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.transformer = Transformer_appnoise(dim, depth, heads, mlp_dim)
#         self.transformer_srm = Transformer(dim, 3, heads, mlp_dim)
#         self.transformer_edg = Transformer(dim, 3, heads, mlp_dim)
#
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.patchsrm_to_embedding = nn.Linear(patch_dim_srm, dim)
#         self.patchedg_to_embedding = nn.Linear(patch_dim, dim)
#
#         self.to_cls_token = nn.Identity()
#         self.mlp_head = nn.Sequential(
#             nn.Linear(dim, mlp_dim),
#             nn.ReLU(),
#             nn.Linear(mlp_dim, num_classes)
#         )
#         self.SRM = SRMConv2d_simple()
#
#     def forward(self, img, mask=None):
#         p = self.patch_size
#         numpy_array = img.detach().cpu().numpy()
#         rich_patches = []
#         rgb_images = []
#         sobel_images = []
#         # 对数组进行迭代，逐个取出并转换为灰度图像
#         for i in range(numpy_array.shape[0]):
#             image = numpy_array[i]
#             image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
#             sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
#             # 计算梯度的幅值，即特征图
#             image_fre = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
#             # 计算通道中的最大值
#             # max_value = np.amax(image_fre)
#             # min_value = np.amin(image_fre)
#             # 最小-最大缩放
#             image_fre = image_fre / 255
#             sobel_images.append(image_fre)
#
#             image_rgb = cv2.resize(image, (224, 224))
#             image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
#             arr_transposed = image_rgb.transpose((2, 0, 1))
#             # 然后，使用 np.expand_dims 增加批次维度
#             arr_expanded = np.expand_dims(arr_transposed, axis=0)
#             # 最后，将NumPy数组转换为PyTorch张量
#             image_tensor = torch.tensor(arr_expanded).float() / 255
#             image_tensor = image_tensor.squeeze(0)
#             # image_tensor =train_transforms(img)
#             # Get the patches from the image
#             patches = get_patches(image_tensor)
#             # Calculate the texture diversity for each patch
#             diversities = texture_diversity(patches)
#             # # Find the patch with the smallest texture diversity
#             max_homogeneity_index = np.argmin(diversities)
#             rich_patch = patches[max_homogeneity_index]
#             rich_patch = rich_patch.unsqueeze(0).cuda()
#             out = self.SRM(rich_patch).squeeze(0) # tensor 1,3,32,32
#             # print(simplest_patch )
#             rich_patches.append(out.detach().cpu().numpy())
#             rgb_images.append(image_rgb)
#
#             # 将灰度图像列表转换为张量，并增加维度为 (b, 1, 224, 224)
#         img_rich_patches = torch.tensor(rich_patches).float().cuda()
#         img_rgb_t = torch.tensor(rgb_images).cuda()
#         img_rgb_t = img_rgb_t.permute(0, 3, 2, 1).float()
#         img_fre_t = torch.tensor(sobel_images).unsqueeze(1).float().cuda()
#         # print(img_fre_t.shape)
#
#         x_rgb = self.features(img_rgb_t)
#         #print(img_rich_patches.shape)
#         x_srm = self.features_SRM(img_rich_patches)
#         #print(x_srm.shape)
#         x_edg = self.features_edge(img_fre_t)
#
#         x_fre_fea, x4, x3, x2, x1 = self.ue.encoder(img_rich_patches)
#         # print(x4.shape)
#         noise_reco = self.ue.decoder(x_fre_fea, x4, x3, x2, x1)
#         # print(x_edg.shape)
#
#         y_srm = rearrange( x_srm, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=28, p2=28)
#         y_edg = rearrange(x_edg, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
#         #print(y_srm.shape)
#         y = rearrange(x_rgb, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
#         y = self.patch_to_embedding(y)
#
#         y_srm = self.patchsrm_to_embedding(y_srm)
#         #print(y_srm.shape)
#         y_edg = self.patchedg_to_embedding(y_edg)
#
#         cls_tokens = self.cls_token.expand(x_rgb.shape[0], -1, -1)
#         cls_tokens_srm = self.cls_token_rm.expand(x_srm.shape[0], -1, -1)
#         cls_tokens_edg  = self.cls_token_edg.expand(x_edg.shape[0], -1, -1)
#
#         x_srm = torch.cat((cls_tokens_srm, y_srm), 1)
#         x = torch.cat((cls_tokens, y), 1)
#         x_edg = torch.cat((cls_tokens_edg, y_edg), 1)
#
#         shape = x.shape[0]
#         x_srm += self.pos_embedding_srm[0:shape]
#         # print(x_srm.shape)
#         x += self.pos_embedding[0:shape]
#         x_edg += self.pos_embedding[0:shape]
#         x_srm = self.transformer_srm(x_srm, mask)
#         x = self.transformer(x,x_srm[:, 0], mask)
#         x_edg = self.transformer_edg(x_edg, mask)
#         x = self.to_cls_token(x[:, 0])
#         x_srm = self.to_cls_token(x_srm[:, 0])
#         x_edg = self.to_cls_token(x_edg[:, 0])
#         y = x + x_srm+x_edg
#         #print("y.shape",y.shape)
#
#         return y#,img_rich_patches,noise_reco
#
