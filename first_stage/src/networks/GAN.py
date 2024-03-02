import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
import copy
from networks.vit_transformer import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling_resnet_skip import gan_ResNetV2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=12, help='output channel of network')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--swin_name', type=str,default='/media/lenovo/新加卷/pointuda/model/swin_tiny_patch4_window7_224.pth',help='select one vit model')
args = parser.parse_args()

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class gan_VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(gan_VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.config = config

    def forward(self, x):

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
       # print("aaaa", x.shape)
        x, attn_weights= self.transformer(x)
       # print("aaaaaaaaa")# (B, n_patch, hidden)#encoder先trans再decoder 然后再segmentation——head
        x = self.decoder(x, features=None)#decoder1*512*32*32
        return x#这是最后输出结果


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = gan_Embeddings(config, img_size=img_size)
        self.encoder = gan_Encoder(config, vis)
        self.decoder = DecoderCup(config)

    def forward(self, input_ids):#先transformer在encorder
        #print(input_ids.shape)#*9*224*224
        embedding_output= self.embeddings(input_ids)#特征提取层
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights

class gan_Embeddings(nn.Module):#先划分patch，然后在transformer
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(gan_Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            #print("aaaaaaaaaa")
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
        self.hybrid_model = gan_ResNetV2(block_units=config.resnet.num_layers,width_factor=config.resnet.width_factor)  # 先卷积
        in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels, out_channels=config.hidden_size, kernel_size=patch_size, stride=patch_size)
        #print(config.hidden_size,patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
      #  print(n_patches)
        self.dropout = Dropout(config.transformer["dropout_rate"])
    def forward(self, x):
        #rint(type(x))
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
       # print(x)# (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        #print(x.shape)#(2*50176*768)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings




class gan_Encoder(nn.Module):
    def __init__(self, config, vis):
        super(gan_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):#transformer，每层是两个，一共是24个
            layer = gan_Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)#最后又进行了一次layer normal
        return encoded, attn_weights
class gan_Block(nn.Module):
    def __init__(self, config, vis):
        super(gan_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = gan_Mlp(config)
        self.attn = gan_Attention(config, vis)
    def forward(self, x):#transformer 块    一个block块是两个attention
        h = x
        x = self.attention_norm(x)#先layer_normal在进行attention
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights
class gan_Attention(nn.Module):
    def __init__(self, config, vis):
        super(gan_Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class gan_Mlp(nn.Module):
    def __init__(self, config):
        super(gan_Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        #print(x.shape,skip.shape) 最后两个输入不一样
        if skip is not None:
            x = torch.cat([x, skip], dim=1)#跳跃连接的地方
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size,head_channels, kernel_size=3, padding=1,use_batchnorm=True,)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        skip_channels=[0,0,0,0]

        blocks = [DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)#转置
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)#卷积#1*512*32*32
        # for i, decoder_block in enumerate(self.blocks):
        #     if features is not None:#这个feature就是需要连接的东西
        #         skip = features[i] if (i < self.config.n_skip) else None#feature就是刚才的resnet提取的东西
        #     else:
        #         skip = None
        #     x = decoder_block(x, skip=skip)
        return x





class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()

        filter_num_list = [4096, 2048, 1024, 1]

        self.fc1 = nn.Linear(24576, filter_num_list[0])
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(filter_num_list[0], filter_num_list[1])
        self.fc3 = nn.Linear(filter_num_list[1], filter_num_list[2])
        self.fc4 = nn.Linear(filter_num_list[2], filter_num_list[3])

        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    # m.bias.data.copy_(1.0)
                    m.bias.data.zero_()


    def forward(self, x):

        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = self.fc4(x)
        return x

class Conv2dReLU(nn.Sequential):
    def __init__( self,in_channels, out_channels, kernel_size,padding=0,stride=1, use_batchnorm=True,):
        conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=not (use_batchnorm),)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)

trans=gan_VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

class OutputDiscriminator(nn.Module):
    def __init__(self, in_channel=2, softmax=False, init=False):
        super(OutputDiscriminator, self).__init__()
        self._softmax = softmax
        filter_num_list = [64, 128, 256, 512, 1]
        self.upsample = nn.UpsamplingBilinear2d(size=(224, 224))
        self.conv1 = nn.Conv2d(in_channel, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.sigmoid = nn.Sigmoid()
        if init:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.upsample(x)
        if self._softmax:
            x = F.softmax(x, dim=1)
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        return x

#越卷积图像越小
class UncertaintyDiscriminator(nn.Module):
    def __init__(self, in_channel=2, heinit=False, ext=False):
        # assert not(softmax and sigmoid), "Only one of 'softmax' or 'sigmoid' can be used for activation function."
        super(UncertaintyDiscriminator, self).__init__()
        # self._softmax = softmax
        # self._sigmoid = sigmoid
        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(in_channel, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)#9-64
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)#64-128
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)#128-256
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)#256-512
        if ext:
            self.conv4_2 = nn.Conv2d(filter_num_list[3], 1024, kernel_size=3, stride=2, padding=1, bias=False)#512-1024
            self.conv4_3 = nn.Conv2d(1024, filter_num_list[2], kernel_size=3, stride=2, padding=1, bias=False)#1024-256
            self.conv5 = nn.Conv2d(filter_num_list[2], filter_num_list[4], kernel_size=4, stride=2, padding=2,  bias=False)#256-1
        else:
            self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self._ext = ext
        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights(heinit=heinit)


    def _initialize_weights(self, heinit=False):
        if heinit:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    prod = float(np.prod(m.weight.size()[1:]))
                    prod = np.sqrt(2 / prod)
                    m.weight.data.normal_(0.0, prod)
                    if m.bias is not None:
                        m.bias.data.zero_()
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()


    def forward(self, x):
        #print(x.shape)
        # if self._softmax:
        #     x = F.softmax(x, dim=1)
        # elif self._sigmoid:
        #     x = F.sigmoid(x)
        #print(x.shape)
        #x=trans(x)
        #print(x.shape)#torch.Size([2, 512, 14, 14])
        x = self.leakyrelu(self.conv1(x))
        #print(x.shape)
        x = self.leakyrelu(self.conv2(x))
        #print(x.shape)
        x = self.leakyrelu(self.conv3(x))
       # print(x.shape)
        x = self.leakyrelu(self.conv4(x))#6*512*15*15
       # print(x.shape)



        #print(x.shape)#6*512*15*15
        if self._ext:
            x = self.leakyrelu(self.conv4_2(x))
           # print(x.shape)#6*1024*8*8
            x = self.leakyrelu(self.conv4_3(x))
          #  print(x.shape)

        # #print(x.shape)
        x = self.conv5(x)
       # print(x.shape)
        #8*256*5*5
        #print(x.shape)
        return x


class BoundaryDiscriminator(nn.Module):
    def __init__(self, ):
        super(BoundaryDiscriminator, self).__init__()

        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(1, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        return x

class BoundaryEntDiscriminator(nn.Module):
    def __init__(self, ):
        super(BoundaryEntDiscriminator, self).__init__()

        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(3, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        return x

if __name__ == '__main__':
    model_dis = UncertaintyDiscriminator(in_channel=2).cuda()
    img = torch.rand((1, 2, 1024, 1024)).cuda()
    output = model_dis(img)
    print(output.size())
