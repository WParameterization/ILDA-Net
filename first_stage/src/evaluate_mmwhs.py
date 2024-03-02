import cv2
import numpy as np
import torch as t
import torch
import pandas as pd
import os
from matplotlib import pyplot as plt
from utils.utils import load_nii, keep_largest_connected_components, to_categorical
from utils.timer import timeit
from PIL import Image
import cv2 as cv
from networks.vit_transformer_ import VisionTransformer
from networks.GAN import UncertaintyDiscriminator
from networks.vit_transformer import CONFIGS as CONFIGS_ViT_seg
import argparse

parser = argparse.ArgumentParser()
# general settings:
parser.add_argument("-bs", help="the batch size of training", type=int, default=2)
parser.add_argument("-ns", help="number of samples per epoch", type=int, default=1000)
parser.add_argument("-e", help="number of epochs to train", type=int, default=150)
parser.add_argument("-offdecay", help="whether not to use learning rate decay for unet", action='store_false')
parser.add_argument("-apdx", help="the appendix to the checkpoint", type=str, default='train_point_tf')
parser.add_argument("-he", help="whether to use He initializer", action='store_true')
parser.add_argument("-cvinit", help="whether to use constant variance initializer", action='store_true')
parser.add_argument("-multicuda", help="whether to use two cuda gpus", action='store_true')
parser.add_argument("-data_dir", help="the directory to the data", type=str,  default="/home/jidian/sdb1/jinlei/pointuda/input/")
# data augmentation:
parser.add_argument("-aug", help='the type of the augmentation, should be one of "", "heavy" or "light"', type=str,  default='')
parser.add_argument("-mh", help='turn on "histogram matching" (not needed)', action='store_true')
# unet setting:
parser.add_argument("-lr", help="the actual learning rate of the unet", type=float, default=1e-3)
parser.add_argument("-lr_fix", help="the base learning rate of the unet(used to be written in the 'appendix' string)", type=float, default=1e-3)
parser.add_argument("-sgd", help="whether to use sgd for the unet", action='store_true')
parser.add_argument("-nf", help="base number of the filters for the unet", type=int, default=32)
parser.add_argument("-out_ch", help="the out channels of the first conv layer(only used for SKUnet) (deprecated)",  type=int, default=32)
parser.add_argument("-drop", help="whether to apply dropout in the decoder of the unet", action='store_true')
parser.add_argument("-softmax", help="whether to apply softmax as the last activation layer of segmentation model",  default=True, action='store_true')
parser.add_argument("-dice", help="whether to use dice loss instead of jaccard loss", action='store_true')
parser.add_argument("-mmt", help="the value of momentum when using sgd", type=float, default=0.95)
# discriminator setting:
parser.add_argument("-d1", help="whether to apply outer space discriminator", action='store_true')
parser.add_argument("-d1lr", help="the learning rate for outer space discriminator", type=float, default=2.5e-5)
parser.add_argument("-ft", help="whether to apply feature transformation layer in pointcloudcls", action='store_true')
parser.add_argument("-dmmt", help="to set the momentum of the optimizers of the discriminators", type=float, default=0.95)
parser.add_argument("-d1mmt", help="to set the momentum of the optimizers of the discriminators", type=float,default=0.95)
parser.add_argument("-extd1", help="whether to extend output discriminator", action='store_true')
# weights of losses:
parser.add_argument("-dr", help="the ratio of the adversarial loss for the unet", type=float, default=.01)
parser.add_argument("-w1", help="the weight for the adversarial loss of unet to the output space discriminator", type=float, default=1.)
# 网络相关参数
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()
assert args.aug == '' or args.aug == 'heavy' or args.aug == 'light'

img_dir = "path/img/"
mask_dir= "path/mask/"

def img_mask(img_dir,mask_dir):

    files = os.listdir(img_dir)
    files.sort(key=lambda x: (x[:-4]))
    imgs = np.empty((len(files), 256, 256,3))
    masks = np.empty((len(files), 256, 256))

    for i in range(len(files)):


        img = cv.imread(img_dir + files[i])
        img=cv.resize(img,(256,256))
        img=np.array(img)
        mask=np.array(Image.open(mask_dir+files[i]), dtype=int)
        imgs[i] = np.array(img)

        masks[i]=np.array(mask)

    imgss = np.array(imgs, dtype=np.float32)
    masks=np.array(masks)

    imgss=np.transpose(imgss,(0,3,1,2))
    imgss = np.array(imgss, dtype=np.float32)


    return imgss ,masks


@timeit
def evaluate_segmentation(weight_dir='', unet_model=None, bs=2, save=False, model_name='', ifhd=True, ifasd=True):
    print("start to evaluate......")
    checkpoint = t.load(weight_dir)
    try:
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        print('load from dict')
    except:
        unet_model.load_state_dict(checkpoint)
        print('load from single state')
    print("model loaded")
    x_batch,mask=  img_mask(img_dir,mask_dir)

    pred = []
    for i in range(0, len(x_batch), bs):
        index = np.arange(i, min(i + bs, len(x_batch)))
        imgs = x_batch[index]
        pred1 = unet_model(torch.from_numpy(imgs).float().cuda())#_, _
        pred1 = pred1.cpu().detach().numpy()
        pred.append(pred1)

    pred = np.concatenate(pred, axis=0)
    pred = np.argmax(pred, axis=1)



    for j in range (pred.shape[0]):
        preddd=pred[j,:,:]
        mask1=mask[j,:,:]
        for m in range(0,256):
            for h in range(0,256):
                if preddd[m,h]==0.0:
                    preddd[m,h]=8.0
                elif preddd[m,h]==8.0:
                    preddd[m,h]=0.0
                else:
                    continue
        for e in range(0,256):
            for r in range(0,256):
                if mask1[e,r]==0.0:
                    mask1[e,r]=8.0
                elif mask1[e,r]==8.0:
                    mask1[e,r]=0.0
                else:
                    continue

        zhezhao=np.where(mask1==0.0,0.0,1.0)
        chu=np.zeros_like(zhezhao)
        for kk in range(0,256):
            for kkk in range(0,256):
                if preddd[kk,kkk]==mask1[kk,kkk]:
                    chu[kk,kkk]=1.0
                else:
                    continue

        print("image:",j,np.sum(chu*zhezhao),np.sum(zhezhao),(np.sum(chu*zhezhao)/np.sum(zhezhao)))#计算的准确率

        predd=np.expand_dims(preddd,axis=2)
        predfuben=np.concatenate([predd,predd,predd],axis=2)
        mask2=np.expand_dims(mask1,axis=2)
        mask3=np.concatenate([mask2,mask2,mask2],axis=2)
        cv.imwrite("path/pred_id/" + str(j) + "_pred.png", predd)
        cv.imwrite("path/label_id/" + str(j) + "_pred.png", mask2)
        #print(predfuben.shape,mask3.shape)
        #print(preddd.max())
        for k in range(0,256):
            for m in range(0,256):
                if preddd[k, m] ==0:
                    predfuben[k, m, 0] = 255.0
                    predfuben[k, m, 1] = 255.0
                    predfuben[k, m, 2] = 255.0
                elif preddd[k, m] ==1:
                    predfuben[k, m, 0] = 255
                    predfuben[k, m, 1] = 0
                    predfuben[k, m, 2] = 255
                elif preddd[k, m] ==2:
                    predfuben[k, m, 0] = 34
                    predfuben[k, m, 1] = 0
                    predfuben[k, m, 2] = 0
                elif preddd[k, m] ==3:
                    predfuben[k, m, 0] = 150
                    predfuben[k, m, 1] = 150
                    predfuben[k, m, 2] = 150
                elif preddd[k, m] ==4:
                    predfuben[k, m, 0] = 0
                    predfuben[k, m, 1] = 255
                    predfuben[k, m, 2] = 0
                elif preddd[k, m] ==5:
                    predfuben[k, m, 0] = 0
                    predfuben[k, m, 1] = 255
                    predfuben[k, m, 2] = 255
                elif preddd[k, m] ==6:
                    predfuben[k, m, 0] = 0
                    predfuben[k, m, 1] =34
                    predfuben[k, m, 2] = 0
                elif preddd[k, m] ==7:
                    predfuben[k, m, 0] = 255
                    predfuben[k, m, 1] = 255
                    predfuben[k, m, 2] = 0
                else:
                    predfuben[k, m, 0] = 0
                    predfuben[k, m, 1] = 0
                    predfuben[k, m, 2] = 255.0
        for p in range(0,256):
            for o in range(0,256):
                if mask2[p, o] ==0:
                    mask3[p, o, 0] = 255.0
                    mask3[p, o, 1] = 255.0
                    mask3[p, o, 2] = 255.0
                elif mask2[p, o] ==1:
                    mask3[p, o, 0] = 255
                    mask3[p, o, 1] = 0
                    mask3[p, o, 2] = 255
                elif mask2[p, o] ==2:
                    mask3[p, o, 0] = 34
                    mask3[p, o, 1] = 0
                    mask3[p, o, 2] = 0
                elif mask2[p, o] ==3:
                    mask3[p, o, 0] = 150
                    mask3[p, o, 1] = 150
                    mask3[p, o, 2] = 150
                elif mask2[p, o] ==4:
                    mask3[p, o, 0] = 0
                    mask3[p, o, 1] = 255
                    mask3[p, o, 2] = 0
                elif mask2[p, o] ==5:
                    mask3[p, o, 0] = 0
                    mask3[p, o, 1] = 255
                    mask3[p, o, 2] = 255
                elif mask2[p, o] ==6:
                    mask3[p, o, 0] = 0
                    mask3[p, o, 1] = 34
                    mask3[p, o, 2] = 0
                elif mask2[p, o] ==7:
                    mask3[p, o, 0] = 255
                    mask3[p, o, 1] = 255
                    mask3[p, o, 2] = 0
                else:
                    mask3[p, o, 0] = 0
                    mask3[p, o, 1] = 0
                    mask3[p, o, 2] = 255.0
        # plt.imshow(predfuben)
        # plt.show()

        predsave=np.concatenate([predfuben[:,:,2:3],predfuben[:,:,1:2],predfuben[:,:,0:1]],axis=2)

        mask4 = np.concatenate([mask3[:, :, 2:3], mask3[:, :, 1:2], mask3[:, :, 0:1]], axis=2)

        zhezhao=np.expand_dims(zhezhao,axis=2)
        zhezhaoo=np.concatenate([zhezhao,zhezhao,zhezhao],axis=2)

        all_save=np.hstack([predsave,mask4,zhezhaoo*255])
        cv.imwrite("path/pred/" + str(j) + "_pred.png", all_save)





if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-save", help='whether to save the evaluation result',default="True", action='store_true')
    parser.add_argument("-model_name", help="the name of the model", type=str, default='')
    parser.add_argument("-weight_dir", help="the path to the weight", type=str, default="path_weights")
    parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args()

    seed = 0
    np.random.seed(seed)
    t.manual_seed(seed)
    if args.weight_dir == '':
        model_names = {"unet": "rr.pt", "d1": "pp.pt", "d2": "pp", "d1d2": "pp.pt", "d4": "pp.pt","d2d4": "pp.pt", "d1d2d4": "pp.pt"}
        file_name = model_names[args.model_name]
        weight_dir = '../weights/' + file_name
    else:
        weight_dir = args.weight_dir
    toprint = "model: "
    if "d1lr" in weight_dir:
        toprint += "d1"
    pointnet = True if 'd4lr' in weight_dir else False
    extpn = True if 'extpn' in weight_dir else False
    extd1=True if 'extd1' in weight_dir else False
    extd2 = True if 'extd2' in weight_dir else False
    extd4 = True if 'extd4' in weight_dir else False

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    unet_model=VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

    if 'aug' in weight_dir:
        toprint += 'aug'
    if 'offmh' in weight_dir:
        toprint += '.offmh'
    if 'gn' in weight_dir:
        toprint += '.gn'
    if 'softmax' in weight_dir:
        toprint += '.softmax'
    if 'etpls' in weight_dir:
        toprint += '.etpls'
    if 'Tetpls' in weight_dir:
        toprint += '.Tetpls'
    if extpn:
        toprint += '.extpn'
    if extd1:
        toprint += '.extd1'
    if extd2:
        toprint += '.extd2'
    if extd4:
        toprint += '.extd4'
    if toprint != "":
        print(toprint)
    unet_model.cuda()
    unet_model.eval()
    evaluate_segmentation(weight_dir=weight_dir, unet_model=unet_model, save=args.save, model_name=args.model_name, ifhd=True, ifasd=True)
