import numpy as np
import torch as t
import torch
import cv2 as cv
import os.path as osp
import shutil
import torch.nn as nn
from networks.vit_transformer_ import VisionTransformer
from data_generator_st import  DataGenerator_PointNet, ImageProcessor
from networks.vit_transformer import CONFIGS as CONFIGS_ViT_seg

import os

import st_utils
from PIL import Image
import math
import time


SAVE_PATH = 'debug'
LOG_FILE = 'self_training_log'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-model_name", help="the name of the model", type=str, default='')
parser.add_argument("-d4", help='whether to use pointcloud discriminator', default=False, action='store_true')
parser.add_argument("-d4aux", help="whether to learn point cloud generator (often used to train pointnet when pointcloud discriminator is not applied)", default=False, action='store_true')
parser.add_argument("-extpn", help="whether to extend pointcloud generator(pointnet)", default=False, action='store_true')  #
parser.add_argument("-batch_size", help="batch_size",type= int ,default=1)
parser.add_argument("-test_num", help="test_num",type= int ,default=900)
parser.add_argument("-tgt_num", help="tgt_num",type= int ,default=900)
parser.add_argument("-num_classes", help="test_num",type= int ,default=9)
parser.add_argument("--save", type=str, default=SAVE_PATH, help="Path to save result for self-training.")
parser.add_argument('--init-tgt-port', default=0.5, type=float, dest='init_tgt_port', help='The initial portion of target to determine kc')
parser.add_argument("-data_dir", help="the directory to the data", type=str, default="path/input_vit/")
parser.add_argument("-mh", help='turn on "histogram matching" (not needed)', action='store_true')
parser.add_argument("-aug", help='the type of the augmentation, should be one of "", "heavy" or "light"', type=str, default='')
parser.add_argument('--debug', help='True means logging debug info.', default=False, action='store_true')
parser.add_argument("--log-file", type=str, default=LOG_FILE, help="The name of log file.")
parser.add_argument('--rm-prob', help='If remove the probability maps generated in every round.',default=False, action='store_true')
parser.add_argument("--data-tgt-train-list", type=str, default="path", help="Path to the file listing the images*GT labels in the target train dataset.")
parser.add_argument("--data-src-list", type=str, default="path/source_train_list_st.csv", help="Path to the file listing the images&labels in the source dataset.")
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

args = parser.parse_args()

logger = st_utils.set_logger(args.save, args.log_file, args.debug)
logger.info('start with arguments %s', args)

def parse_split_list(list_name):
    image_list = []
    image_name_list = []
    file_num = 0
    with open(list_name) as f:
        for item in f.readlines():
            fields = item.strip()
            image_name = fields.split('/')[-1]
            image_list.append(fields)
            image_name_list.append(image_name)
            file_num += 1
    return image_list, image_name_list, file_num

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def savelst_SrcTgt(image_tgt_list, image_name_tgt_list, image_src_list, save_lst_path, save_pseudo_label_path, src_num, tgt_num, args):
    src_train_lst = osp.join(save_lst_path,'src_train.txt')
    tgt_train_lst = osp.join(save_lst_path, 'tgt_train.txt')

    with open(src_train_lst, 'w') as f:
        for idx in range(src_num):
            f.write("%s\n" % (image_src_list[idx]))

    with open(tgt_train_lst, 'w') as f:
        for idx in range(tgt_num):
            image_tgt_path = osp.join(save_pseudo_label_path,image_name_tgt_list[idx])
            f.write("%s\t%s\n" % (image_tgt_list[idx], image_tgt_path))

    return src_train_lst, tgt_train_lst, src_num



def label_selection(cls_thresh,  round_idx, save_prob_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args, logger,id):
    logger.info('###### Start pseudo-label generation in round {} ! ######'.format(round_idx))
    start_pl = time.time()
    print(id)
    for idx in range(0,len(id)):

        prob_map_path = osp.join(save_prob_path, '{}_depth.npy'.format(str(id[idx][0][:-4])))
        print(prob_map_path)

        pred_prob = np.load(prob_map_path)

        save_wpred_vis_path = osp.join(save_round_eval_path, 'weighted_pred_vis')
        if not os.path.exists(save_wpred_vis_path):
            os.makedirs(save_wpred_vis_path)

        weighted_prob = pred_prob/cls_thresh
        print("weighted_probweighted_prob",weighted_prob)#
        weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=2), dtype=np.uint8)

        wpred_label_col = weighted_pred_trainIDs.copy()
        wpred_label_col = colorize_mask(wpred_label_col)
        wpred_label_col.save('%s/%s_color.png' % (save_wpred_vis_path, str(idx).zfill(4)))
        weighted_conf = np.amax(weighted_prob, axis=2)
        pred_label_trainIDs = weighted_pred_trainIDs.copy()
        pred_label_trainIDs[weighted_conf < 1] = 255


        pseudo_label_trainIDs = pred_label_trainIDs.copy()

        pseudo_label_col = colorize_mask(pseudo_label_trainIDs)
        pseudo_label_col.save('%s/%s_color.png' % (save_pseudo_label_color_path, str(id[idx][0][:-4])))

        pseudo_label_save = Image.fromarray(pseudo_label_trainIDs.astype(np.uint8))
        pseudo_label_save.save('%s/%s.png' % (save_pseudo_label_path, str(id[idx][0][:-4])))

    if args.rm_prob:
        shutil.rmtree(save_prob_path)

    logger.info('###### Finish pseudo-label generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_pl))

def kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx,  args, logger):
    logger.info('###### Start kc generation in round {} ! ######'.format(round_idx))
    start_kc = time.time()

    cls_thresh = np.ones(args.num_classes,dtype = np.float32)
    cls_sel_size = np.zeros(args.num_classes, dtype=np.float32)
    cls_size = np.zeros(args.num_classes, dtype=np.float32)

    for idx_cls in np.arange(0, args.num_classes):
        cls_size[idx_cls] = pred_cls_num[idx_cls]
        if conf_dict[idx_cls] != None:
            conf_dict[idx_cls].sort(reverse=True)
            len_cls = len(conf_dict[idx_cls])

            cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
            len_cls_thresh = int(cls_sel_size[idx_cls])

            if len_cls_thresh != 0:
            	if conf_dict[idx_cls][len_cls_thresh-1]<0.9:
            		cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
            	else:
            		cls_thresh[idx_cls] = 0.9
            conf_dict[idx_cls] = None
    # save thresholds #阈值
    np.save("path/pseudo_label/" + '/cls_thresh_round' + str(round_idx) + '.npy', cls_thresh)
    np.save("path/pseudo_label/" + '/cls_sel_size_round' + str(round_idx) + '.npy', cls_sel_size)
    logger.info('###### Finish kc generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_kc))
    return cls_thresh

class ScoreUpdater(object):
    def __init__(self, c_num, x_num, logger=None, label=None, info=None):
        self._confs = np.zeros((c_num, c_num))
        self._per_cls_iou = np.zeros(c_num)
        self._logger = logger
        self._label = label
        self._info = info
        self._num_class = c_num
        self._num_sample = x_num

    @property
    def info(self):
        return self._info

    def reset(self):
        self._start = time.time()
        self._computed = np.zeros(self._num_sample) # one-dimension
        self._confs[:] = 0

    def fast_hist(self,label, pred_label, n):
        k = (label >= 0) & (label < n)

        return np.bincount(n * label[k].astype(int) + pred_label[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self,hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def do_updates(self, conf, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._per_cls_iou = self.per_class_iu(conf)

    def update(self, pred_label, label, i, computed=True):
        conf = self.fast_hist(label, pred_label, self._num_class)
        self._confs += conf
        self.do_updates(self._confs, i, computed)
        self.scores(i)

    def scores(self, i=None, logger=None):
        x_num = self._num_sample
        ious = np.nan_to_num( self._per_cls_iou )
        print(ious)

        logger = self._logger if logger is None else logger
        if logger is not None:
            if i is not None:
                speed = 1. * self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}mean iou: {:.2f}%'. format(name, np.mean(ious) * 100))

            with st_utils.np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(ious * 100))

        return ious

def pred_to_color_test(pred):
    for j in range (pred.shape[0]):
        print("test_save", j)
        preddd=pred[j,:,:]
        predd=np.expand_dims(preddd,axis=2)
        predfuben=np.concatenate([predd,predd,predd],axis=2)
        for k in range(0,256):
            for m in range(0,256):
                if preddd[k, m] ==0:
                    predfuben[k, m, 0] = 0.0
                    predfuben[k, m, 1] = 0.0
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
                    predfuben[k, m, 2] = 0
        predsave=np.concatenate([predfuben[:,:,2:3],predfuben[:,:,1:2],predfuben[:,:,0:1]],axis=2)
        cv.imwrite("path/pseudo_label/" +"testset_vis/"+ str(j).zfill(4) + "_depth_color.png", predsave)
def pred_to_color_val(pred):
    for j in range (pred.shape[0]):
        print("val_save",j)
        preddd=pred[j,:,:]
        predd=np.expand_dims(preddd,axis=2)
        predfuben=np.concatenate([predd,predd,predd],axis=2)
        for k in range(0,256):
            for m in range(0,256):
                if preddd[k, m] ==0:
                    predfuben[k, m, 0] = 0.0
                    predfuben[k, m, 1] = 0.0
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
                    predfuben[k, m, 2] = 0
        predsave=np.concatenate([predfuben[:,:,2:3],predfuben[:,:,1:2],predfuben[:,:,0:1]],axis=2)
        cv.imwrite("path/pseudo_label/" +"pred_vis/"+ str(j).zfill(4) + "_depth_color.png", predsave)

def get_generators(ids_train, batch_size=1, n_samples=2000, crop_size=0, mh=False):
    test_tar = DataGenerator_PointNet(df=ids_train, channel="channel_first", phase="train",batch_size=batch_size, source="target", crop_size=crop_size, n_samples=n_samples, match_hist=mh, ifvert=args.d4 or args.d4aux,aug=args.aug, data_dir=args.data_dir)
    return iter(test_tar)

def st_test(weight_dir='', unet_model=None,trainA_iterator=None):

    scorer = ScoreUpdater(args.num_classes, args.test_num, logger)
    scorer.reset()

    checkpoint = t.load(weight_dir)
    try:
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        print('load from dict')
    except:
        unet_model.load_state_dict(checkpoint)
        print('load from single state')
    print("model loaded")
    unet_model.eval()
    unet_model.cuda()
    pred=[]
    label=[]
    with torch.no_grad():
        for imgA, maskA, vertexA in trainA_iterator:

            pred1, _, _ = unet_model(torch.from_numpy(imgA).float().cuda())
            pred1 = pred1.cpu().detach().numpy()
            pred.append(pred1)
            label.append(maskA)
        pred = np.concatenate(pred, axis=0)
        label=np.concatenate(label,axis=0)
        label=np.asarray(np.argmax(label, axis=1),dtype=np.uint8)
        pred = np.asarray(np.argmax(pred, axis=1),dtype=np.uint8)
        pred_label = pred.copy()
        for l in range(0,label.shape[0]):

            scorer.update(pred_label.flatten(), label.flatten(), l)

        pred_to_color_test(pred)

def val(weight_dir='', unet_model=None,trainA_iterator=None):

    conf_dict = {k: [] for k in range(args.num_classes)}
    pred_cls_num = np.zeros(args.num_classes)

    softmax2d = nn.Softmax2d()
    scorer = ScoreUpdater(args.num_classes, args.test_num, logger)
    scorer.reset()
    checkpoint = t.load(weight_dir)
    try:
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        print('load from dict')
    except:
        unet_model.load_state_dict(checkpoint)
        print('load from single state')
    print("model loaded")
    #print(unet_model)
    unet_model.eval()
    unet_model.cuda()
    pred = []
    label = []
    output1=[]
    conff=[]
    id=[]
    with torch.no_grad():
        for imgA, maskA, lineA ,lines, path in trainA_iterator:
           # imgA = imgA[:, 0:1, :, :]
            #print(path[0])
            maskA = (np.argmax(maskA, axis=1))

            prediction,line= unet_model(torch.from_numpy(imgA).float().cuda(),torch.from_numpy(lines).float().cuda())
            output = softmax2d(prediction).cpu().numpy()
            amax_output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
            conf = np.amax(output, axis=1)
            pred.append(amax_output)
            label.append(maskA)
            output1.append(output)
            conff.append(conf)
            id.append(path)

        pred = np.concatenate(np.array(pred), axis=0)

        label = np.concatenate(label, axis=0)
        output1=np.concatenate(output1, axis=0)
        conff=np.concatenate(conff, axis=0)


        pred_label = pred.copy()
        for p in range(0,label.shape[0]):
           print(pred_label.shape,label.shape)
           scorer.update(pred_label[p,:,:].flatten(), label[p,:,:].flatten(), p)

        pred_to_color_val(pred)

        for m in range (0,len(id)):
            pred_temp = Image.fromarray(pred[m,:,:])
            pred_temp.save("path/pseudo_label/"+"pred/"+str(id[m][0][:-4])+"_depth.png")

        output1=np.array(output1)
        output1 = output1.transpose(0,2,3,1)
        for l in range(0,len(id)):
            print("val_save_prob", l)
            prob_temp=output1[l,:,:,:]
            np.save("path/pseudo_label/"+"prob/"+str(id[l][0][:-4])+"_depth.npy",prob_temp)

        # save class-wise confidence maps
        for y in range(0,pred.shape[0]):
            for idx_cls in range(args.num_classes):
                confff=conff[y,:,:]
                idx_temp = pred[y,:,:] == idx_cls

                pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                if idx_temp.any():

                    conf_cls_temp = confff[idx_temp].astype(np.float32)

                    len_cls_temp = conf_cls_temp.size

                    conf_cls = conf_cls_temp[0:len_cls_temp:4]#
                    conf_dict[idx_cls].extend(conf_cls)
    return conf_dict, pred_cls_num,id

ct_train = ImageProcessor.split_data(os.path.join(args.data_dir, "ct_train_list_st.csv"))
model_name="weights_path"
trainA_iterator = get_generators(ct_train, batch_size=args.batch_size,n_samples=-1,  crop_size=0, mh=args.mh)

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
unet_model = VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)


image_src_list, _, src_num = parse_split_list(args.data_src_list)
image_tgt_list, image_name_tgt_list, tgt_num = parse_split_list(args.data_tgt_train_list)

save_prob_path="path/prob/"
save_pred_path="path/pred/"
save_pseudo_label_path="path/pseudo_label/"
save_pseudo_label_color_path="path/pseudo_label_color/"
save_round_eval_path="path/pseudo_label/3/"
save_lst_path="path/list/"

conf_dict, pred_cls_num,id=val(model_name,unet_model,trainA_iterator)

cls_thresh = kc_parameters(conf_dict, pred_cls_num, args.init_tgt_port, 3, args, logger)

label_selection(cls_thresh, args.tgt_num, image_name_tgt_list, 3, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args, logger,id)



