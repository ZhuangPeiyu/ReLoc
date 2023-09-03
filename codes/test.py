import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from models.unet import SCSEUnet
import denseFCN
import random
import time
import logging
import pickle
from sklearn.metrics import roc_auc_score
from metrics import get_metrics
from torch.nn.modules.loss import _Loss
from SCUNet_main.models import network_scunet
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from MVSS_net.models.mvssnet import get_mvss
from skimage import io

def create_dir(path):
    if(os.path.exists(path)==False):
        os.makedirs(path)
class argparse():
    pass


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
args = argparse()
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
args.mode = 'test'
args.quality = 75
args.loc_patch_size = None
args.aug = True
args.loss = 'bce'
args.model_type = 'MVSS_net' #SCSEUnet,denseFCN, MVSS_net
args.bn_in = 'bn'
args.test_images_split = "/data2/zhuangpy/tamperingLocalization/ReLoc/Datasets/DEFACTO/mix_all/val_images.pickle"
args.test_image_path = "/data2/zhuangpy/tamperingLocalization/ReLoc/Datasets/DEFACTO/mix_all/tamper/"
args.test_mask_path = "/data2/zhuangpy/tamperingLocalization/ReLoc/Datasets/DEFACTO/mix_all/mask/"
if args.model_type == 'denseFCN':
    args.restore_path = "../checkpoints/denseFCN/QF75/Restore_model/restore_model.pth"
    args.loc_restore_path = "../checkpoints/denseFCN/QF75/Loc_model/loc_model.pth"
elif args.model_type == 'SCSEUnet':
    args.restore_path = "../checkpoints/SCSEUnet/QF75/Restore_model/restore_model.pth"
    args.loc_restore_path = "../checkpoints/SCSEUnet/QF75/Loc_model/loc_model.pth"
elif args.model_type == 'MVSS_net':
    args.restore_path = "../checkpoints/MVSSNet/QF75/Restore_model/restore_model.pth"
    args.loc_restore_path = "../checkpoints/MVSSNet/QF75/Loc_model/loc_model.pth"
args.save_result_path = "../results/"+args.model_type+"/"+"/DEFACTO/QF_" + str(args.quality)
IMG_SIZE = 512
image_step = 512

create_dir(args.save_result_path)
create_dir(os.path.join(args.save_result_path,'figs'))

def create_file_list(image_path,mask_path,image_file):
    if(image_file!=''):
        with open(image_file,'rb') as f:
            images = pickle.load(f)
    else:
        images = os.listdir(image_path)
    files = []
    random.shuffle(images)
    for image in images:
        if(mask_path!=''):
            mask_name = image
            files.append([os.path.join(image_path, image), os.path.join(mask_path, mask_name)])
        else:
            files.append(os.path.join(image_path,image))
    return files

class Tampering_Dataset(Dataset):
    def __init__(self, file,choice='train',patch_size=768,with_aug = False,compress_quality = 75):
        self.patch_size = patch_size
        self.choice = choice
        self.filelist = file
        self.with_aug = with_aug
        self.compress_quality = compress_quality
    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return len(self.filelist)


    def load_item(self, idx):

        fname1, fname2 = self.filelist[idx]
        img = cv2.imread(fname1)
        # mask_zeros = np.zeros([H, W, 3])
        if fname2 == '':
            mask = np.zeros([H, W, 3])
        else:
            mask = cv2.imread(fname2)
        if self.choice == 'test':
            h, w = img.shape[0], img.shape[1]
            if (h % 32 != 0 or w % 32 != 0):
                img = img[:h // 32 * 32, :w // 32 * 32, :]
                mask = mask[:h // 32 * 32, :w // 32 * 32, :]

            result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.compress_quality])
            degraded_img = cv2.imdecode(encimg, 1)

        img = img[:,:,::-1].astype('float') / 255.
        degraded_img = degraded_img[:,:,::-1].astype('float') / 255.
        mask = mask.astype('float')
        mask_copy = np.copy(mask)
        mask[np.where(mask_copy < 127.5)] = 0
        mask[np.where(mask_copy >= 127.5)] = 1
        return self.tensor(img), self.tensor(mask[:, :, :1]), self.tensor(degraded_img),fname1.split('/')[-1]
    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)

def compute_metrics(image_batch,label_batch,outputs_batch,f1_all,iou_all,auc_all,directly_predict = None):
    # #计算每一个batch里面每一张图的F1和AUC
    image_batch = image_batch.cpu().detach().numpy()
    label_batch = label_batch.cpu().detach().numpy()
    outputs_batch = outputs_batch.cpu().detach().numpy()
    for image, label, predict_map in zip(image_batch, label_batch, outputs_batch):
        predict_map = predict_map[0,:,:]
        label = label[0,:,:]
        predict_threshold = np.copy(predict_map)
        predict_threshold[np.where(predict_map<0.5)] = 0
        predict_threshold[np.where(predict_map>=0.5)] = 1

        try:
            tpr_recall, tnr, precision, f1, mcc, iou, tn, tp, fn, fp = get_metrics(predict_threshold, label)
            auc = roc_auc_score(label.reshape(-1, ), predict_map.reshape(-1, ))
            f1_all.append(f1)
            auc_all.append(auc)
            iou_all.append(iou)
        except Exception as e:
            print(e)
            continue

    return f1_all,iou_all,auc_all


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
def read_image_block_with_mask(images,IMG_SIZE,step):
    test_blocks = []
    row,col = images.shape[2],images.shape[3]
    if(images.shape[1]==3):
        for i in np.arange(0, row - IMG_SIZE, step):
            for j in np.arange(0, col - IMG_SIZE, step):
                test_blocks.append(images[:,:,i:int(i + IMG_SIZE), j:int(j + IMG_SIZE)])
            test_blocks.append(images[:,:,i:int(i + IMG_SIZE), int(col - IMG_SIZE):col])
        for j in np.arange(0, int(col - IMG_SIZE), step):
            test_blocks.append(images[:,:,int(row - IMG_SIZE):row, j:int(j + IMG_SIZE)])
        test_blocks.append(images[:,:,row - IMG_SIZE:row, col - IMG_SIZE:col])
    return test_blocks,row,col

def consist_whole_mask(test_results,row,col,step,IMG_SIZE,is_threshold = False):
    mask = torch.zeros((1,1,row,col),device=test_results[0].device)
    num_every_pixel_scan = torch.zeros((1,1,row,col),device=test_results[0].device)
    count = 0
    for i in np.arange(0,row - IMG_SIZE,step):
        for j in np.arange(0,col-IMG_SIZE,step):
            mask[:,:,i:int(i+IMG_SIZE),j:int(j+IMG_SIZE)] += test_results[count]
            num_every_pixel_scan[:,:,i:int(i+IMG_SIZE),j:int(j+IMG_SIZE)] += 1
            count+=1
        mask[:,:,i:int(i+IMG_SIZE),int(col-IMG_SIZE):col] += test_results[count]
        num_every_pixel_scan[:,:,i:int(i + IMG_SIZE), int(col - IMG_SIZE):col] += 1
        count += 1
    for j in np.arange(0,int(col-IMG_SIZE),step):
        mask[:,:,int(row-IMG_SIZE):row,j:int(j+IMG_SIZE)] += test_results[count]
        num_every_pixel_scan[:,:,int(row-IMG_SIZE):row,j:int(j+IMG_SIZE)] +=1
        count+=1
    mask[:,:,row-IMG_SIZE:row,col-IMG_SIZE:col] += test_results[count]
    num_every_pixel_scan[:,:,row-IMG_SIZE:row,col-IMG_SIZE:col] += 1
    mask = mask/num_every_pixel_scan
    if(is_threshold==True):
        mask_copy = np.copy(mask)
        mask[np.where(mask_copy<0.5)] = 0
        mask[np.where(mask_copy >= 0.50)] = 1

    return mask

def consist_whole_restore_image(test_results,row,col,step,IMG_SIZE):
    mask = torch.zeros((1,3,row,col),device=test_results[0].device)
    num_every_pixel_scan = torch.zeros((1,3,row,col),device=test_results[0].device)
    count = 0
    for i in np.arange(0,row - IMG_SIZE,step):
        for j in np.arange(0,col-IMG_SIZE,step):
            mask[:,:,i:int(i+IMG_SIZE),j:int(j+IMG_SIZE)] += test_results[count]
            num_every_pixel_scan[:,:,i:int(i+IMG_SIZE),j:int(j+IMG_SIZE)] += 1
            count+=1
        mask[:,:,i:int(i+IMG_SIZE),int(col-IMG_SIZE):col] += test_results[count]
        num_every_pixel_scan[:,:,i:int(i + IMG_SIZE), int(col - IMG_SIZE):col] += 1
        count += 1
    for j in np.arange(0,int(col-IMG_SIZE),step):
        mask[:,:,int(row-IMG_SIZE):row,j:int(j+IMG_SIZE)] += test_results[count]
        num_every_pixel_scan[:,:,int(row-IMG_SIZE):row,j:int(j+IMG_SIZE)] +=1
        count+=1
    mask[:,:,row-IMG_SIZE:row,col-IMG_SIZE:col] += test_results[count]
    num_every_pixel_scan[:,:,row-IMG_SIZE:row,col-IMG_SIZE:col] += 1
    mask = mask/num_every_pixel_scan
    return mask

if args.mode == 'test':
    test_file = create_file_list(args.test_image_path, args.test_mask_path, args.test_images_split)

    test_dataset = Tampering_Dataset(test_file, choice='test', patch_size=args.loc_patch_size,compress_quality=args.quality)

    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False, pin_memory=False, num_workers=1, drop_last=False)

    restore_model = network_scunet.SCUNet(in_nc=3)
    # loc_model = denseFCN.normal_denseFCN(bn_in=args.bn_in)
    if(args.model_type=="denseFCN"):
        loc_model = denseFCN.normal_denseFCN(bn_in=args.bn_in).cuda()
    elif(args.model_type=='SCSEUnet'):
        loc_model = SCSEUnet(backbone_arch='senet154').cuda()
    elif (args.model_type == 'MVSS_net'):
        loc_model = get_mvss(backbone='resnet50',
                             pretrained_base=True,
                             nclass=1,
                             sobel=True,
                             constrain=True,
                             n_input=3).cuda()

    restore_model = restore_model.cuda()

    if (args.restore_path != ""):
        # restore_model.load_state_dict(torch.load(args.restore_path))


        pretrain_dict = torch.load(args.restore_path)
        model_dict = {}
        state_dict = restore_model.state_dict()
        for (k_pretrained, v_pretrained),(k, v) in zip(pretrain_dict.items(),state_dict.items()):
            model_dict[k] = v_pretrained
        state_dict.update(model_dict)
        restore_model.load_state_dict(state_dict)

    if (args.loc_restore_path != ""):
        pretrain_dict = torch.load(args.loc_restore_path)
        model_dict = {}
        state_dict = loc_model.state_dict()
        for (k_pretrained, v_pretrained),(k, v) in zip(pretrain_dict.items(),state_dict.items()):
            model_dict[k] = v_pretrained
        state_dict.update(model_dict)
        loc_model.load_state_dict(state_dict)

    logger = get_logger(os.path.join(args.save_result_path, "test_"+args.restore_path.split("/")[-1]+".log"))
    logger.info(args.restore_path)
    logger.info(args.loc_restore_path)
    restore_model.eval()
    loc_model.eval()
    val_f1, val_iou, val_auc = [], [], []
    with torch.no_grad():
        for val_index, (data_x, data_y, data_degraded, file_name) in enumerate(test_dataloader):
            print(file_name)

            data_x = data_x.cuda()
            data_y = data_y.cuda()
            data_degraded = data_degraded.cuda()

            if(data_x.shape[2]<IMG_SIZE or data_x.shape[3]<IMG_SIZE):
                restore_image = restore_model(data_degraded)
                if args.model_type != 'MVSS_net':
                    outputs = loc_model(restore_image)
                else:
                    edge_outputs, outputs = loc_model(restore_image)
                    outputs = torch.sigmoid(outputs)
                    edge_outputs = torch.sigmoid(edge_outputs)
            else:
                test_degraded_images,row,col = read_image_block_with_mask(data_degraded,IMG_SIZE=IMG_SIZE,step = image_step)
                test_results,test_restores = [],[]
                for degraded in test_degraded_images:
                    degraded = degraded.cuda()
                    restored = restore_model(degraded)
                    test_restores.append(restored)
                    # outputs_patch = loc_model(restored)
                    if args.model_type != 'MVSS_net':
                        outputs_patch = loc_model(restored)
                    else:
                        edge_outputs, outputs_patch = loc_model(restored)
                        outputs_patch = torch.sigmoid(outputs_patch)
                        edge_outputs = torch.sigmoid(edge_outputs)
                    test_results.append(outputs_patch)
                restore_image = consist_whole_restore_image(test_restores,row,col,step = image_step,IMG_SIZE=IMG_SIZE)
                outputs = consist_whole_mask(test_results,row,col,step = image_step,IMG_SIZE=IMG_SIZE)

            val_f1, val_iou, val_auc = compute_metrics(data_x, data_y, outputs, val_f1, val_iou, val_auc)

            logger.info(
                "Image index {}/{} Image name {} Mean F1 = {} Mean IOU = {} Mean AUC = {}".format(
                    val_index, len(test_dataset), file_name[0], np.mean(val_f1), np.mean(val_iou), np.mean(val_auc)))


