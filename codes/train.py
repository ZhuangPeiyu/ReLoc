import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from models.unet import SCSEUnet #SCSEUnet, localization module
from MVSS_net.models.mvssnet import get_mvss # MVSSNet, localization module
import denseFCN #DFCN, localization module
from SCUNet_main.models import network_scunet # SCUNet, restoration module
import random
import time
import logging
import pickle
from sklearn.metrics import roc_auc_score
from metrics import get_metrics
from torch.nn.modules.loss import _Loss

import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import configs
torch.set_num_threads(8)
def create_dir(path):
    if(os.path.exists(path)==False):
        os.makedirs(path)
class argparse():
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
args = argparse()
'''设置随机数'''
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


'''SingleQF,JPEG compression'''

# args = configs.configs_DFCN_DEFACTO(args)
# args = configs.configs_SCSEUnet_DEFACTO(args)
args = configs.configs_MVSSNet_DEFACTO(args)

create_dir(args.save_model_path)
create_dir(os.path.join(args.save_model_path,"Loc_model"))
create_dir(os.path.join(args.save_model_path,"Restore_model"))


def random_crop(img, mask, crop_shape):

    if (img.shape[0]==crop_shape[0] and img.shape[1]==crop_shape[1]):
        return img,mask
    if img.shape[0] < crop_shape[0] or img.shape[1] < crop_shape[1]:
        img = cv2.resize(img, (crop_shape[1], crop_shape[0]))
        mask = cv2.resize(mask, (crop_shape[1], crop_shape[0]), interpolation=cv2.INTER_NEAREST)

    original_shape = mask.shape
    crop_mask = np.zeros((crop_shape[0],crop_shape[1],3))
    count = 0
    # print(mask.shape,crop_mask.shape)
    while(np.sum(crop_mask)==0 and count <1):
        start_h = np.random.randint(0, original_shape[0] - crop_shape[0] + 1)
        start_w = np.random.randint(0, original_shape[1] - crop_shape[1] + 1)
        crop_img = img[start_h: start_h + crop_shape[0], start_w: start_w + crop_shape[1], :]
        crop_mask = mask[start_h: start_h + crop_shape[0], start_w: start_w + crop_shape[1],:]
        count += 1
    return crop_img, crop_mask
def aug(img, mask, degraded_img,patch_size,file_list):
    H, W, _ = img.shape

    # Flip
    if random.random() < 0.5:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
        degraded_img = cv2.flip(degraded_img, 0)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
        degraded_img = cv2.flip(degraded_img, 1)

    if random.random() < 0.5:
        tmp = random.random()
        if tmp < 0.33:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            degraded_img = cv2.rotate(degraded_img, cv2.ROTATE_90_CLOCKWISE)
        elif (tmp >= 0.33 and tmp < 0.66):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            degraded_img = cv2.rotate(degraded_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        else:
            img = cv2.rotate(img, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
            degraded_img = cv2.rotate(degraded_img, cv2.ROTATE_180)
    return img, mask,degraded_img

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
    def __init__(self, file,choice='train',patch_size=512,with_aug = False,compress_quality = None):
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
        if self.choice != 'test':
            fname1, fname2 = self.filelist[idx]
        else:
            fname1, fname2 = self.filelist[idx], ''

        img = cv2.imread(fname1)

        H, W, _ = img.shape

        if fname2 == '':
            mask = np.zeros([H, W, 3])
        else:
            mask = cv2.imread(fname2)
            if(np.max(mask)<=1):
                mask = mask*255

        if self.choice =='train':
            img,mask = random_crop(img, mask,(self.patch_size, self.patch_size)) #先分块再数据增强
            if(self.compress_quality!=None and self.compress_quality!='70_to_100'):
                result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.compress_quality])
                degraded_img = cv2.imdecode(encimg, 1)
            elif(self.compress_quality!=None and self.compress_quality=='70_to_100'):
                quality = random.randint(70,100)
                result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                degraded_img = cv2.imdecode(encimg, 1)
            else:
                degraded_img = img

        if self.choice == 'val':
            if(self.compress_quality!=None and self.compress_quality!='70_to_100'):
                result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.compress_quality])
                degraded_img = cv2.imdecode(encimg, 1)
            elif(self.compress_quality!=None and self.compress_quality=='70_to_100'):
                quality = random.randint(70,100)
                result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                degraded_img = cv2.imdecode(encimg, 1)
            else:
                degraded_img = img
            H,W,_ = img.shape

        img = img[:,:,::-1].astype('float') / 255.
        degraded_img = degraded_img[:,:,::-1].astype('float') / 255.
        mask = mask.astype('float')
        mask_copy = np.copy(mask)
        mask[np.where(mask_copy < 127.5)] = 0
        mask[np.where(mask_copy >= 127.5)] = 1
        if(self.choice=='train' or self.choice=='val'):
            return self.tensor(img), self.tensor(mask[:, :, :1]), self.tensor(degraded_img),fname1.split('/')[-1]

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)

def compute_metrics(image_batch,label_batch,outputs_batch,f1_all,iou_all,auc_all):
    # #计算每一个batch里面每一张图的F1和AUC
    image_batch = image_batch.cpu().detach().numpy()
    label_batch = label_batch.cpu().detach().numpy()
    outputs_batch = outputs_batch.cpu().detach().numpy()
    for image, label, predict_map in zip(image_batch, label_batch, outputs_batch):
        predict_map = predict_map[0,:,:]
        label = label[0,:,:]
        # print(np.unique(predict_map),np.unique(label))
        predict_threshold = np.copy(predict_map)
        predict_threshold[np.where(predict_map<0.5)] = 0
        predict_threshold[np.where(predict_map>=0.5)] = 1
        if(len(np.unique(label))<2):
            continue
        try:
            # f1 = f1_score(label.reshape(-1, ), predict_threshold.reshape(-1, ), zero_division=0)
            tpr_recall, tnr, precision, f1, mcc, iou, tn, tp, fn, fp = get_metrics(predict_threshold, label)
            auc = roc_auc_score(label.reshape(-1, ), predict_map.reshape(-1, ))
            # auc = 0
            f1_all.append(f1)
            auc_all.append(auc)
            iou_all.append(iou)
        except Exception as e:
            print(e)
            continue
    return f1_all,iou_all,auc_all
class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self,y_pred, y_true, eps=1e-8):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.size() == y_true.size(), "the size of predict and target must be equal."
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1.0 - dice
        return dice_loss

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

class Discriminator(nn.Module):
    def __init__(self,nc = 3,ndf = 64):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.ndf * 8, 1, 1, 1, 0, bias=False),

            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
if args.mode=='train':
    train_file = create_file_list(args.train_image_path,args.train_mask_path,args.train_images_split)
    val_file = create_file_list(args.test_image_path,args.test_mask_path,args.test_images_split)

    train_dataset_restore = Tampering_Dataset(train_file, choice='train', patch_size=args.restore_patch_size,with_aug=args.aug,compress_quality=args.quality)
    train_dataset_loc = Tampering_Dataset(train_file, choice='train', patch_size=args.loc_patch_size,with_aug=args.aug,compress_quality=args.quality)
    val_dataset = Tampering_Dataset(val_file, choice='val', patch_size=args.loc_patch_size,compress_quality=args.quality)

    train_dataloader_restore = DataLoader(dataset=train_dataset_restore, batch_size=args.restore_batch_size, shuffle=True,drop_last=True,pin_memory=False,num_workers=8)
    train_dataloader_loc = DataLoader(dataset=train_dataset_loc, batch_size=args.loc_batch_size, shuffle=True,
                                          drop_last=True, pin_memory=False, num_workers=8)

    valid_dataloader = DataLoader(dataset=val_dataset, batch_size=args.loc_batch_size if 'OnlyOneRma2' in args.train_image_path else 1, shuffle=False,pin_memory=False,num_workers=8,drop_last=True)


    restore_model = network_scunet.SCUNet(in_nc=3).cuda()

    if(args.localization_model == "SCSEUnet"):
        loc_model = SCSEUnet(backbone_arch='senet154').cuda()
    elif args.localization_model =='denseFCN':
        loc_model = denseFCN.normal_denseFCN(bn_in=args.bn_in).cuda()
    elif (args.localization_model == 'MVSS_net'):
        loc_model = get_mvss(backbone='resnet50',
                             pretrained_base=True,
                             nclass=1,
                             sobel=True,
                             constrain=True,
                             n_input=3).cuda()
    dis_model = Discriminator(nc= 3,ndf = 64)
    dis_model.apply(weights_init)

    if (args.restore_path != ""):
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

    loc_model = nn.DataParallel(loc_model).cuda()
    dis_model = nn.DataParallel(dis_model).cuda()
    restore_model = nn.DataParallel(restore_model).cuda()

    criterion = torch.nn.BCELoss()
    criterion_dice = SoftDiceLoss().cuda()
    criterion_l1 = torch.nn.L1Loss()
    optimizer_restore = torch.optim.Adam(restore_model.parameters(),lr = args.restore_learning_rate)
    optimizer_loc = torch.optim.Adam(loc_model.parameters(),lr=args.loc_learning_rate)
    optimizer_dis = torch.optim.Adam(dis_model.parameters(),lr = args.dis_learning_rate,betas=(0.5,0.999))

    if(args.lr_schdular==''):
        lr_schdular_restore = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_restore,'min',factor=0.8,patience=args.patience)
        lr_schdular_loc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_loc,'min',factor=0.8,patience=args.patience)
    logger = get_logger(os.path.join(args.save_model_path,"log.log"))

    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    best_f1 = 0
    best_auc = 0
    best_val_loss = 99999
    logger.info("Train Loc data:{} Train Restore data: {} Val data:{}".format(len(train_dataloader_loc),len(train_dataloader_restore),len(valid_dataloader)))
    total_iter_restore = 0
    total_iter_loc = 0

    for epoch in range(args.epochs):
        lr_restore = optimizer_restore.param_groups[0]['lr']
        lr_loc = optimizer_loc.param_groups[0]['lr']
        restore_model.train()
        loc_model.train()
        '''训练restore model'''
        if(epoch%2==0):
            for param in restore_model.parameters():
                param.requires_grad = True
            for param in loc_model.parameters():
                param.requires_grad = True
            loc_model.eval()
            restore_model.train()

            train_epoch_loss = []
            train_epoch_restore_loss, train_epoch_loc_loss = [], []
            train_f1, train_iou, train_auc = [], [], []
            train_epoch_D,train_epoch_G,train_epoch_Dx,train_epoch_D_G_z1,train_epoch_D_G_z2 = [],[],[],[],[]
            train_epoch_D_err_fake,train_epoch_D_err_real = [],[]
            for idx, (data_x, data_y, data_degraded, file_name) in enumerate(train_dataloader_restore, 0):
                data_x = data_x.cuda()
                data_y = data_y.cuda()
                data_degraded = data_degraded.cuda()
                t1 = time.time()
                restore_image = restore_model(data_degraded)
                t2 = time.time()
                if (args.localization_model != "MVSS_net"):
                    outputs = loc_model(restore_image)
                else:
                    edge_outputs, outputs = loc_model(restore_image)
                    outputs = torch.sigmoid(outputs)
                    edge_outputs = torch.sigmoid(edge_outputs)

                total_iter_restore += 1
                for _ in range(args.dis_step_iters):
                    dis_model.zero_grad()
                    b_size = data_x.size(0)
                    real_label = 1
                    fake_label = 0
                    label = torch.full((b_size,), real_label, dtype=torch.float).cuda()
                    output_hr = dis_model(data_x).view(-1)
                    errD_real = criterion(output_hr, label)
                    errD_real.backward()
                    D_x = output_hr.mean().item()
                    label.fill_(fake_label)
                    # Classify all fake batch with D
                    output_restore = dis_model(restore_image.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = criterion(output_restore, label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    D_G_z1 = output_restore.mean().item()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    # errD.backward()
                    # Update D
                    optimizer_dis.step()

                optimizer_restore.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output_restore = dis_model(restore_image).view(-1)
                errG = criterion(output_restore, label)
                D_G_z2 = output_restore.mean().item()

                loss_restore = args.restore_weight * criterion_l1(restore_image.view(restore_image.size(0), -1),data_x.view(data_x.size(0), -1)) + \
                               args.loc_weight * (args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),data_y.view(data_y.size(0), -1)) + (1 - args.cross_entropy_weight) * criterion_dice(outputs.view(outputs.size(0), -1),data_y.view(data_y.size(0), -1))) + \
                               args.GAN_weight * errG

                loss_restore.backward()
                optimizer_restore.step()
                train_epoch_restore_loss.append(args.restore_weight * criterion_l1(restore_image.view(data_degraded.size(0), -1),data_x.view(data_x.size(0), -1)).item())
                train_epoch_loc_loss.append(args.loc_weight * (args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),data_y.view(data_y.size(0), -1)) + (1-args.cross_entropy_weight) * criterion_dice(outputs.view(outputs.size(0), -1), data_y.view(data_y.size(0), -1))).item())

                if total_iter_restore % args.display_step == 0:
                    logger.info("epoch={}/{},{}/{}of train {}, Learning rate={} Restore loss = {} Loc loss = {}".format(
                        epoch, args.epochs, idx, len(train_dataloader_restore), total_iter_restore, lr_restore,
                        np.mean(train_epoch_restore_loss),np.mean(train_epoch_loc_loss)))

        else:
            for param in loc_model.parameters():
                param.requires_grad = True
            for param in restore_model.parameters():
                param.requires_grad = False
            loc_model.train()
            restore_model.eval()
            train_epoch_loss = []
            train_epoch_restore_loss, train_epoch_loc_loss = [], []
            train_f1, train_iou, train_auc = [], [], []
            for idx, (data_x, data_y, data_degraded, file_name) in enumerate(train_dataloader_loc, 0):
                data_x = data_x.cuda()
                data_y = data_y.cuda()
                data_degraded = data_degraded.cuda()
                restore_image = restore_model(data_degraded)
                if (args.localization_model != 'MVSS_net'):
                    outputs = loc_model(restore_image)
                    loss_loc = args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),data_y.view(data_y.size(0), -1)) + \
                               (1-args.cross_entropy_weight) * criterion_dice(outputs.view(outputs.size(0), -1), data_y.view(data_y.size(0), -1))
                else:
                    edge_outputs, outputs = loc_model(restore_image)
                    outputs = torch.sigmoid(outputs)
                    edge_outputs = torch.sigmoid(edge_outputs)
                    loss_loc = args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),data_y.view(data_y.size(0), -1)) + \
                               (1-args.cross_entropy_weight) * criterion_dice(outputs.view(outputs.size(0), -1), data_y.view(data_y.size(0), -1))
                total_iter_loc += 1

                optimizer_loc.zero_grad()
                loss_loc.backward()
                optimizer_loc.step()
                t5 = time.time()
                train_epoch_loc_loss.append(loss_loc.item())
                if total_iter_loc % args.display_step == 0:
                    logger.info("epoch={}/{},{}/{}of train {}, Learning rate={} Loc loss = {} ".format(
                        epoch, args.epochs, idx, len(train_dataloader_loc), total_iter_loc, lr_loc,
                        np.mean(train_epoch_loc_loss)))

            if (True):
                restore_model.eval()
                loc_model.eval()
                valid_epoch_loss = []
                val_epoch_loc_loss,val_epoch_restore_loss = [],[]
                val_f1, val_iou, val_auc = [], [], []
                with torch.no_grad():
                    for val_index, (data_x, data_y,data_degraded,file_name) in enumerate(valid_dataloader):
                        data_x = data_x.cuda()
                        data_y = data_y.cuda()
                        data_degraded = data_degraded.cuda()
                        restore_image = restore_model(data_degraded)

                        if (args.localization_model != 'MVSS_net'):
                            outputs = loc_model(restore_image)
                            loss_loc = args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),data_y.view(data_y.size(0), -1)) + (1-args.cross_entropy_weight) * criterion_dice(outputs.view(outputs.size(0), -1), data_y.view(data_y.size(0), -1))
                        else:
                            edge_outputs, outputs = loc_model(restore_image)
                            outputs = torch.sigmoid(outputs)
                            edge_outputs = torch.sigmoid(edge_outputs)
                            loss_loc = args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),data_y.view(data_y.size(0), -1)) + (1-args.cross_entropy_weight) * criterion_dice(outputs.view(outputs.size(0), -1), data_y.view(data_y.size(0), -1))

                        loss_restore = criterion_l1(restore_image.contiguous().view(data_degraded.size(0),-1),data_x.contiguous().view(data_x.size(0),-1))
                        loss = loss_loc + loss_restore
                        valid_epoch_loss.append(loss.item())
                        val_epoch_loc_loss.append(loss_loc.item())
                        val_epoch_restore_loss.append(loss_restore.item())
                        val_f1, val_iou, val_auc = compute_metrics(data_x, data_y, outputs, val_f1, val_iou, val_auc)

                if(np.mean(valid_epoch_loss)<best_val_loss or np.mean(val_f1)>best_f1 or np.mean(val_auc)>best_auc):
                    if(np.mean(valid_epoch_loss)<best_val_loss):
                        best_val_loss = np.mean(valid_epoch_loss)
                    if (np.mean(val_f1)>best_f1):
                        best_f1 = np.mean(val_f1)
                    if (np.mean(val_auc)>best_auc):
                        best_auc = np.mean(val_auc)

                    torch.save(restore_model.state_dict(), os.path.join(args.save_model_path, "Restore_model",
                                                                        "Epoch_{}_Loss_{}_F1_{}_AUC_{}.pth".format(
                                                                            epoch,
                                                                            round(np.mean(valid_epoch_loss), 4),
                                                                            round(np.mean(val_f1), 4),
                                                                            round(np.mean(val_auc), 4))))
                    torch.save(loc_model.state_dict(), os.path.join(args.save_model_path, "Loc_model",
                                                                    "Epoch_{}_Loss_{}_F1_{}_AUC_{}.pth".format(
                                                                        epoch, round(np.mean(valid_epoch_loss), 4),
                                                                        round(np.mean(val_f1), 4),
                                                                        round(np.mean(val_auc), 4))))



                logger.info("Validation {}, Restore Learning rate = {} Loc learning rate = {} Total Loss = {} Restore loss = {} Loc Loss = {} F1 = {} IOU = {} AUC = {} Best loss = {} Best f1 = {} Best auc = {}".format(epoch,lr_restore,lr_loc,np.mean(valid_epoch_loss),np.mean(val_epoch_restore_loss),np.mean(val_epoch_loc_loss),np.mean(val_f1),np.mean(val_iou),np.mean(val_auc),best_val_loss,best_f1,best_auc))

                val_mean_loss = np.mean(valid_epoch_loss)

            if(epoch%2==0):
                lr_schdular_restore.step(np.mean(valid_epoch_loss))
            else:
                lr_schdular_loc.step(np.mean(valid_epoch_loss))