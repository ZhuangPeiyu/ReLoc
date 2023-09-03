import argparse


def configs_DEFACTO(args):
    args.train_images_split = "../Datasets/DEFACTO/train_images.pickle"
    args.test_images_split = "../Datasets/DEFACTO/val_images.pickle"
    args.train_image_path = "../Datasets/DEFACTO/tamper/"
    args.train_mask_path = "../Datasets/DEFACTO/mask/"
    args.test_image_path = "../Datasets/DEFACTO/tamper/"
    args.test_mask_path = "../Datasets/DEFACTO/mask/"
    return args

def configs_DFCN_singleQF(args):
    args.localization_model = "denseFCN"
    args.mode = 'train'
    args.bn_in = 'bn'
    args.quality = 75
    args.epochs, args.loc_learning_rate, args.patience, args.restore_learning_rate, args.dis_learning_rate = [60,0.0001, 2,0.0001,0.0001]
    args.lr_schdular = ''
    args.loc_batch_size, args.loc_patch_size = 56, 128
    args.restore_batch_size, args.restore_patch_size = 56, 128
    args.aug = True
    args.display_step = 100

    return args

def configs_SCSEUnet_singleQF(args):
    args.localization_model = "SCSEUnet"
    args.mode = 'train'
    args.bn_in = 'bn'
    args.quality = 75
    args.epochs, args.loc_learning_rate, args.patience, args.restore_learning_rate, args.dis_learning_rate = [60,0.0001, 2,0.0001,0.0001]
    args.lr_schdular = ''
    args.loc_batch_size, args.loc_patch_size = 40, 128
    args.restore_batch_size, args.restore_patch_size = 40, 128
    args.aug = True
    args.display_step = 100

    return args

def configs_MVSS_net_singleQF(args):
    args.localization_model = "MVSS_net"
    args.mode = 'train'
    args.bn_in = 'bn'
    args.quality = 75
    args.epochs, args.loc_learning_rate, args.patience, args.restore_learning_rate, args.dis_learning_rate = [60,0.0001, 2,0.0001,0.0001]
    args.lr_schdular = ''
    args.loc_batch_size, args.loc_patch_size = 48, 128
    args.restore_batch_size, args.restore_patch_size = 48, 128
    args.aug = True
    args.display_step = 100
    return args


def configs_DFCN_DEFACTO(args):
    args = configs_DEFACTO(args) #
    args = configs_DFCN_singleQF(args)
    args.dataset_name = "DEFACTO"
    args.compress_type = "JPEG"
    args.IMG_SIZE = 512
    args.image_step = 512
    args.loc_restore_path = "../pretrained/DEFACTO/denseFCN/QF75/Loc_model/loc_model.pth"
    args.restore_path = "../pretrained/DEFACTO/denseFCN/QF75/Restore_model/restore_model.pth"
    args.cross_entropy_weight = 0.2
    args.restore_weight = 100
    args.GAN_weight = 0.1
    args.loc_weight = 1
    args.dis_step_iters = 1
    args.save_model_path = "../checkpoints/"+args.localization_model+"/"+args.dataset_name+"/QF_"+str(args.quality) + "/M_ReLoc/" + "/Cross_entroy_weight_{}_Restore_weight{}_Dis_weight_{}_Loc_weight_{}/".format(
        str(args.cross_entropy_weight),str(args.restore_weight), str(args.GAN_weight),str(args.loc_weight))
    return args

def configs_SCSEUnet_DEFACTO(args):
    args = configs_DEFACTO(args) #
    args = configs_SCSEUnet_singleQF(args)
    args.dataset_name = "DEFACTO"
    args.compress_type = "JPEG"
    args.IMG_SIZE = 512
    args.image_step = 512
    args.loc_restore_path = "../pretrained/DEFACTO/SCSEUnet/QF75/Loc_model/loc_model.pth"
    args.restore_path = "../pretrained/DEFACTO/SCSEUnet/QF75/Restore_model/restore_model.pth"
    args.cross_entropy_weight = 0.2
    args.restore_weight = 100
    args.GAN_weight = 0.1
    args.loc_weight = 1
    args.dis_step_iters = 1
    args.save_model_path = "../checkpoints/"+args.localization_model+"/"+args.dataset_name+"/QF_"+str(args.quality) + "/M_ReLoc/" + "/Cross_entroy_weight_{}_Restore_weight{}_Dis_weight_{}_Loc_weight_{}/".format(
        str(args.cross_entropy_weight),str(args.restore_weight), str(args.GAN_weight),str(args.loc_weight))
    return args


def configs_MVSSNet_DEFACTO(args):
    args = configs_DEFACTO(args) #
    args = configs_MVSS_net_singleQF(args)
    args.dataset_name = "DEFACTO"
    args.compress_type = "JPEG"
    args.IMG_SIZE = 512
    args.image_step = 512
    args.loc_restore_path = "../pretrained/DEFACTO/MVSSNet/QF75/Loc_model/loc_model.pth"
    args.restore_path = "../pretrained/DEFACTO/MVSSNet/QF75/Restore_model/restore_model.pth"
    args.cross_entropy_weight = 0.2
    args.restore_weight = 100
    args.GAN_weight = 0.1
    args.loc_weight = 1
    args.dis_step_iters = 1
    args.save_model_path = "../checkpoints/"+args.localization_model+"/"+args.dataset_name+"/QF_"+str(args.quality) + "/M_ReLoc/" + "/Cross_entroy_weight_{}_Restore_weight{}_Dis_weight_{}_Loc_weight_{}/".format(
        str(args.cross_entropy_weight),str(args.restore_weight), str(args.GAN_weight),str(args.loc_weight))
    return args
