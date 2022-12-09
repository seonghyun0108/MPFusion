import argparse
from torchvision import transforms
import torch
import torch.nn as nn
import pytorch_ssim
from torch.utils.data import DataLoader

from utils import Customdataset, Customdataset_with_hist
from losses import Per_loss, Cos_loss, Spa_loss
from model import IR_Net, Fusion_Net
from IR_train import ir_train
from Fus_train import fus_train


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dataset', type=str, default='/home/seonghyun/visible_20000_/visible_20000/',
                        help='path of rgb dataset')
    parser.add_argument('--ir_dataset', type=str, default='/home/seonghyun/lwir_20000_/lwir_20000/',
                        help='path of ir dataset')
    parser.add_argument('--test_images', type=str, default='test_images', help='path of image visualization')
    parser.add_argument('--train_mode', type=int, default=1, help='train mode: 1(IRNet) 2(FusionNet)')
    parser.add_argument('--sample_interval', type=int, default=1000, help='interval of saving image')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dataset_name', type=str, default='basic', help='dataset name')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval of saving model')
    parser.add_argument('--b1', type=float, default=0.5, help='fixed!')
    parser.add_argument('--b2', type=float, default=0.999, help='fixed!')
    args = parser.parse_args()

    # -----------------
    # device setting
    # -----------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    torch.backends.cudnn.enabled = False

    cuda = True if torch.cuda.is_available() else False

    trans = transforms.Compose(
        [transforms.RandomCrop(256),
         transforms.ToTensor(),
         transforms.Grayscale(num_output_channels=1),
         transforms.Normalize((0.5,), (0.5,))])

    MSE_loss = nn.MSELoss()
    SSIM_loss = pytorch_ssim.SSIM()

    if args.train_mode == 1:

        # ---------------
        #  DataLoaders
        # ---------------

        ir_dataset = Customdataset(transform=trans, rgb_dataset=args.rgb_dataset, ir_dataset=args.ir_dataset)
        ir_dataloader = torch.utils.data.DataLoader(dataset=ir_dataset, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=2)

        # -------------------
        # Loss & Optimizer
        # -------------------

        if cuda:
            torch.cuda.set_device('cuda:0')

            IR_Net_model = torch.nn.DataParallel(IR_Net(), device_ids=[0])
            IR_Net_model.cuda()
            IR_Net_model.train()
            ir_optim = torch.optim.Adam(IR_Net_model.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=1e-8,
                                        weight_decay=1e-8)
            ir_train(IR_Net_model, ir_optim, MSE_loss, SSIM_loss, ir_dataloader, args.epochs, args.sample_interval, args.checkpoint_interval, args.dataset_name)


    else:

        # ---------------
        #  DataLoaders
        # ---------------

        fus_dataset = Customdataset_with_hist(transform=trans, rgb_dataset=args.rgb_dataset, ir_dataset=args.ir_dataset)
        fus_dataloader = torch.utils.data.DataLoader(dataset=fus_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=2)

        # -------------------
        # Loss & Optimizer
        # -------------------

        loss_cos = Cos_loss()
        loss_p = Per_loss()
        loss_spa = Spa_loss()

        if cuda:
            torch.cuda.set_device('cuda:0')

            IR_Net_model = torch.nn.DataParallel(IR_Net(), device_ids=[0,1,2,3])
            IR_Net_model.load_state_dict(
                torch.load("saved_models/%s/IR_Net_model_%d.pth" % (args.dataset_name, 15000)))
            IR_Net_model.cuda()
            IR_Net_model.eval()

            Fusion_Net_model = torch.nn.DataParallel(Fusion_Net(), device_ids=[0,1,2,3])
            Fusion_Net_model.cuda()
            Fusion_Net_model.train()
            fus_optim = torch.optim.Adam(Fusion_Net_model.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=1e-8,
                                         weight_decay=1e-8)

            num_params = 0
            for param in IR_Net_model.parameters():
                num_params += param.numel()
            for param in Fusion_Net_model.parameters():
                num_params += param.numel()
            print('# of params : %d' % num_params)

            fus_train(IR_Net_model, Fusion_Net_model, fus_optim, MSE_loss, loss_spa, loss_p, fus_dataloader,
                      args.epochs, args.sample_interval, args.checkpoint_interval, args.dataset_name)

if __name__ == '__main__':

    main()