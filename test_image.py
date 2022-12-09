import argparse
import os
from torchvision.utils import save_image
import torch
from torch.autograd import Variable
import utils
import numpy as np
from model import IR_Net, Fusion_Net

cuda = True if torch.cuda.is_available() else False


def load_model(dataset_name):
    torch.cuda.set_device('cuda:0')

    IR_Net_model = torch.nn.DataParallel(IR_Net(), device_ids=[0])
    Fusion_Net_model = torch.nn.DataParallel(Fusion_Net(), device_ids=[0])

    IR_Net_model.load_state_dict(
        torch.load("saved_models/%s/IR_Net_model_%d.pth" % (dataset_name, 0)))
    Fusion_Net_model.load_state_dict(
        torch.load("saved_models/%s/Fusion_Net_model_%d.pth" % (dataset_name, 0)))

    IR_Net_model.cuda()
    IR_Net_model.eval()
    Fusion_Net_model.cuda()
    Fusion_Net_model.eval()

    return IR_Net_model, Fusion_Net_model


def image_fusion(IR_Net_model, Fusion_Net_model, real_rgb_imgs, real_ir_imgs, rgb_hist, ir_hist):
    is_test = True
    _, _, _, feature3_1, feature3_2, feature3_3, feature2_1, feature2_2, feature2_3, feature1_1, feature1_2, feature1_3 = IR_Net_model(real_rgb_imgs,real_ir_imgs)
    _, _, fusion_imgs = Fusion_Net_model(feature3_1, feature3_2, feature3_3, feature2_1, feature2_2, feature2_3, feature1_1, feature1_2, feature1_3, real_rgb_imgs, real_ir_imgs, rgb_hist, ir_hist,is_test)

    return fusion_imgs


def run_demo(IR_Net_model, Fusion_Net_model, infrared_path, visible_path, index, out_path):
    ir_img = utils.get_test_images(infrared_path, height=None, width=None)
    vis_img = utils.get_test_images(visible_path, height=None, width=None)

    # -----------------
    # make histogram
    # -----------------

    img1_hist = ((vis_img * 0.5) + 0.5) * 255
    img2_hist = ((ir_img * 0.5) + 0.5) * 255
    rgb_hist, _ = np.histogram(img1_hist.flatten(), 256, [0, 256])
    rgb_hist = rgb_hist / np.sum(rgb_hist)
    ir_hist, _ = np.histogram(img2_hist.flatten(), 256, [0, 256])
    ir_hist = ir_hist / np.sum(ir_hist)
    if cuda:
        ir_img = ir_img.cuda()
        vis_img = vis_img.cuda()
    ir_img = Variable(ir_img, requires_grad=False)
    vis_img = Variable(vis_img, requires_grad=False)
    rgb_hist = torch.Tensor(rgb_hist)

    rgb_hist.unsqueeze(0)
    ir_hist = torch.Tensor(ir_hist)
    ir_hist.unsqueeze(0)

    img_fusion = image_fusion(IR_Net_model, Fusion_Net_model, vis_img, ir_img, rgb_hist, ir_hist)
    file_name = str(index) + '.png'
    output_path = out_path + file_name

    # --------------
    # save images
    # --------------

    save_image(img_fusion, "fusion_outputs/%d.png" % index, normalize=True)
    print(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='fusion_outputs/', help='path of fused image')
    parser.add_argument('--test_images', type=str, default='test_images/', help='path of source image')
    parser.add_argument('--dataset_name', type=str, default='basic', help='dataset name')
    args = parser.parse_args()

    if os.path.exists(args.out_path) is False:
        os.mkdir(args.out_path)

    with torch.no_grad():
        IR_Net_model, Fusion_Net_model = load_model(args.dataset_name)
        for i in range(20):
            index = i + 1
            infrared_path = args.test_images + '/' + 'IR' + str(index) + '.png'
            visible_path = args.test_images + '/' + 'VIS' + str(index) + '.png'

            # -------------
            # test image
            # -------------

            run_demo(IR_Net_model, Fusion_Net_model, infrared_path, visible_path, index, args.out_path)

    print('Done!')


if __name__ == '__main__':
    main()
