import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import SpatialGradient
from torch import Tensor


##########################################################################
class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()

    def forward(self, x1, x2):
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right % 2 is 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot % 2 is 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


##########################################################################
class EdgeDetect(nn.Module):
    def __init__(self):
        super(EdgeDetect, self).__init__()
        self.spatial = SpatialGradient('diff')
        self.max_pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        y = self.max_pool(u)
        return y


##########################################################################
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, is_last=False):
        super(ConvLayer, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)

    def forward(self, x):
        out = self.conv2d(x)
        return out


##########################################################################
class CALayer(nn.Module):
    def __init__(self, channel, reduction=32, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################


class IR_MDRB(nn.Module):
    def __init__(self):
        super(IR_MDRB, self).__init__()
        conv = ConvLayer
        n_feats = 32
        kernel_size_1 = 3
        kernel_size_2 = 3

        self.conv_3_1_1 = conv(n_feats, n_feats, kernel_size_1, 1)
        self.conv_3_2_1 = conv(n_feats * 2, n_feats * 2, kernel_size_1, 1)
        self.conv_5_1_1 = conv(n_feats, n_feats, kernel_size_2, 1, 3, 3)
        self.conv_5_2_1 = conv(n_feats * 2, n_feats * 2, kernel_size_2, 1, 3, 3)

        self.conv_3_1_2 = conv(n_feats, n_feats, kernel_size_1, 1)
        self.conv_3_2_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1, 1)
        self.conv_5_1_2 = conv(n_feats, n_feats, kernel_size_2, 1, 3, 3)
        self.conv_5_2_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2, 1, 3, 3)

        self.conv_3_1_3 = conv(n_feats, n_feats, kernel_size_1, 1)
        self.conv_3_2_3 = conv(n_feats * 2, n_feats * 2, kernel_size_1, 1)
        self.conv_5_1_3 = conv(n_feats, n_feats, kernel_size_2, 1, 3, 3)
        self.conv_5_2_3 = conv(n_feats * 2, n_feats * 2, kernel_size_2, 1, 3, 3)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.bottleneck = nn.Conv2d(n_feats * 3, n_feats, 1, padding=0, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1_1 = self.relu(self.conv_3_1_1(input_1))
        output_5_1_1 = self.relu(self.conv_5_1_1(input_1))
        input_2_1 = torch.cat([output_3_1_1, output_5_1_1], 1)
        output_3_2_1 = self.relu(self.conv_3_2_1(input_2_1))
        output_5_2_1 = self.relu(self.conv_5_2_1(input_2_1))
        input_3_1 = torch.cat([output_3_2_1, output_5_2_1], 1)
        output_1 = self.confusion(input_3_1)
        output_1 += x

        output_3_1_2 = self.relu(self.conv_3_1_2(output_1))
        output_5_1_2 = self.relu(self.conv_5_1_2(output_1))
        input_2_2 = torch.cat([output_3_1_2, output_5_1_2], 1)
        output_3_2_2 = self.relu(self.conv_3_2_2(input_2_2))
        output_5_2_2 = self.relu(self.conv_5_2_2(input_2_2))
        input_3_2 = torch.cat([output_3_2_2, output_5_2_2], 1)
        output_2 = self.confusion(input_3_2)
        output_2 += output_1

        output_3_1_3 = self.relu(self.conv_3_1_3(output_2))
        output_5_1_3 = self.relu(self.conv_5_1_3(output_2))
        input_2_3 = torch.cat([output_3_1_3, output_5_1_3], 1)
        output_3_2_3 = self.relu(self.conv_3_2_3(input_2_3))
        output_5_2_3 = self.relu(self.conv_5_2_3(input_2_3))
        input_3_3 = torch.cat([output_3_2_3, output_5_2_3], 1)
        output_3 = self.confusion(input_3_3)
        output_3 += output_2

        output_4 = torch.cat([output_1, output_2, output_3], 1)
        output_4 = self.bottleneck(output_4)

        return output_1, output_2, output_3, output_4


##########################################################################
class Fus_MDRB(nn.Module):
    def __init__(self):
        super(Fus_MDRB, self).__init__()
        conv = ConvLayer
        n_feats = 32
        kernel_size_1 = 3
        kernel_size_2 = 3

        self.conv_3_1_1 = conv(n_feats, n_feats, kernel_size_1, 1)
        self.conv_3_2_1 = conv(n_feats * 2, n_feats * 2, kernel_size_1, 1)
        self.conv_5_1_1 = conv(n_feats, n_feats, kernel_size_2, 1, 3, 3)
        self.conv_5_2_1 = conv(n_feats * 2, n_feats * 2, kernel_size_2, 1, 3, 3)

        self.conv_3_1_2 = conv(n_feats, n_feats, kernel_size_1, 1)
        self.conv_3_2_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1, 1)
        self.conv_5_1_2 = conv(n_feats, n_feats, kernel_size_2, 1, 3, 3)
        self.conv_5_2_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2, 1, 3, 3)

        self.conv_3_1_3 = conv(n_feats, n_feats, kernel_size_1, 1)
        self.conv_3_2_3 = conv(n_feats * 2, n_feats * 2, kernel_size_1, 1)
        self.conv_5_1_3 = conv(n_feats, n_feats, kernel_size_2, 1, 3, 3)
        self.conv_5_2_3 = conv(n_feats * 2, n_feats * 2, kernel_size_2, 1, 3, 3)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)

        self.confusion2 = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.bottleneck = nn.Conv2d(n_feats * 3, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, Ir_input, Ir_input2, Ir_input3, ir_param_final, rgb_param_final):
        input_1 = x
        output_3_1_1 = self.relu(self.conv_3_1_1(input_1))
        output_5_1_1 = self.relu(self.conv_5_1_1(input_1))
        input_2_1 = torch.cat([output_3_1_1, output_5_1_1], 1)
        output_3_2_1 = self.relu(self.conv_3_2_1(input_2_1))
        output_5_2_1 = self.relu(self.conv_5_2_1(input_2_1))
        input_3_1 = torch.cat([output_3_2_1, output_5_2_1], 1)
        output_1 = self.confusion(input_3_1)
        output_1 += x
        output_1_feature = output_1

        output_1 = rgb_param_final[:, 0:32, :, :] * output_1 + ir_param_final[:, 0:32, :, :] * Ir_input

        output_3_1_2 = self.relu(self.conv_3_1_2(output_1))
        output_5_1_2 = self.relu(self.conv_5_1_2(output_1))
        input_2_2 = torch.cat([output_3_1_2, output_5_1_2], 1)
        output_3_2_2 = self.relu(self.conv_3_2_1(input_2_2))
        output_5_2_2 = self.relu(self.conv_5_2_1(input_2_2))
        input_3_2 = torch.cat([output_3_2_2, output_5_2_2], 1)
        output_2 = self.confusion(input_3_2)
        output_2 += output_1
        output_2_feature = output_2
        output_2 = rgb_param_final[:, 32:64, :] * output_2 + ir_param_final[:, 32:64, :] * Ir_input2

        output_3_1_3 = self.relu(self.conv_3_1_3(output_2))
        output_5_1_3 = self.relu(self.conv_5_1_3(output_2))
        input_2_3 = torch.cat([output_3_1_3, output_5_1_3], 1)
        output_3_2_3 = self.relu(self.conv_3_2_3(input_2_3))
        output_5_2_3 = self.relu(self.conv_5_2_3(input_2_3))
        input_3_3 = torch.cat([output_3_2_3, output_5_2_3], 1)
        output_3 = self.confusion(input_3_3)
        output_3 += output_2
        output_3_feature = output_3
        output_3 = rgb_param_final[:, 64:96, :] * output_3 + ir_param_final[:, 64:96, :] * Ir_input3

        output_4 = torch.cat([output_1, output_2, output_3], 1)
        output_4 = self.bottleneck(output_4)

        return output_1, output_2, output_3, output_4, output_1_feature, output_2_feature, output_3_feature


##########################################################################
class Recon_residual(nn.Module):
    def __init__(self):
        super(Recon_residual, self).__init__()
        kernel_size = 3
        stride = 1
        self.conv5 = ConvLayer(32, 32, kernel_size, stride)
        self.conv6 = ConvLayer(64, 32, kernel_size, stride)
        self.conv7 = ConvLayer(64, 32, kernel_size, stride)
        self.conv8 = ConvLayer(32, 1, 1, 1, 0)

    def forward(self, x, skip1, skip2):
        x = self.conv5(x)
        x2 = torch.cat([x, skip1], 1)
        x3 = self.conv6(x2)
        x3 = torch.cat([x3, skip2], 1)
        output_3 = self.conv7(x3)
        image = self.conv8(output_3)
        return output_3, image


##########################################################################
class IR_Net(nn.Module):
    def __init__(self):
        super(IR_Net, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        nb_filter = [32, 32, 48, 64]
        kernel_size = 3
        stride = 1
        self.M = 16

        self.ed = EdgeDetect()

        self.up_eval = UpsampleReshape_eval()
        self.up = nn.Upsample(scale_factor=2)
        self.down = nn.Upsample(scale_factor=0.5)
        self.Stage1 = IR_MDRB()
        self.Stage2 = IR_MDRB()
        self.Stage3 = IR_MDRB()

        self.conv6 = Recon_residual()
        self.conv7 = Recon_residual()
        self.conv8 = Recon_residual()

        self.encoder_level1 = [CAB(32, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.encoder_level2 = [CAB(32, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.encoder_level3 = [CAB(32, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.conv1 = ConvLayer(2, 1, 1, stride, 0)
        self.conv2 = ConvLayer(2, nb_filter[0], kernel_size, stride)
        self.conv3 = ConvLayer(2 * nb_filter[0], nb_filter[0], 1, stride, 0)
        self.conv4 = ConvLayer(2 * nb_filter[0], nb_filter[0], 1, stride, 0)

    def forward(self, rgb, input):
        rgb_attention = ((rgb * 127.5) + 127.5) / 255
        ir_attention = ((input * 127.5) + 127.5) / 255
        edgemap = self.ed(ir_attention)
        edgemap2 = self.ed(rgb_attention)
        edgemap_ir = edgemap / (edgemap + edgemap2 + 0.00001)
        edgemap_ir = (edgemap_ir - 0.5) * 2
        input_ = torch.cat([input, edgemap_ir], 1)
        input_ = self.conv2(input_)

        input_2 = F.interpolate(input_, scale_factor=0.5, mode='bilinear')
        input_3 = F.interpolate(input_2, scale_factor=0.5, mode='bilinear')

        CA_3 = self.encoder_level1(input_3)
        feature3_1, feature3_2, feature3_3, bottle1 = self.Stage1(input_3)
        bottle1 = bottle1 + CA_3
        fake_input_2, output3 = self.conv6(bottle1, feature3_1, feature3_2)

        fake_input_2 = F.interpolate(fake_input_2, scale_factor=2, mode='bilinear')
        fake_input_2 = self.up_eval(input_2, fake_input_2)
        input_2 = torch.cat([input_2, fake_input_2], 1)
        input_2 = self.conv3(input_2)
        CA_2 = self.encoder_level2(input_2)
        feature2_1, feature2_2, feature2_3, bottle2 = self.Stage2(input_2)
        bottle2 = bottle2 + CA_2
        fake_input, output2 = self.conv7(bottle2, feature2_1, feature2_2)

        fake_input = F.interpolate(fake_input, scale_factor=2, mode='bilinear')
        fake_input = self.up_eval(input_, fake_input)
        input_ = torch.cat([input_, fake_input], 1)
        input_ = self.conv4(input_)
        CA_1 = self.encoder_level3(input_)
        feature1_1, feature1_2, feature1_3, bottle3 = self.Stage3(input_)
        bottle3 = bottle3 + CA_1
        output, output1 = self.conv8(bottle3, feature1_1, feature1_2)

        return output3, output2, output1, feature3_1, feature3_2, feature3_3, feature2_1, feature2_2, feature2_3, feature1_1, feature1_2, feature1_3


##########################################################################
class Fusion_Net(nn.Module):
    def __init__(self):
        super(Fusion_Net, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        nb_filter = [32, 32, 48, 64]
        kernel_size = 3
        stride = 1
        self.M = 16
        self.ed = EdgeDetect()
        self.sigmoid = nn.Sigmoid()
        self.up_eval = UpsampleReshape_eval()
        self.up = nn.Upsample(scale_factor=2)
        self.down = nn.Upsample(scale_factor=0.5)
        self.Stage1 = Fus_MDRB()
        self.Stage2 = Fus_MDRB()
        self.Stage3 = Fus_MDRB()

        self.conv6 = Recon_residual()
        self.conv7 = Recon_residual()
        self.conv8 = Recon_residual()

        self.encoder_level1 = [CAB(32, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.encoder_level2 = [CAB(32, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.encoder_level3 = [CAB(32, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.conv1 = ConvLayer(2, 1, 1, stride, 0)
        self.conv2 = ConvLayer(2, nb_filter[0], kernel_size, stride)
        self.conv3 = ConvLayer(2 * nb_filter[0], nb_filter[0], 1, stride, 0)
        self.conv4 = ConvLayer(2 * nb_filter[0], nb_filter[0], 1, stride, 0)

        self.conv9 = ConvLayer(nb_filter[0], 1, 1, stride, 0)

        self.ir_lin1 = nn.Linear(256, 128)
        self.ir_lin2 = nn.Linear(128, 288)
        self.rgb_lin1 = nn.Linear(256, 128)
        self.rgb_lin2 = nn.Linear(128, 288)

    def forward(self, ir_feature3_1, ir_feature3_2, ir_feature3_3, ir_feature2_1, ir_feature2_2, ir_feature2_3,
                ir_feature1_1, ir_feature1_2, ir_feature1_3, input, ir, rgb_hist, ir_hist, is_test):
        rgb_attention = ((input * 127.5) + 127.5) / 255
        ir_attention = ((ir * 127.5) + 127.5) / 255
        edgemap = self.ed(rgb_attention)
        edgemap2 = self.ed(ir_attention)
        edgemap_rgb = edgemap / (edgemap + edgemap2 + 0.00001)
        edgemap_rgb = (edgemap_rgb - 0.5) * 2
        input_ = torch.cat([input, edgemap_rgb], 1)
        ir_param = self.ir_lin1(ir_hist)
        ir_param = self.sigmoid(self.ir_lin2(ir_param))
        rgb_param = self.rgb_lin1(rgb_hist)
        rgb_param = self.sigmoid(self.rgb_lin2(rgb_param))

        ir_param_final = ir_param / (ir_param + rgb_param + 0.00001)
        rgb_param_final = rgb_param / (rgb_param + ir_param + 0.00001)
        input_ = self.conv2(input_)
        input_2 = F.interpolate(input_, scale_factor=0.5, mode='bilinear')
        input_3 = F.interpolate(input_2, scale_factor=0.5, mode='bilinear')

        if is_test:
            rgb_param_final = torch.unsqueeze(rgb_param_final, dim=0)
            ir_param_final = torch.unsqueeze(ir_param_final, dim=0)

        rgb_param_final = torch.unsqueeze(rgb_param_final, dim=2)
        rgb_param_final = torch.unsqueeze(rgb_param_final, dim=3)
        ir_param_final = torch.unsqueeze(ir_param_final, dim=2)
        ir_param_final = torch.unsqueeze(ir_param_final, dim=3)

        rgb_param_final_0 = rgb_param_final[:, 0:96, :, :]
        rgb_param_final_1 = rgb_param_final[:, 96:192, :, :]
        rgb_param_final_2 = rgb_param_final[:, 192:288, :, :]

        ir_param_final_0 = ir_param_final[:, 0:96, :, :]
        ir_param_final_1 = ir_param_final[:, 96:192, :, :]
        ir_param_final_2 = ir_param_final[:, 192:288, :, :]

        CA_3 = self.encoder_level1(input_3)
        feature3_1, feature3_2, feature3_3, bottle1, output_1_feature, output_2_feature, output_3_feature = self.Stage1(
            input_3, ir_feature3_1,
            ir_feature3_2,
            ir_feature3_3,
            ir_param_final_0,
            rgb_param_final_0)
        bottle1 = bottle1 + CA_3
        fake_input_2, output3 = self.conv6(bottle1, feature3_1, feature3_2)

        fake_input_2 = F.interpolate(fake_input_2, scale_factor=2, mode='bilinear')
        fake_input_2 = self.up_eval(input_2, fake_input_2)
        input_2 = torch.cat([input_2, fake_input_2], 1)
        input_2 = self.conv3(input_2)

        CA_2 = self.encoder_level1(input_2)
        feature2_1, feature2_2, feature2_3, bottle2, output_1_feature, output_2_feature, output_3_feature = self.Stage2(
            input_2, ir_feature2_1,
            ir_feature2_2,
            ir_feature2_3,
            ir_param_final_1,
            rgb_param_final_1)
        bottle2 = bottle2 + CA_2
        fake_input, output2 = self.conv7(bottle2, feature2_1, feature2_2)

        fake_input = F.interpolate(fake_input, scale_factor=2, mode='bilinear')
        fake_input = self.up_eval(input_, fake_input)
        input_ = torch.cat([input_, fake_input], 1)
        input_ = self.conv4(input_)
        CA_1 = self.encoder_level1(input_)
        feature1_1, feature1_2, feature1_3, bottle3, output_1_feature, output_2_feature, output_3_feature = self.Stage3(
            input_, ir_feature1_1,
            ir_feature1_2,
            ir_feature1_3,
            ir_param_final_2,
            rgb_param_final_2)
        bottle3 = bottle3 + CA_1
        output, output1 = self.conv8(bottle3, feature1_1, feature1_2)

        return output3, output2, output1

##########################################################################
