import sys
from torchvision.utils import save_image
from torch.autograd import Variable
import datetime
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

writer = SummaryWriter()


def step(self, epoch):
    return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def fus_train(IR_Net_model, Fusion_Net_model, fus_optim, MSE_loss, Spa_loss, per_loss, fus_dataloader, epochs, sample_interval, checkpoint_interval, dataset_name):
    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # -------------
    #  Training
    # -------------

    prev_time = time.time()
    for epoch in range(epochs):
        for i, (rgb_target, ir_target, rgb_hist, ir_hist) in enumerate(fus_dataloader):

            # -------------------
            # Configure input
            # -------------------

            real_rgb_imgs = Variable(rgb_target.type(Tensor))
            real_ir_imgs = Variable(ir_target.type(Tensor))
            rgb_hist = Variable(rgb_hist.type(Tensor))
            ir_hist = Variable(ir_hist.type(Tensor))
            real_ir_imgs_2 = F.interpolate(real_ir_imgs, scale_factor=0.5, mode='bilinear')
            real_ir_imgs_3 = F.interpolate(real_ir_imgs_2, scale_factor=0.5, mode='bilinear')
            real_rgb_imgs_2 = F.interpolate(real_rgb_imgs, scale_factor=0.5, mode='bilinear')
            real_rgb_imgs_3 = F.interpolate(real_rgb_imgs_2, scale_factor=0.5, mode='bilinear')
            fus_optim.zero_grad()

            # -------------------
            # train Fusion_Net
            # -------------------

            is_test = False
            _, _, ir_imgs1, feature3_1, feature3_2, feature3_3, feature2_1, feature2_2, feature2_3, feature1_1, feature1_2, feature1_3 = IR_Net_model(
                real_rgb_imgs,
                real_ir_imgs)
            fusion_imgs3, fusion_imgs2, fusion_imgs1 = Fusion_Net_model(feature3_1, feature3_2, feature3_3,
                                                                              feature2_1, feature2_2, feature2_3,
                                                                              feature1_1, feature1_2, feature1_3,
                                                                              real_rgb_imgs,
                                                                              real_ir_imgs, rgb_hist, ir_hist,
                                                                              is_test)

            # -------------
            # total_loss
            # -------------
            
            mse_loss = MSE_loss(input=fusion_imgs3, target=real_rgb_imgs_3) \
                       + MSE_loss(input=fusion_imgs3, target=real_ir_imgs_3) \
                       + MSE_loss(input=fusion_imgs2, target=real_rgb_imgs_2) \
                       + MSE_loss(input=fusion_imgs2, target=real_ir_imgs_2) \
                       + MSE_loss(input=fusion_imgs1, target=real_rgb_imgs) \
                       + MSE_loss(input=fusion_imgs1, target=real_ir_imgs)
            loss_spa = 0.5 * torch.mean(Spa_loss(fusion_imgs1, real_rgb_imgs)) + 0.5 * torch.mean(
                Spa_loss(fusion_imgs1, real_ir_imgs))
            loss_per = 0.5 * per_loss(fusion_imgs1, real_rgb_imgs) + 0.5 * per_loss(fusion_imgs1, real_ir_imgs)
            fuse_loss = (mse_loss / 6) + 0.05 * loss_spa + 0.5 * loss_per

            total_loss = fuse_loss

            total_loss.backward()
            fus_optim.step()

            # -----------------
            #  Log Progress
            # -----------------

            batches_done = epoch * len(fus_dataloader) + i
            batches_left = epochs * len(fus_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [mse loss: %f][spa loss: %f][per loss: %f] ETA: %s batches_done : %d"
                % (
                    epoch,
                    epochs,
                    i,
                    len(fus_dataloader),
                    mse_loss.item(),
                    loss_spa.item(),
                    loss_per.item(),
                    time_left,
                    batches_done,
                )
            )

            writer.add_scalar("Loss/mse_loss", mse_loss, batches_done)
            writer.add_scalar("Loss/spa_loss", loss_spa, batches_done)
            writer.add_scalar("Loss/per_loss", loss_per, batches_done)

            # -----------------------
            # output visualization
            # -----------------------

            if batches_done % sample_interval == 0:
                save_image(fusion_imgs1.data[:16], "train_images/fake_%d.png" % batches_done, nrow=4, normalize=True)
                save_image(ir_imgs1.data[:16], "train_images/ir_fake_%d.png" % batches_done, nrow=4,
                           normalize=True)
                save_image(real_ir_imgs.data[:16], "train_images/ir_real_%d.png" % batches_done, nrow=4, normalize=True)
                save_image(real_rgb_imgs.data[:16], "train_images/real_rgb_%d.png" % batches_done, nrow=4,
                           normalize=True)

            # -------------------------
            # Save model checkpoints
            # -------------------------

            if checkpoint_interval != -1 and batches_done % checkpoint_interval == 0:
                torch.save(Fusion_Net_model.state_dict(),
                           "saved_models/%s/Fusion_Net_model_%d.pth" % (dataset_name, batches_done))
