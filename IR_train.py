import sys
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.autograd import Variable
import datetime
import time
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def step(self, epoch):
    return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def ir_train(IR_Net_model, ir_optim, MSE_loss, SSIM_loss, dataloader, batch_size, epochs, sample_interval, checkpoint_interval,
             dataset_name):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    prev_time = time.time()
    for epoch in range(epochs):
        for i, (rgb_target, ir_target) in enumerate(dataloader):

            # ------------------
            # Configure input
            # ------------------

            real_rgb_imgs = Variable(rgb_target.type(Tensor))
            real_ir_imgs = Variable(ir_target.type(Tensor))
            real_ir_imgs_2 = F.interpolate(real_ir_imgs, scale_factor=0.5, mode='bilinear')
            real_ir_imgs_3 = F.interpolate(real_ir_imgs_2, scale_factor=0.5, mode='bilinear')
            ir_optim.zero_grad()

            # ---------------
            # train IR_Net
            # ---------------

            ir_fake_imgs3, ir_fake_imgs2, ir_fake_imgs1, _, _, _, _, _, _, _, _, _ = IR_Net_model(real_rgb_imgs,
                                                                                                  real_ir_imgs)

            # -------------
            # total_loss
            # -------------

            ir_loss = 0.01 * MSE_loss(input=ir_fake_imgs3, target=real_ir_imgs_3) + 0.01 * MSE_loss(input=ir_fake_imgs2,
                                                                                                    target=real_ir_imgs_2) + 0.01 * MSE_loss(
                input=ir_fake_imgs1, target=real_ir_imgs) + (1 - SSIM_loss(ir_fake_imgs1.cuda(), real_ir_imgs.cuda()))

            ir_loss.backward()
            ir_optim.step()

            # -----------------
            #  Log Progress
            # -----------------

            batches_done = epoch * len(dataloader) + i
            batches_left = epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # ------------
            # Print log
            # ------------

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [ir loss: %f] ETA: %s batches_done : %d"
                % (
                    epoch,
                    epochs,
                    i,
                    len(dataloader),
                    ir_loss.item(),
                    time_left,
                    batches_done,
                )
            )

            # -----------------------
            # output visualization
            # -----------------------

            if batches_done % sample_interval == 0:
                save_image(ir_fake_imgs1.data[:batch_size], "train_images/ir_fake1_%d.png" % batches_done, nrow=4,
                           normalize=True)
                save_image(ir_fake_imgs2.data[:batch_size], "train_images/ir_fake2_%d.png" % batches_done, nrow=4,
                           normalize=True)
                save_image(ir_fake_imgs3.data[:batch_size], "train_images/ir_fake3_%d.png" % batches_done, nrow=4,
                           normalize=True)
                save_image(real_ir_imgs.data[:batch_size], "train_images/ir_real_%d.png" % batches_done, nrow=4,
                           normalize=True)

            # -------------------------
            # Save model checkpoints
            # -------------------------

            if checkpoint_interval != -1 and batches_done % checkpoint_interval == 0:
                torch.save(IR_Net_model.state_dict(),
                           "saved_models/%s/IR_Net_model_%d.pth" % (dataset_name, batches_done))
