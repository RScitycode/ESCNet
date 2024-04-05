"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import torch.backends.cudnn as cudnn
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
import numpy as np
import cv2
from util.util import tensor2im
from PIL import Image


def cal_psnr(img1, img2):
    return PSNR(img1, img2)


def cal_ssim(img1, img2):
    return SSIM(img1, img2, channel_axis=2)


def cal_mse(img1, img2):
    return MSE(img1, img2)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # opt.dataroot = './datasets/UAV-WG/A/val'
    # opt.dataroot = './datasets/UAV-WG/A/train_ori'
    opt.dataroot = './datasets/test/shaded'

    # # opt.dataroot = './datasets/facades'
    # opt.dataset_mode = 'aligned'
    opt.name = 'UAV-WG_pix2pix_batch4_loadsize512_unet256_vanilla'
    # opt.gpu_ids = -1
    # opt.name = 'UAV-WG-jpg_pix2pix'
    # # opt.name = 'UAV-WG-jpg_pix2pix_batch10_loadsize256'
    # # opt.name = 'facades_pix2pix'
    # opt.model = 'pix2pix'
    # opt.direction = 'AtoB'
    # opt.batch_size = 5
    # # opt.direction = 'BtoA'
    # # opt.netG = 'unet_256'
    # opt.netG = 'resnet_9blocks'
    # # opt.norm = 'instance'

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # if opt.eval:
    #     model.eval()
    model.eval()
    # psnr_list1, psnr_list2, psnr_list3, psnr_list4 = [[] for _ in range(4)]
    # rmse_list1, rmse_list2, rmse_list3, rmse_list4 = [[] for _ in range(4)]
    # ssim_list1, ssim_list2, ssim_list3, ssim_list4 = [[] for _ in range(4)]
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_fake = tensor2im(visuals['fake'])
        img_path = model.get_image_paths()     # get image paths/datasets/UAV-WG/test/shaded
    #     img_name = img_path[0].split('\\')[-1]
    #     img_unshaded = cv2.imread(os.path.join('./datasets/UAV-WG/test/unshaded', img_name), cv2.IMREAD_UNCHANGED)
    #     img_unshaded = Image.open(os.path.join('./datasets/test/unshaded', img_name)).convert('RGB')
    #     img_unshaded = np.array(img_unshaded)
    #     psnr = cal_psnr(img_fake, img_unshaded)
    #     rmse = np.sqrt(cal_mse(img_fake, img_unshaded))
    #     # ssim = cal_ssim(img_fake, img_unshaded)
    #     print(f'{img_name},PSNR:{psnr:.2f}, RMSE:{rmse:.2f}')
    #     if 'lihe' in img_name:
    #         psnr_list1.append(psnr)
    #         rmse_list1.append(rmse)
    #     if 'nanhu' in img_name:
    #         psnr_list2.append(psnr)
    #         rmse_list2.append(rmse)
    #     if 'wg' in img_name:
    #         psnr_list3.append(psnr)
    #         rmse_list3.append(rmse)
    #     if 'whu' in img_name:
    #         psnr_list4.append(psnr)
    #         rmse_list4.append(rmse)
        if i % 1 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # print(f'lihe avg psnr:{(sum(psnr_list1)/len(psnr_list1)):.2f}, avg rmse:{(sum(rmse_list1)/len(rmse_list1)):.2f}')
    # print(f'nanhu avg psnr:{(sum(psnr_list2) / len(psnr_list2)):.2f}, avg rmse:{(sum(rmse_list2)/len(rmse_list2)):.2f}')
    # print(f'wg avg psnr:{(sum(psnr_list3) / len(psnr_list3)):.2f}, avg rmse:{(sum(rmse_list3)/len(rmse_list3)):.2f}')
    # print(f'whu avg psnr:{(sum(psnr_list4) / len(psnr_list4)):.2f}, avg rmse:{(sum(rmse_list4)/len(rmse_list4)):.2f}')
    webpage.save()  # save the HTML
