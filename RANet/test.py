from __future__ import print_function

import os
import time
import logging

import torch
import torch.utils.data as data

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.model_zoo import get_segmentation_model
from utils.score import SegmentationMetric
from utils.visualize import get_color_pallete
from utils.lr_scheduler import LRScheduler
from utils.score import SegmentationMetric
from utils import utils_image as util
from utils import utils_logger
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from utils import util_pix2pix

from train_path import parse_args
from models import unet
from PIL import Image


def eval(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    testset_name = 'whu-Val'
    model_name = 'U-Net'
    show_img = False  # default: False
    model_dir = './ckptmodels'
    outdir = './results/test'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # model path
    foldername = '{}_{}_size{}_batch{}'.format(args.model, args.dataset, args.crop_size,
                                                       args.batch_size)

    # result path
    result_name = '{}_{}_size{}_batch{}_epoch{}'.format(args.model, args.dataset, args.crop_size,
                                                       args.batch_size, args.epochs)
    result_path = os.path.join(outdir, result_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    resultFolder = result_path

    result_name = testset_name + '_' + model_name  # fixed
    logger_name = result_name

    logger_path = os.path.join(resultFolder, result_name)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    utils_logger.logger_info(logger_name, log_path=os.path.join(logger_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
    test_dataset = get_segmentation_dataset(args.dataset, split='test', mode='testval', **data_kwargs)


    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False)

    # create network
    model_path = os.path.join(model_dir, foldername)

    model = unet.get_unet(model=args.model, dataset=args.dataset, root=model_path,
                               aux=args.aux, pretrained=True, epoch=args.epochs,
                               base_size=args.base_size, crop_size=args.crop_size).to(device)


    model.eval()
    for i, (image, label, filename) in enumerate(test_loader):
        time_start = time.time()

        # print(image.size(), label.size())
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)

            pred = util.tensor2uint(outputs)
            target = util.tensor2uint(label)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------
            psnr = util.calculate_psnr(pred, target, border=0)
            ssim = util.calculate_ssim(pred, target, border=0)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(filename[0], psnr, ssim))
            util.imshow(np.concatenate([pred, target], axis=1),
                        title='Recovered / Ground-truth') if show_img else None

            # ------------------------------------
            # save results
            # ------------------------------------
            result_filename = '%s.tif' % (filename[0])
            util.imsave(pred, os.path.join(resultFolder, result_filename))
            # pred.save(outdir + '/%s.tif' % (filename))

        time_end = time.time()
        cost_time = time_end - time_start
        print('testing time is: %f' % (cost_time))

        # save name and time
        # filename_time = "" + "time.txt"
        filename_time = os.path.join(resultFolder, "time.txt")
        fc_time = open(filename_time, "a+")
        fc_time.write("%f\n" % (cost_time))

        # filename_name = "" + "Name.txt"
        filename_name = os.path.join(resultFolder, "Name.txt")
        fc_name = open(filename_name, "a+")
        fc_name.write("%s\n" % (filename))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

    # log.write(str(epoch) + ' ' + str(ave_psnr) + ' ' + str(ave_ssim) + ' ' + '\n')
    print('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':
    args = parse_args()
    save_result = True
    args.save_result = save_result
    args.epochs = 200
    print('Testing model: ', args.model)
    eval(args)
