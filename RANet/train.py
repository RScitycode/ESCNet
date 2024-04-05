import argparse
import time
import os
import glob
import re
import logging

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
from data_loader import get_segmentation_dataset
# from models.model_zoo import get_segmentation_model
from utils.lr_scheduler import LRScheduler
from utils.score import SegmentationMetric
from utils import utils_image as util
from utils import utils_logger
from collections import OrderedDict
import numpy as np
from models import unet
from data_loader import uav

def parse_args():
    parser = argparse.ArgumentParser(description='Shadow Removal Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='unet',
                        help='model name (default: fcn32s)')
    parser.add_argument('--dataset', type=str, default='uav',
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N', ##改动
                        help='input batch size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    # checking point
    parser.add_argument('--resume', type=str, default='./ckptmodels',
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-folder', default='./ckptmodels',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-logs', default='./logs',
                        help='Directory for saving logs')
    parser.add_argument('--logPath', default='./logs',
                        help='Directory for saving logs')
    parser.add_argument('--modelFolder', default='./ckptmodels',
                        help='Directory for saving models')
    parser.add_argument('--foldername', default='',
                        help='Name for folder')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    args.device = device
    print(args)
    return args

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'unet_uav_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*unet_uav_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # model path
        directory = os.path.expanduser(self.args.save_folder)
        foldername = '{}_{}_size{}_batch{}'.format(self.args.model, self.args.dataset, self.args.crop_size,
                                                           self.args.batch_size)
        # self.args.foldername = foldername
        model_path = os.path.join(directory, foldername)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.args.modelFolder = model_path

        # log path
        log_directory = os.path.expanduser(self.args.save_logs)
        log_path = os.path.join(log_directory, foldername)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.args.logPath = log_path

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}

        train_dataset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)  # get train datasets folder path
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', **data_kwargs)
        # train_dataset = uav.UAVSegmentation(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        # val_dataset = uav.UAVSegmentation(args.dataset, split='val', mode='testval', **data_kwargs)
        # test_dataset = get_segmentation_dataset(args.dataset, split='test', mode='testval', **data_kwargs)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            drop_last=True,
                                            shuffle=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          drop_last=False,
                                          shuffle=False)

        # create network
        # self.model = get_segmentation_model(model=args.model, dataset=args.dataset,
        #                                     aux=args.aux, norm_layer=nn.BatchNorm2d,
        #                                     base_size=args.base_size, crop_size=args.crop_size).to(args.device)
        self.model = unet.get_unet(model=args.model, dataset=args.dataset,
                                            aux=args.aux, norm_layer=nn.BatchNorm2d,
                                            base_size=args.base_size, crop_size=args.crop_size).to(args.device)
        print(self.model)

        # create criterion
        # self.criterion = nn.BCEWithLogitsLoss().to(args.device)
        self.criterion = nn.MSELoss().to(args.device)

        # resume checkpoint if needed
        # args.start_epoch = findLastCheckpoint(save_dir=args.resume)  # load the last model
        args.start_epoch = findLastCheckpoint(save_dir=model_path)  # load the last model
        if args.start_epoch > 0:
            name, ext = os.path.splitext(model_path)
            # name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            print('Resuming by loading epoch %d' % args.start_epoch)
            # self.model.load_state_dict(torch.load(os.path.join(args.resume, 'unet_uav_%d.pth' % args.start_epoch),
            #                                       map_location=lambda storage, loc: storage))
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'unet_uav_%d.pth' % args.start_epoch),
                                                  map_location=lambda storage, loc: storage))


        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9)

    def train(self):
        # cur_iters = self.args.start_epoch * len(self.train_loader) / self.args.batch_size
        cur_iters = self.args.start_epoch * len(self.train_loader)
        start_time = time.time()

        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
            # log_loss = open('./logs/loss' + str(epoch + 1) + '.txt', 'w+')
            # log_acc = open('./logs/acc' + str(epoch + 1) + '.txt', 'w+')
            log_loss = open(self.args.logPath + '/loss' + str(epoch + 1) + '.txt', 'w+')
            log_acc = open(self.args.logPath + '/acc' + str(epoch + 1) + '.txt', 'w+')
            for i, (images, targets,masks, filename) in enumerate(self.train_loader):#加掩膜
                cur_lr = self.lr_scheduler(cur_iters)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr

                images = images.to(self.args.device)
                targets = targets.to(self.args.device)
                masks = masks.to(self.args.device)
                # print('images:', images.size())
                pred = self.model(images)

                pred = pred.to(dtype=torch.float32)
                targets = targets.to(dtype=torch.float32)
                masks = masks.to(dtype=torch.float32)

                pred_1 = pred * (1-masks)
                targets_1 = targets * (1 - masks)

                loss = self.criterion(pred_1, targets_1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cur_iters += 1
                log_loss.write(str(cur_iters) + ' ' + str(loss.item()) + '\n')
                print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f' % (
                        epoch + 1, self.args.epochs, i + 1, len(self.train_loader),
                        time.time() - start_time, cur_lr, loss.item()))

            # save every 5 epochs
            if (epoch + 1) % 5 == 0:
                print('Saving state, epoch:', epoch + 1)
                self.save_checkpoint(epoch + 1)
            # eval every epoch
            if not self.args.no_val:
                self.validation(epoch + 1, log_acc)
                # self.test(epoch + 1, log_acc)
            log_loss.close()
            log_acc.close()

    def validation(self, epoch, log):

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        testset_name = 'UAV-WG-Val9'
        model_name = 'U-Net'
        show_img = False                 # default: False
        outdir = './results/val1'

        result_name = testset_name + '_' + model_name  # fixed
        logger_name = result_name

        logger_path = os.path.join(outdir, result_name)
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        utils_logger.logger_info(logger_name, log_path=os.path.join(logger_path, logger_name + '.log'))
        logger = logging.getLogger(logger_name)

        self.model.eval()

        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.args.device)
            # img_name, ext = os.path.splitext(filename)

            with torch.no_grad():
                outputs = self.model(image)
                pred = util.tensor2uint(outputs)
                target = util.tensor2uint(target)

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
                result_filename = '%s.tif'% (filename[0])
                util.imsave(pred, os.path.join(outdir, result_filename))
                # pred.save(outdir + '/%s.tif' % (filename))


        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr,ave_ssim))

        log.write(str(epoch) + ' ' + str(ave_psnr) + ' ' + str(ave_ssim)  + ' ' + '\n')
        print('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr,ave_ssim))

    def save_checkpoint(self, epoch, is_best=False):
        """Save Checkpoint"""
        if is_best:
            best_filename = '{}_{}_best_model.pth'.format(self.args.model, self.args.dataset)
            save_path_best = os.path.join(self.args.modelFolder, best_filename)
            torch.save(self.model.state_dict(), save_path_best)
        else:
            filename = '{}_{}_{}.pth'.format(self.args.model, self.args.dataset, epoch)

            # save_path = os.path.join(model_path, filename)
            save_path = os.path.join(self.args.modelFolder, filename)
            torch.save(self.model.state_dict(), save_path)


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    # args.eval = True
    if args.eval:
        print('Evaluation model: ', args.resume)
        log_acc = open('./logs/acc' + str(args.start_epoch) + '.txt', 'w+')
        trainer.validation(args.start_epoch,log_acc)
    else:
        print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch + 1, args.epochs))
        trainer.train()
