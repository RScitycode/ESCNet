"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
        python train.py  --name UAV-WG-jpg_cyclegan_batch10_loadsize256_unet256_lsgan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

To do this, you should have visdom installed and a server running by the command: python -m visdom.server.
The default server URL is http://localhost:8097.


See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # opt = opt.parse_args()
    # opt.dataroot = './datasets/UAV-WG-jpg'
    # opt.name = 'UAV-WG-jpg_pix2pix_batch10_loadsize256_unet256_wgangp'
    # opt.model = 'pix2pix'
    # opt.direction = 'AtoB'
    # opt.batch_size = 10
    # opt.load_size = 256
    # opt.netG = 'unet_256'
    # opt.gan_mode = 'wgangp'


    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            # total_iters += opt.batch_size
            # epoch_iter += opt.batch_size
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

#LYQ修改
    def validation(self, epoch, log):
    # def test(self, epoch, log):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        # self.args.device = torch.device('cpu')

        # self.metric.reset()
        # t_ps = t_pn = t_us = t_un = t_t = t_f = t_b = 0.0
        # total = len(self.val_loader)
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        testset_name = 'UAV-WG-Val'
        model_name = 'pix2pix'
        show_img = False                 # default: False
        outdir = './results/val'

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