import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

EPOCH_NUMBER = 200
# IMAGE_NUMBER = 64
ITER = 0
# iters, loss_total, loss_fuse, loss_down8, loss_down4, loss_down2 = [], [], [], [], [], []
iters, loss = [], []
# epochs, ps, pn, us, un, total, fscore, ber = [], [], [], [], [], [], [], []

for i in range(1, EPOCH_NUMBER + 1):
    log_loss = open('./logs/loss' + str(i) + '.txt', 'r')
    for line in log_loss.readlines():
        ITER += 1
        iters.append(ITER)
        loss.append(float(line.split(' ')[1]))
        # loss_total.append(float(line.split(' ')[1]))
        # loss_fuse.append(float(line.split(' ')[2]))
        # loss_down8.append(float(line.split(' ')[3]))
        # loss_down4.append(float(line.split(' ')[4]))
        # loss_down2.append(float(line.split(' ')[5]))
    log_loss.close()

    # log_acc = open('./logs/acc' + str(i) + '.txt', 'r')
    # epochs.append(i)
    # line = log_acc.readlines()[-1]
    # ps.append(float(line.split(' ')[1]))
    # pn.append(float(line.split(' ')[2]))
    # us.append(float(line.split(' ')[3]))
    # un.append(float(line.split(' ')[4]))
    # total.append(float(line.split(' ')[5]))
    # fscore.append(float(line.split(' ')[6]))
    # ber.append(float(line.split(' ')[7]) / 100.0)
    # log_acc.close()

# losses = [loss_down8, loss_down4, loss_down2, loss_fuse, loss_total]
# labels = ['loss_downscale×8', 'loss_downscale×4', 'loss_downscale×2', 'loss_fusion', 'loss_total']
# colors = ['red', 'blue', 'green', 'orange', 'purple']
losses = [loss]
labels = ['loss']
colors = ['red']

for loss, label, color in zip(losses, labels, colors):
    myfig = plt.figure()
    ax = myfig.add_axes([0.175, 0.16, 0.675, 0.77])  # [left, bottom, width, height]
    # ax = myfig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.plot(iters, loss, label=label, c=color)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
    }
    legend = plt.legend(prop=font1)

    plt.tick_params(labelsize=20)

    # plt.xlabel(u'Iteration', font1)
    # plt.ylabel(u'Loss', font1)
    plt.xlabel('Iteration', font1)
    plt.ylabel('Loss', font1)

    # plt.show()
    plt.savefig('./{}.png'.format(label))

# accuracy plot

# name_list = ['阴影生产者精度', '非阴影生产者精度', '阴影用户精度', '非阴影用户精度']
# img1 = [0.9529520, 0.9556119, 0.9050907, 0.9785986]
# img2 = [0.9047732, 0.9881418, 0.9536676, 0.9746612]
# img3 = [0.9183298, 0.9873823, 0.9660454, 0.9686788]
# img_total = [0.9057709, 0.9642525, 0.8849983, 0.9717736]
#
# x = list(range(len(name_list)))
# total_width, n = 0.8, 4
# width = total_width / n
# plt.bar(x, img1, width=width, label='img1', fc='lightskyblue', ec='white')
#
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, img2, width=width, label='img2', fc='yellowgreen', ec='white')
#
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, img3, width=width, label='img3', fc='pink', ec='white')
#
# for i in range(len(x)):
#     x[i] = x[i] + width
# # plt.bar(x, img_total, width=width, label='img_total', tick_label=name_list, fc='r', ec='white')
# plt.bar(x, img_total, width=width, label='img_total', fc='r', ec='white')
#
# for i in range(len(x)):
#     x[i] = x[i] - width * 2
#     plt.text(i - 0.2, -0.1, name_list[i])

# name_num=0
# for name_num in range(len(name_list)):
#     plt.text(name_num-0.2,-10,name_list[name_num],rotation=25)

# plt.legend()
# plt.show()
