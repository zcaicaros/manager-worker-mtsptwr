import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# parameters
size = 50
vehicles = 10
log_type = 'train'  # train or vali
sh_or_mh = 'MH'


path = './training_logs/'
if log_type == 'train':
    f = open(path + 'itr_log_' + str(size) + '-' + str(vehicles) + '_' + sh_or_mh + '.txt', 'r')
else:
    f = open(path + 'vali_log_' + str(size) + '-' + str(vehicles) + '_' + sh_or_mh + '.txt', 'r')
logs = f.readlines()
training_log = []
validation_log = []
# find decimals each log line
digits = re.findall(r'\d*\.?\d+', logs[0])
if log_type == 'train':
    log_arr = np.array(digits, dtype=np.float).reshape(int(len(digits)/3), -1)
else:
    log_arr = np.array(digits, dtype=np.float).reshape(int(len(digits)/2), -1)

# plot log...
if log_type == 'vali':
    rej = log_arr[:, 0]
    length = log_arr[:, 1]
    # print(min(rej))
    # print(min(length))
    # print(rej[23])
    # print(length[23])
    x = np.array([i+1 for i in range(log_arr.shape[0])])

    # rej curve
    _, ax = plt.subplots()
    plt.figure(figsize=(16, 11.6))
    plt.tick_params(labelsize=30)
    plt.xlabel('Every 100 iterations', {'size': 40})
    plt.ylabel('Rej.Rate', {'size': 40})
    plt.grid()
    # add anchor text box upper right corner
    textstr = 'Size:' + str(size) + '\n' + 'Vehicles:' + str(vehicles)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.815, 0.96, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.plot(x, rej, 'k', color='tab:blue')
    plt.tight_layout()
    plt.savefig('../paper_figure/{}.pdf'.format(str(size)+'-'+str(vehicles)+'-'+'rej'), dpi=100)
    # plt.show()

    # len curve
    _, ax = plt.subplots()
    plt.figure(figsize=(16, 11.6))
    plt.tick_params(labelsize=30)
    plt.xlabel('Validation', {'size': 40})
    plt.ylabel('Length', {'size': 40})
    plt.grid()
    # add anchor text box upper right corner
    textstr = 'Size:' + str(size) + '\n' + 'Vehicles:' + str(vehicles)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.815, 0.96, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.plot(x, length, 'k', color='tab:blue')
    plt.tight_layout()
    plt.savefig('../paper_figure/{}.pdf'.format(str(size) + '-' + str(vehicles) + '-' + 'len'), dpi=100)
    # plt.show()
else:
    cost = log_arr[:, 0]
    rej = log_arr[:, 1]
    # print(rej[0:100])
    length = log_arr[:, 2]
    # print(length[0:100])
    x = np.array([i+1 for i in range(log_arr.shape[0])])

    # rej curve
    _, ax = plt.subplots()
    plt.figure(figsize=(16, 11.6))
    plt.tick_params(labelsize=30)
    plt.xlabel('Iteration', {'size': 40})
    plt.ylabel('Rej.Rate', {'size': 40})
    plt.grid()
    # add anchor text box upper right corner
    textstr = 'Size:' + str(size) + '\n' + 'Vehicles:' + str(vehicles)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.815, 0.96, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.plot(x, rej, 'k', color='tab:blue')
    plt.tight_layout()
    plt.savefig('../paper_figure/{}.pdf'.format(str(size) + '-' + str(vehicles) + '-' + 'rej'), dpi=100)
    # plt.show()

    # len curve
    _, ax = plt.subplots()
    plt.figure(figsize=(16, 11.6))
    plt.tick_params(labelsize=30)
    plt.xlabel('Iteration', {'size': 40})
    plt.ylabel('Length', {'size': 40})
    plt.grid()
    # add anchor text box upper right corner
    textstr = 'Size:' + str(size) + '\n' + 'Vehicles:' + str(vehicles)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.815, 0.96, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.plot(x, length, 'k', color='tab:blue')
    plt.tight_layout()
    plt.savefig('../paper_figure/{}.pdf'.format(str(size) + '-' + str(vehicles) + '-' + 'len'), dpi=100)
    # plt.show()
