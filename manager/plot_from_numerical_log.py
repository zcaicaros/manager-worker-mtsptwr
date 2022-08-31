import re
import numpy as np
import matplotlib.pyplot as plt

# parameters
size = 50
vehicles = 5
log_type = 'train'  # train or vali
sh_or_mh = 'MH'
plot_individual = False
node_embedding_type = 'gin'
hidden_dim = 32
beta = 100
reward_type = 'total'
anchor_box_font_size = 30
save = False
dpi = 100

if plot_individual:
    path = './training_logs/'
    if log_type == 'train':
        f = open('./training_logs/training_log_{}_{}_{}_{}_{}_{}_{}.txt'.format(
                    beta, size, vehicles, sh_or_mh, node_embedding_type, hidden_dim, reward_type), 'r')
    else:
        f = open('./training_logs/validation_log_{}_{}_{}_{}_{}_{}_{}.txt'.format(
                    beta, size, vehicles, sh_or_mh, node_embedding_type, hidden_dim, reward_type), 'r')
    logs = f.readlines()
    training_log = []
    validation_log = []
    # find decimals each log line
    digits = re.findall(r'\d*\.?\d+', logs[0])
    if log_type == 'train':
        log_arr = np.array(digits, dtype=float).reshape(int(len(digits)/3), -1)
    else:
        log_arr = np.array(digits, dtype=float).reshape(int(len(digits)/2), -1)

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
        plt.savefig('../paper_figure/{}.pdf'.format(str(size)+'-'+str(vehicles)+'-'+'rej'), dpi=dpi)
        plt.show()

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
        plt.savefig('../paper_figure/{}.pdf'.format(str(size) + '-' + str(vehicles) + '-' + 'len'), dpi=dpi)
        plt.show()
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
        plt.savefig('../paper_figure/{}.pdf'.format(str(size) + '-' + str(vehicles) + '-' + 'rej'), dpi=dpi)
        plt.show()

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
        plt.savefig('../paper_figure/{}.pdf'.format(str(size) + '-' + str(vehicles) + '-' + 'len'), dpi=dpi)
        plt.show()
else:
    n_set = [50, 100, 150]
    m_set = [5, 10]
    path = './training_logs/'

    log_arr_all = []
    if log_type == 'train':
        for n in n_set:
            for m in m_set:
                f = open('./training_logs/training_log_{}_{}_{}_{}_{}_{}_{}.txt'.format(
                    beta, n, m, sh_or_mh, node_embedding_type, hidden_dim, reward_type), 'r')
                logs = f.readlines()
                training_log = []
                validation_log = []
                # find decimals each log line
                digits = re.findall(r'\d*\.?\d+', logs[0])
                log_arr_all.append(np.array(digits, dtype=float).reshape(int(len(digits) / 3), -1))
    else:
        for n in n_set:
            for m in m_set:
                f = open('./training_logs/validation_log_{}_{}_{}_{}_{}_{}_{}.txt'.format(
                    beta, n, m, sh_or_mh, node_embedding_type, hidden_dim, reward_type), 'r')
                logs = f.readlines()
                training_log = []
                validation_log = []
                # find decimals each log line
                digits = re.findall(r'\d*\.?\d+', logs[0])
                log_arr_all.append(np.array(digits, dtype=float).reshape(int(len(digits) / 2), -1))
    log_arr_all = np.stack(log_arr_all)

    if log_type == 'vali':
        rej = log_arr_all[:, :, 0]
        length = log_arr_all[:, :, 1]
        cost = rej * beta + length
        x = np.array([i + 1 for i in range(log_arr_all.shape[1])])
        # rej curve
        plt.figure(figsize=(16, 11.6))
        plt.tick_params(labelsize=30)
        plt.xlabel('Every dpi iterations', {'size': 40})
        plt.ylabel('Rej.Rate', {'size': 40})
        plt.grid()
        plt.plot(x, rej[0], 'k', color='tab:blue', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, rej[1], 'k', color='tab:brown', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, rej[2], 'k', color='tab:cyan', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, rej[3], 'k', color='tab:green', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, rej[4], 'k', color='tab:purple', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, rej[5], 'k', color='tab:orange', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[1]))
        plt.tight_layout()
        plt.legend(fontsize=anchor_box_font_size)
        plt.savefig('../paper_figure/{}.pdf'.format('vali_rej_all_in_one'), dpi=dpi)
        plt.show()

        # len curve
        plt.figure(figsize=(16, 11.6))
        plt.tick_params(labelsize=30)
        plt.xlabel('Validation', {'size': 40})
        plt.ylabel('Length', {'size': 40})
        plt.grid()
        plt.tight_layout()
        plt.plot(x, length[0], 'k', color='tab:blue', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, length[1], 'k', color='tab:brown', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, length[2], 'k', color='tab:cyan', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, length[3], 'k', color='tab:green', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, length[4], 'k', color='tab:purple', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, length[5], 'k', color='tab:orange', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[1]))
        plt.legend(fontsize=anchor_box_font_size)
        plt.savefig('../paper_figure/{}.pdf'.format('vali_len_all_in_one'), dpi=100)
        plt.show()

        # cost curve
        plt.figure(figsize=(16, 11.6))
        plt.tick_params(labelsize=30)
        plt.xlabel('Validation', {'size': 40})
        plt.ylabel('Cost', {'size': 40})
        plt.grid()
        plt.tight_layout()
        plt.plot(x, cost[0], 'k', color='tab:blue', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, cost[1], 'k', color='tab:brown', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, cost[2], 'k', color='tab:cyan', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, cost[3], 'k', color='tab:green', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, cost[4], 'k', color='tab:purple', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, cost[5], 'k', color='tab:orange', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[1]))
        plt.legend(fontsize=anchor_box_font_size)
        if save:
            plt.savefig('../paper_figure/{}.pdf'.format('vali_cost_all_in_one'), dpi=100)
        plt.show()
    else:
        cost = log_arr_all[:, :, 0]
        rej = log_arr_all[:, :, 1]
        length = log_arr_all[:, :, 2]
        x = np.array([i + 1 for i in range(log_arr_all.shape[1])])

        # rej curve
        plt.figure(figsize=(16, 11.6))
        plt.tick_params(labelsize=30)
        plt.xlabel('Iteration', {'size': 40})
        plt.ylabel('Rej.Rate', {'size': 40})
        plt.grid()
        plt.plot(x, rej[0], 'k', color='tab:blue', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, rej[1], 'k', color='tab:brown', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, rej[2], 'k', color='tab:cyan', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, rej[3], 'k', color='tab:green', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, rej[4], 'k', color='tab:purple', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, rej[5], 'k', color='tab:orange', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[1]))
        plt.tight_layout()
        plt.legend(fontsize=anchor_box_font_size)
        plt.savefig('../paper_figure/{}.pdf'.format('train_rej_all_in_one'), dpi=dpi)
        plt.show()

        # len curve
        plt.figure(figsize=(16, 11.6))
        plt.tick_params(labelsize=30)
        plt.xlabel('Iteration', {'size': 40})
        plt.ylabel('Length', {'size': 40})
        plt.grid()
        plt.plot(x, length[0], 'k', color='tab:blue', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, length[1], 'k', color='tab:brown', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, length[2], 'k', color='tab:cyan', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, length[3], 'k', color='tab:green', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, length[4], 'k', color='tab:purple', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, length[5], 'k', color='tab:orange', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[1]))
        plt.tight_layout()
        plt.legend(fontsize=anchor_box_font_size)
        plt.savefig('../paper_figure/{}.pdf'.format('train_len_all_in_one'), dpi=dpi)
        plt.show()

        # cost curve
        plt.figure(figsize=(16, 11.6))
        plt.tick_params(labelsize=30)
        plt.xlabel('Iteration', {'size': 40})
        plt.ylabel('Cost', {'size': 40})
        plt.grid()
        plt.plot(x, cost[0], 'k', color='tab:blue', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, cost[1], 'k', color='tab:brown', label='n=' + str(n_set[0]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, cost[2], 'k', color='tab:cyan', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, cost[3], 'k', color='tab:green', label='n=' + str(n_set[1]) + ', ' + 'm=' + str(m_set[1]))
        plt.plot(x, cost[4], 'k', color='tab:purple', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[0]))
        plt.plot(x, cost[5], 'k', color='tab:orange', label='n=' + str(n_set[2]) + ', ' + 'm=' + str(m_set[1]))
        plt.tight_layout()
        plt.legend(fontsize=anchor_box_font_size)
        if save:
            plt.savefig('../paper_figure/{}.pdf'.format('train_cost_all_in_one'), dpi=dpi)
        plt.show()
