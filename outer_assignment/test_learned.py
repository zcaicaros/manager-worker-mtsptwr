import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from outer_assignment.policy_ import Policy, action_sample, get_reward, action_greedy
import torch
import time
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
from utils import load_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def test(model, dataset, inner_model, assgn_type, cluster_type, show_cluster, no_agent, beta, device):
    # to batch graph
    adj = torch.ones([dataset.shape[0], dataset.shape[1], dataset.shape[1]])  # adjacent matrix fully connected
    data_list = [Data(x=dataset[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t(), as_tuple=False) for i in range(dataset.shape[0])]
    batch_graph = Batch.from_data_list(data_list=data_list).to(device)

    dataset_without_depot = dataset[:, 1:, :]

    # get pi
    action = None
    pi = model(batch_graph, n_nodes=data.shape[1], n_batch=dataset.shape[0])
    if assgn_type == 'sampling':
        action, _ = action_sample(pi)
        # plot cluster
        if show_cluster:
            y_drl = action.squeeze().cpu().numpy()
            points = dataset_without_depot[0]
            plt.scatter(points[:, 0], points[:, 1], c=y_drl, s=50, cmap='viridis')
            plt.show()
    elif assgn_type == 'greedy':
        action = action_greedy(pi)
        # plot cluster
        if show_cluster:
            y_drl = action.squeeze().cpu().numpy()
            # print(y_drl)
            points = dataset_without_depot[0]
            plt.scatter(points[:, 0], points[:, 1], c=y_drl, s=100, cmap='viridis')
            plt.show()
    else:
        kmeans_assgn = []
        if cluster_type == 'temporal+spacial':
            for i in range(dataset_without_depot.shape[0]):
                # kmeans assignment
                assgn = KMeans(n_clusters=no_agent, random_state=0, max_iter=1000).fit(dataset_without_depot[i])
                kmeans_assgn.append(assgn.labels_)
                # plot
                if show_cluster:
                    points = dataset_without_depot[i]
                    y_kmeans = assgn.predict(points)
                    plt.scatter(points[:, 0], points[:, 1], c=y_kmeans, s=50, cmap='viridis')
                    centers = assgn.cluster_centers_
                    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
                    plt.show()
            kmeans_action = np.stack(kmeans_assgn)
            action = torch.from_numpy(kmeans_action)
        elif cluster_type == 'spacial':
            # kmeans assignment
            assgn = KMeans(n_clusters=no_agent, random_state=0, max_iter=1000).fit(dataset_without_depot[0][:, :2])
            kmeans_assgn.append(assgn.labels_)
            # plot
            if show_cluster:
                points = dataset_without_depot[0][:, :2]
                y_kmeans = assgn.predict(points)
                plt.scatter(points[:, 0], points[:, 1], c=y_kmeans, s=50, cmap='viridis')
                centers = assgn.cluster_centers_
                plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
                plt.show()
            kmeans_action = np.stack(kmeans_assgn)
            action = torch.from_numpy(kmeans_action)

    # get reward for each instance
    assert action is not None
    reward_, rej_, length_ = get_reward(action, data.to(device), no_agent, inner_model, beta=beta)  # reward: tensor [batch, 1]

    # count assignment for each vehicle
    counts = []
    for i in range(no_agent):
        counts.append((action == i).sum().item())

    return reward_[0], rej_[0], length_[0], counts, action


if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)  # 1

    n_vehicles = 10
    manager_size = [150, n_vehicles]
    sh_or_mh = 'MH'
    node_embd_type = 'mlp'
    GIN_dim = 32
    n_nodes = [50, 100, 150, 200, 300, 400, 500]
    # n_nodes = [50]
    batch_size = 100
    beta = 100
    assignment_type = 'greedy'  # 'sampling', 'greedy', or 'k-means'
    k_means_cluster_type = 'spacial'  # 'temporal+spacial', or 'spacial'
    show_cluster = False

    # load net
    policy = Policy(vehicle_embd_type=sh_or_mh, node_embedding_type=node_embd_type,
                    in_chnl=4, hid_chnl=GIN_dim, n_agent=n_vehicles, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    path = '../pretrained_assgnmt_beta'+str(beta)+'/{}.pth'.format(str(manager_size[0]) + '_' + str(n_vehicles) + '_' + sh_or_mh + '_' + str(GIN_dim))
    # print(path)
    # load assignment network
    policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
    policy.eval()

    '''for name, param in policy.named_parameters():
        if param.requires_grad:
            print(name, param.data)'''

    for size in n_nodes:
        print('Size:', size, 'Agents:', n_vehicles, 'Worker-size:', str(int(size / n_vehicles)), 'Manager-size:', str(manager_size[0])+'-'+str(n_vehicles))
        # load routing network
        validation_net = load_model('../pretrained_inner_route_planner_beta'+str(beta)+'/' + str(int(size / n_vehicles)) + '.pt', dev)
        validation_net.eval()
        validation_net.to(dev)
        validation_net.decode_type = 'greedy'

        testing_data = torch.load(
            '../testing-instances/' + str(size) + '/testing_data_' + str(size) + '_' + str(100))
        # print(testing_data[0][0])
        objs_per_seed = []
        rejs_per_seed = []
        lengths_per_seed = []

        objs = []
        rejs = []
        lengths = []
        start = time.time()
        count_max = []
        count_min = []
        count_mean = []
        for j in range(batch_size):
            # prepare random generated tesing data
            '''location = torch.rand(size=[1, size-1, 2])  # nodes - 1: locations without depot
            win_start = 3 * torch.rand(size=[1, size-1, 1])  # nodes - 1: start time without depot
            win_end = 3 + win_start
            location_with_tw = torch.cat([location, win_start, win_end], dim=-1)  # locations+tw without depot
            depot = torch.tensor([0.5, 0.5, 0, 10], dtype=torch.float).repeat(1, 1, 1)  # constant depot
            data = torch.cat([depot, location_with_tw], dim=1)  # final instances with depot'''

            # use preloaded testing data
            data = testing_data[j].unsqueeze(0)

            # testing
            obj, rej, length, cts, assign = test(policy, data, validation_net, assignment_type, k_means_cluster_type, show_cluster, n_vehicles, beta, dev)
            objs.append(obj)
            rejs.append(rej)
            lengths.append(length)
            count_max.append(max(cts))
            count_min.append(min(cts))
            count_mean.append(sum(cts)/len(cts))
            # print('Instance', j, ':', 'rej.rate', format(rej, '.5f'), 'length:', format(length, '.5f'))
        end = time.time()
        # np.save('./counts/count_max'+'_'+str(size)+'_'+str(n_vehicles), np.array(count_max))
        # np.save('./counts/count_min'+'_'+str(size)+'_'+str(n_vehicles), np.array(count_min))
        # np.save('./counts/count_mean' + '_' + str(size) + '_' + str(n_vehicles), np.array(count_mean))
        objs_per_seed.append(format(np.array(objs).mean(), '.5f'))
        rejs_per_seed.append(format(np.array(rejs).mean(), '.5f'))
        lengths_per_seed.append(format(np.array(lengths).mean(), '.5f'))
        print('Size', size, 'Vehicles', n_vehicles, ':')
        print('Rej.rate:', rejs_per_seed)
        print('Length:', lengths_per_seed)
        print('Cost:', objs_per_seed)
        print('Time(s):', (end - start)/batch_size)
        print()




