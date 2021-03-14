import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool


class Net(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl):
        super(Net, self).__init__()

        ## init projection
        # 1st mlp layer
        self.lin1_1 = torch.nn.Linear(in_chnl, hid_chnl)
        self.bn1_1 = torch.nn.BatchNorm1d(hid_chnl)
        self.lin1_2 = torch.nn.Linear(hid_chnl, hid_chnl)

        ## GIN conv layers
        nn1 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        self.conv1 = GINConv(nn1, eps=0, train_eps=False, aggr='mean')
        self.bn1 = torch.nn.BatchNorm1d(hid_chnl)
        nn2 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        self.conv2 = GINConv(nn2, eps=0, train_eps=False, aggr='mean')
        self.bn2 = torch.nn.BatchNorm1d(hid_chnl)
        nn3 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        self.conv3 = GINConv(nn3, eps=0, train_eps=False, aggr='mean')
        self.bn3 = torch.nn.BatchNorm1d(hid_chnl)
        # nn4 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        # self.conv4 = GINConv(nn4, eps=0, train_eps=False, aggr='mean')
        # self.bn4 = torch.nn.BatchNorm1d(hid_chnl)

        ## layers used in graph pooling
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(1+3):  # 1+x: 1 projection layer + x GIN layers
            self.linears_prediction.append(nn.Linear(hid_chnl, hid_chnl))

    def forward(self, x, edge_index, batch):

        # init projection
        h = self.lin1_2(F.relu(self.bn1_1(self.lin1_1(x))))
        hidden_rep = [h]

        # GIN conv
        h = F.relu(self.bn1(self.conv1(h, edge_index)))
        node_pool_over_layer = h
        hidden_rep.append(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        node_pool_over_layer += h
        hidden_rep.append(h)
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        node_pool_over_layer += h
        hidden_rep.append(h)
        # h = F.relu(self.bn4(self.conv4(h, edge_index)))
        # node_pool_over_layer += h
        # hidden_rep.append(h)

        gPool_over_layer = 0
        # Graph pool
        for layer, layer_h in enumerate(hidden_rep):
            g_pool = global_mean_pool(layer_h, batch)
            gPool_over_layer += F.dropout(self.linears_prediction[layer](g_pool),
                                          0.5,
                                          training=self.training)

        return node_pool_over_layer, gPool_over_layer


if __name__ == '__main__':

    import gym
    from uniform_instance_gen import uni_instance_gen
    from torch_sparse import SparseTensor
    import torch
    import time
    import numpy as np
    from torch_geometric.data import Data
    from torch_geometric.data import Batch

    torch.manual_seed(0)
    np.random.seed(1)

    dev = 'cpu'
    nj = 3
    nm = 3
    low = 1
    high = 99
    delay = 1
    shaped_reward = 0

    env = gym.make('active_jssp_hom:active_jssp_hom-v0',
                   n_j=nj,
                   n_m=nm,
                   low=low,
                   high=high,
                   alpha=delay,
                   shaped_val=shaped_reward,
                   init_quality_flag=False,
                   et_normalize_coef=1)
    data = uni_instance_gen(n_j=nj, n_m=nm, low=low, high=high)

    adj, fea, candidate, mask = env.reset(data)
    adj = torch.from_numpy(adj).long()
    fea = torch.from_numpy(fea)

    # get batch graphs data list
    data = [Data(x=fea, edge_index=torch.nonzero(adj).t())]
    # generate batch graph
    batch_graph = Batch.from_data_list(data_list=data).to(dev)

    gin = Net(in_chnl=2, hid_chnl=64).to(dev)

    p = gin(batch_graph.x, batch_graph.edge_index, batch=batch_graph.batch)

    # grad = torch.autograd.grad(gPool.sum() + node_pool.sum(), [param for param in gin.parameters()])

    # def global_mean_pool(x, batch, size = None):
    #     from torch_scatter import scatter
    #     size = int(batch.max().item() + 1) if size is None else size
    #     return scatter(x, batch, dim=0, dim_size=size, reduce='mean')
    #
    # x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    # batch = torch.tensor([0, 1], dtype=torch.long)
    # pool = global_mean_pool(x, batch)
    # print(pool)

