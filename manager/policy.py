from gin import GIN_embedding, MLP_embedding
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.distributions import Categorical
# import os, sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)


class Agentembedding(nn.Module):
    def __init__(self, node_feature_size, key_size, value_size):
        super(Agentembedding, self).__init__()
        self.key_size = key_size
        self.q_agent = nn.Linear(2 * node_feature_size, key_size)
        self.k_agent = nn.Linear(node_feature_size, key_size)
        self.v_agent = nn.Linear(node_feature_size, value_size)

    def forward(self, f_c, f):
        q = self.q_agent(f_c)
        k = self.k_agent(f)
        v = self.v_agent(f)
        u = torch.matmul(k, q.transpose(-1, -2)) / math.sqrt(self.key_size)
        u_ = F.softmax(u, dim=-2).transpose(-1, -2)
        agent_embedding = torch.matmul(u_, v)

        return agent_embedding


class AgentAndNode_embedding(torch.nn.Module):
    def __init__(self, vehicle_embd_type, in_chnl, hid_chnl, n_agent, key_size, value_size, dev, node_embedding_type='gin'):
        super(AgentAndNode_embedding, self).__init__()

        self.n_agent = n_agent
        self.vehicle_embd_type = vehicle_embd_type

        # gin
        if node_embedding_type == 'gin':
            self.embedding_net = GIN_embedding(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)
        else:
            # mlp
            self.embedding_net = MLP_embedding(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)

        if self.vehicle_embd_type == 'MH':
            self.agents = torch.nn.ModuleList()
            for i in range(n_agent):
                self.agents.append(Agentembedding(node_feature_size=hid_chnl, key_size=key_size, value_size=value_size).to(dev))
        elif self.vehicle_embd_type == 'SH':
            self.agent = Agentembedding(node_feature_size=hid_chnl, key_size=key_size, value_size=value_size).to(dev)
        else:
            print('Invalid, Should be `MH` or `SH`')

    def forward(self, batch_graphs, n_nodes, n_batch):

        # get node embedding using gin
        nodes_h, g_h = self.embedding_net(x=batch_graphs.x, edge_index=batch_graphs.edge_index, batch=batch_graphs.batch)
        nodes_h = nodes_h.reshape(n_batch, n_nodes, -1)
        g_h = g_h.reshape(n_batch, 1, -1)

        depot_cat_g = torch.cat((g_h, nodes_h[:, 0, :].unsqueeze(1)), dim=-1)
        # output nodes embedding should not include depot, refer to paper: https://www.sciencedirect.com/science/article/abs/pii/S0950705120304445
        nodes_h_no_depot = nodes_h[:, 1:, :]

        agents_embedding = []
        if self.vehicle_embd_type == 'MH':
            # multi-head agent embed
            for i in range(self.n_agent):
                agents_embedding.append(self.agents[i](depot_cat_g, nodes_h_no_depot))
        elif self.vehicle_embd_type == 'SH':
            # single-head agent embed
            for i in range(self.n_agent):
                agents_embedding.append(self.agent(depot_cat_g, nodes_h_no_depot))
        else:
            print('Invalid, Should be `MH` or `SH`')

        agent_embeddings = torch.cat(agents_embedding, dim=1)

        return agent_embeddings, nodes_h_no_depot


class Policy(nn.Module):
    def __init__(self, vehicle_embd_type, node_embedding_type, in_chnl, hid_chnl, n_agent, key_size_embd, key_size_policy, val_size, clipping, dev):
        super(Policy, self).__init__()
        self.c = clipping
        self.key_size_policy = key_size_policy
        self.key_policy = nn.Linear(hid_chnl, self.key_size_policy).to(dev)
        self.q_policy = nn.Linear(val_size, self.key_size_policy).to(dev)

        # embed network
        self.embed = AgentAndNode_embedding(vehicle_embd_type=vehicle_embd_type, in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent,
                                            key_size=key_size_embd, value_size=val_size, dev=dev, node_embedding_type=node_embedding_type)

    def forward(self, batch_graph, n_nodes, n_batch):

        agent_embeddings, nodes_h_no_depot = self.embed(batch_graph, n_nodes, n_batch)

        k_policy = self.key_policy(nodes_h_no_depot)
        q_policy = self.q_policy(agent_embeddings)
        u_policy = torch.matmul(q_policy, k_policy.transpose(-1, -2)) / math.sqrt(self.key_size_policy)
        imp = self.c * torch.tanh(u_policy)
        prob = F.softmax(imp, dim=-2)

        return prob


def action_sample(pi):
    dist = Categorical(pi.transpose(2, 1))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


def action_greedy(pi):
    action = pi.argmax(dim=1)
    return action


def test_learned_model(data, model, beta):
    with torch.no_grad():
        cost, _, rej, length, rej_count = model(data, beta=beta)
    return cost.item(), rej.item(), length.item(), rej_count.item()


def get_reward(action, data, n_agent, validation_model, beta, reward_type='minmax'):

    cost = [0 for _ in range(data.shape[0])]
    # log average rej.rate for batch
    subtour_rej = [0 for _ in range(data.shape[0])]
    # log average length for batch
    subtour_len = [0 for _ in range(data.shape[0])]
    # number of rejected nodes for batch
    total_rej_count = [0 for _ in range(data.shape[0])]
    # total length for all vehicle
    total_len = [0 for _ in range(data.shape[0])]

    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for n, m in zip(action.tolist()[i], data.tolist()[i][1:]):
            sub_tours[i][n].append(m)
        # no need to append depot as it will be considered while evaluating.
        # for tour in sub_tours[i]:
        #     tour.append(depot[i])

    for k in range(data.shape[0]):
        for a in range(n_agent):
            instance = sub_tours[k][a]
            if len(instance) == 0:  # if sub-tour is null (empty set), then return 0 cost
                sub_tour_cost = 0
                rej = 0
                length = 0
                rej_count = 0
            else:  # if sub-tour is not null, then evaluate it with pretrained model
                sub_tour_cost, rej, length, rej_count = test_learned_model(
                    torch.tensor(
                        instance,
                        device=data.device,
                        dtype=torch.float).unsqueeze(0),
                    validation_model,
                    beta=beta
                )
            if reward_type == 'minmax':
                if sub_tour_cost >= cost[k]:
                    cost[k] = sub_tour_cost
                    # log rej.rate
                    subtour_rej[k] = rej
                    # log length
                    subtour_len[k] = length
            elif reward_type == 'overall':
                total_rej_count[k] += rej_count
                total_len[k] += length
            else:
                raise RuntimeError('Not supported reward type, select form ["minmax", "overall"]')
    if reward_type == 'minmax':
        return cost, subtour_rej, subtour_len
    elif reward_type == 'overall':
        avg_len = [l/n_agent for l in total_len]
        overall_rej = [(c/data.shape[1]) for c in total_rej_count]
        cost = [beta * r + l for r, l in zip(overall_rej, avg_len)]
        return cost, overall_rej, avg_len


if __name__ == '__main__':
    from torch_geometric.data import Data
    from torch_geometric.data import Batch

    dev = 'cpu'
    torch.manual_seed(2)

    n_agent = 4
    n_nodes = 6
    n_batch = 3
    # get batch graphs data list
    fea = torch.randint(low=0, high=100, size=[n_batch, n_nodes, 2]).to(torch.float)  # [batch, nodes, fea]
    adj = torch.ones([fea.shape[0], fea.shape[1], fea.shape[1]])
    data_list = [Data(x=fea[i], edge_index=torch.nonzero(adj[i]).t()) for i in range(fea.shape[0])]
    # generate batch graph
    batch_graph = Batch.from_data_list(data_list=data_list).to(dev)

    # test policy
    policy = Policy(vehicle_embd_type='MH',
                    node_embedding_type='gin',
                    in_chnl=fea.shape[-1], hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)

    pi = policy(batch_graph, n_nodes, n_batch)

    grad = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()])

    action, log_prob = action_sample(pi)
