import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manager.policy import action_sample, get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch


def validate(instances, p_net, no_agent, validation_model, beta, device, reward_type='overall'):

    batch_size = instances.shape[0]
    adj = torch.ones([batch_size, instances.shape[1], instances.shape[1]])  # adjacent matrix

    # get batch graphs instances list
    instances_list = [Data(x=instances[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(batch_size)]
    # generate batch graph
    batch_graph = Batch.from_data_list(data_list=instances_list).to(device)

    # get pi
    pi = p_net(batch_graph, n_nodes=instances.shape[1], n_batch=batch_size)

    # sample action and calculate log probs
    action, log_prob = action_sample(pi)

    # get reward for each batch
    reward, rejs, lengths = get_reward(action, instances.to(device), no_agent, validation_model, beta, reward_type=reward_type)  # reward: tensor [batch, 1]
    # print('Validation result:', format(sum(reward)/batch_size, '.4f'))

    return sum(reward)/batch_size, rejs, lengths

