import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from outer_assignment.validation_ import validate
import time
from outer_assignment.policy_ import Policy, action_sample, get_reward
from utils import load_model


def train(hidden_dim,
          vehicle_embd_type,
          node_mebd_type,
          no_batch,
          no_nodes,
          policy_net,
          l_r,
          no_agent,
          iterations,
          validation_model,
          beta,
          device):

    # prepare validation data
    validation_data = torch.load('../testing-instances/'+str(no_nodes)+'/testing_data_'+str(no_nodes)+'_'+str(100))
    # a large start point for validation
    best_so_far = 1000000
    validation_results = []

    # optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)

    itr_log = []
    vali_log = []
    for itr in range(iterations):
        t1 = time.time()
        # prepare training data
        location = torch.rand(size=[batch_size, no_nodes - 1, 2])  # nodes - 1: locations without depot
        win_start = 3 * torch.rand(size=[batch_size, no_nodes - 1, 1])  # nodes - 1: start time without depot
        win_end = 3 + win_start
        location_with_tw = torch.cat([location, win_start, win_end], dim=-1)  # locations+tw without depot
        depot = torch.tensor([0.5, 0.5, 0, 10], dtype=torch.float).repeat(batch_size, 1, 1)  # constant depot
        data = torch.cat([depot, location_with_tw], dim=1)  # final instances with depot
        adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]])  # adjacent matrix fully connected
        data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)

        # get pi
        pi = policy_net(batch_graph, n_nodes=data.shape[1], n_batch=no_batch)  # nodes assignment to agents without depot: e.g. tsp-20 pi.shape=[batch_size, 19]
        # sample action and calculate log probabilities
        action, log_prob = action_sample(pi)

        # get reward for each batch
        reward, rejs, lengths = get_reward(action, data.to(device), no_agent, validation_model, beta)  # reward: tensor [batch, 1]
        # compute loss
        loss = torch.mul(torch.tensor(reward, device=device) - 2, log_prob.sum(dim=1)).sum()

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        # log each itr
        print('Iteration:', itr,
              'Total loss:', format(sum(reward)/no_batch, '.3f'),
              'Mean rej.rate:', format(sum(rejs)/len(rejs), '.4f'),
              'Mean length:', format(sum(lengths)/len(lengths), '.4f'),
              'Time(s):', format(t2-t1, '.3f'))
        itr_log.append([format(sum(reward)/no_batch, '.6f'), format(sum(rejs)/len(rejs), '.6f'), format(sum(lengths)/len(lengths), '.6f')])

        # validate and save best nets
        if (itr+1) % 100 == 0:
            validation_loss, vali_rejs, vali_lengths = validate(validation_data, policy_net, no_agent, validation_model, device)
            print('Validation mean rej.rate:', format(sum(vali_rejs)/len(vali_rejs), '.4f'),
                  'Validation mean length:', format(sum(vali_lengths)/len(vali_lengths), '.4f'))
            vali_log.append([format(sum(vali_rejs)/len(vali_rejs), '.6f'), format(sum(vali_lengths)/len(vali_lengths), '.6f')])
            if validation_loss < best_so_far:
                best_so_far = validation_loss
                torch.save(policy_net.state_dict(), '../pretrained_assgnmt_beta'+str(beta)+'/{}.pth'.format(
                    str(no_nodes) + '_' + str(no_agent) + '_' + vehicle_embd_type + '_' + node_mebd_type + '_' + str(hidden_dim)))
                print('Found better policy, and the validation loss is:', format(validation_loss, '.3f'))
                validation_results.append(validation_loss)
            file_writing_obj1 = open(
                './training_logs/' + 'itr_log_' + str(no_nodes) + '-' + str(no_agent) + '_' + vehicle_embd_type + '_' + node_mebd_type + '.txt', 'w')
            file_writing_obj1.write(str(itr_log))
            file_writing_obj2 = open(
                './training_logs/' + 'vali_log_' + str(no_nodes) + '-' + str(no_agent) + '_' + vehicle_embd_type + '_' + node_mebd_type + '.txt', 'w')
            file_writing_obj2.write(str(vali_log))

            print()


if __name__ == '__main__':
    import cProfile

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(2)

    n_agent = 5
    n_nodes = 50
    batch_size = 128
    lr = 1e-4
    iteration = 10000
    sh_or_mh = 'MH'  # 'MH'-MultiHead, 'SH'-SingleHead
    node_embedding_type = 'mlp'  # 'mlp', 'gin'
    GIN_dim = 32
    beta = 100

    print('Using', dev, 'to train.\n', 'Size:', str(n_nodes)+'-'+str(n_agent), 'Training...')

    policy = Policy(vehicle_embd_type=sh_or_mh, node_embedding_type=node_embedding_type,
                    in_chnl=4, hid_chnl=GIN_dim, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    policy.train()

    # load routing agent
    validation_net = load_model('../pretrained_inner_route_planner_beta'+str(beta)+'/'+str(int(n_nodes / n_agent))+'.pt', dev)
    validation_net.to(dev)
    validation_net.decode_type = 'greedy'
    # set routing agent to eval mode while training assignment agent
    validation_net.eval()

    cProfile.run('train('
                 'GIN_dim,'
                 'sh_or_mh,'
                 'node_embedding_type,'
                 'batch_size,'
                 'n_nodes,'
                 'policy,'
                 'lr,'
                 'n_agent,'
                 'iteration,'
                 'validation_net,'
                 'beta,'
                 'dev)',
                 filename='restats')
