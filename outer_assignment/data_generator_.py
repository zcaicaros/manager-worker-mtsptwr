import torch


def data_gen(no_nodes, batch_size, flag):
    if flag == 'validation':
        location = torch.rand(size=[batch_size, no_nodes-1, 2])
        win_start = 3 * torch.rand(size=[batch_size, no_nodes-1, 1])
        win_end = 3 + win_start
        location_with_tw = torch.cat([location, win_start, win_end], dim=-1)
        depot = torch.tensor([0.5, 0.5, 0, 10], dtype=torch.float).repeat(batch_size, 1, 1)
        data = torch.cat([depot, location_with_tw], dim=1)
        torch.save(data, './validation_data_'+str(no_nodes)+'_'+str(batch_size))
    elif flag == 'testing':
        location = torch.rand(size=[batch_size, no_nodes - 1, 2])
        win_start = 3 * torch.rand(size=[batch_size, no_nodes - 1, 1])
        win_end = 3 + win_start
        location_with_tw = torch.cat([location, win_start, win_end], dim=-1)
        depot = torch.tensor([0.5, 0.5, 0, 10], dtype=torch.float).repeat(batch_size, 1, 1)
        data = torch.cat([depot, location_with_tw], dim=1)
        torch.save(data, './testing_data_' + str(no_nodes) + '_' + str(batch_size))
    elif flag == 'training':
        location = torch.rand(size=[batch_size, no_nodes - 1, 2])
        win_start = 3 * torch.rand(size=[batch_size, no_nodes - 1, 1])
        win_end = 3 + win_start
        location_with_tw = torch.cat([location, win_start, win_end], dim=-1)
        depot = torch.tensor([0.5, 0.5, 0, 10], dtype=torch.float).repeat(batch_size, 1, 1)
        data = torch.cat([depot, location_with_tw], dim=1)
        torch.save(data, './training_data_' + str(no_nodes) + '_' + str(batch_size))
    else:
        print('flag should be one of "training", "testing", or "validation".')


if __name__ == '__main__':
    n_nodes = 150
    b_size = 1000
    flag = 'validation'
    torch.manual_seed(3)

    data_gen(n_nodes, b_size, flag)