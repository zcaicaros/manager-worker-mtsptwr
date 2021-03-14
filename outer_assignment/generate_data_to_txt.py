import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch


sizes = [50, 100, 150, 200, 300, 400, 500]
# sizes = [100]
batch_size = 100
torch.manual_seed(3)


for size in sizes:
    location = torch.rand(size=[batch_size, size-1, 2])
    win_start = 3 * torch.rand(size=[batch_size, size-1, 1])
    win_end = 3 + win_start
    location_with_tw = torch.cat([location, win_start, win_end], dim=-1)
    depot = torch.tensor([0.5, 0.5, 0, 10], dtype=torch.float).repeat(batch_size, 1, 1)
    dataset = torch.cat([depot, location_with_tw], dim=1)
    # dataset = location_with_tw  # donot include depot as required by Rongkai Matlab code
    torch.save(dataset, '../testing-instances/'+str(size) + '/testing_data_' + str(size) + '_' + str(batch_size))
    for idx in range(dataset.shape[0]):
        np.savetxt('../testing-instances/'+str(size)+'/instance_'+str(idx+1)+'.txt', dataset[idx].numpy(), delimiter=',', fmt='%1.20f')
    print('Generating data for size:', size)