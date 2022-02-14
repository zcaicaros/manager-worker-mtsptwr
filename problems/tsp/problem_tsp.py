from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP


class TSP(object):
    NAME = 'tsp'

    @staticmethod
    # rongkai's get_cost for rejection
    def get_costs(dataset, pi, c, t, beta=100):
        depot = torch.tensor([[[0.5, 0.5, 0, 0]]], device=dataset.device)
        depot = depot.repeat(dataset.size(0), 1, 1)
        dataset1 = torch.cat([depot, dataset], dim=1)
        d = dataset1.gather(1, pi.unsqueeze(-1).expand_as(dataset1))
        rejectionrate = c.squeeze(1)
        t = t.squeeze(1)
        return rejectionrate * beta + t + (d[:, 0, 0:2] - d[:, -1, 0:2]).norm(p=2, dim=1), None, rejectionrate, t + (
                    d[:, 0, 0:2] - d[:, -1, 0:2]).norm(p=2, dim=1)

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=30, num_samples=10000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
        else:
            # Sample points randomly in [0, 1] square

            torch.manual_seed(12)
            x = torch.rand(num_samples, size, 2)
            w_begin = 3 * torch.zeros(num_samples, size, 1)
            w_end = w_begin + 3
            self.data = torch.cat([x, w_begin, w_end], 2)
            '''
            self.data = inputdata
            '''
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
