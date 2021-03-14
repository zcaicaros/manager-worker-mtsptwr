from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


class TSP(object):
    NAME = 'tsp'

    @staticmethod
    #    def get_costs(dataset, pi):
    #        # Check that tours are valid, i.e. contain 0 to n -1
    #        assert (
    #            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
    #            pi.data.sort(1)[0]
    #        ).all(), "Invalid tour"
    #
    #        # Gather dataset in order of tour
    #        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
    #        # timewindow for two clusters
    #        w1 = torch.zeros([d.size(0),d.size(1)//5])
    #        w2 = torch.ones([d.size(0),d.size(1)//5])
    #        w3 = 2*torch.ones([d.size(0),d.size(1)//5])
    #        w4 = 3*torch.ones([d.size(0),d.size(1)//5])
    #        w5 = 4*torch.ones([d.size(0),d.size(1)//5])
    #        w = torch.cat((w1,w2,w3,w4,w5),1)
    #        timeconstraints = 0.5*abs((d[:,:,2]-w)).sum(1)
    #        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
    #        return timeconstraints+(d[:, 1:, 0:2] - d[:, :-1, 0:2]).norm(p=2, dim=2).sum(1) + (d[:, 0, 0:2] - d[:, -1, 0:2]).norm(p=2, dim=1), None
    #
    ##        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None
    #        #need to be changed

    # rongkai's get_cost for rejection
    def get_costs(dataset, pi, c, t):
        # Check that tours are valid, i.e. contain 0 to n -1
        #        assert (
        #            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
        #            pi.data.sort(1)[0]
        #        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        # depot = torch.tensor([[[40.0,50.0,0,0]]])
        depot = torch.tensor([[[0.5, 0.5, 0, 0]]], device=dataset.device)
        depot = depot.repeat(dataset.size(0), 1, 1)
        dataset1 = torch.cat([depot, dataset], dim=1)
        d = dataset1.gather(1, pi.unsqueeze(-1).expand_as(dataset1))
        # timewindow for two clusters
        #        w1 = torch.zeros([d.size(0),d.size(1)//5])
        #        w2 = torch.ones([d.size(0),d.size(1)//5])
        #        w3 = 2*torch.ones([d.size(0),d.size(1)//5])
        #        w4 = 3*torch.ones([d.size(0),d.size(1)//5])
        #        w5 = 4*torch.ones([d.size(0),d.size(1)//5])
        #        w = torch.cat((w1,w2,w3,w4,w5),1).cuda()
        #        timeconstraints = 0.5*abs((d[:,:,2]-w)).sum(1)
        #         Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        rejectionrate = c.squeeze(1)
        t = t.squeeze(1)
        # print(torch.sum(rejectionrate))
        # print("rejectionrate:{rate}".format(rate=torch.mean(rejectionrate)))
        # print("length:{length}".format(length=torch.mean(t + (d[:, 0, 0:2] - d[:, -1, 0:2]).norm(p=2, dim=1))))
        return rejectionrate * 100 + t + (d[:, 0, 0:2] - d[:, -1, 0:2]).norm(p=2, dim=1), None, rejectionrate, t + (
                    d[:, 0, 0:2] - d[:, -1, 0:2]).norm(p=2, dim=1)

    #####

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


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

            #            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

            #            w1 = torch.zeros(num_samples,size//5,1)
            #            w2 = torch.ones(num_samples,size//5,1)
            #            w3 = 2*torch.ones(num_samples,size//5,1)
            #            w4 = 3*torch.ones(num_samples,size//5,1)
            #            w5 = 4*torch.ones(num_samples,size//5,1)
            #            w = torch.cat((w1,w2,w3,w4,w5), 1)
            #            w = torch.zeros(num_samples,size,size, dtype = torch.double)
            #            for i in range(num_samples):
            #                for j in range(size):
            #                    for k in range(size):
            #                        w[i,j,k]=torch.sqrt((x[i,j,0]-x[i,k,0])**2+(x[i,j,1]-x[i,k,1])**2)

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
