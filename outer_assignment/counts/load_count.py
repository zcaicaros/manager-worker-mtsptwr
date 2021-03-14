import numpy as np


size = 50
n_vehicles = 5
max_or_min = 'max'
print(np.load('./count_'+max_or_min+'_'+str(size)+'_'+str(n_vehicles)+'.npy'))