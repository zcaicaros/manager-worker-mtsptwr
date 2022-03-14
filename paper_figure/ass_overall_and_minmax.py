import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl

# mpl.style.use('classic')


n = 100
m = 10

min1 = np.load('../manager/counts/count_min_{}_{}_overall.npy'.format(n, m))
mean1 = np.load('../manager/counts/count_mean_{}_{}_overall.npy'.format(n, m))
max1 = np.load('../manager/counts/count_max_{}_{}_overall.npy'.format(n, m))
min2 = np.load('../manager/counts/count_min_{}_{}_minmax.npy'.format(n, m))
mean2 = np.load('../manager/counts/count_mean_{}_{}_minmax.npy'.format(n, m))
max2 = np.load('../manager/counts/count_max_{}_{}_minmax.npy'.format(n, m))

fig, ax = plt.subplots(figsize=(16, 11.6))
ax.set_xlim(-3, 103)
x = np.arange(1, 101)
l4 = plt.plot(x, max2, '-', label='minmax: max assignment', linewidth=3, color='coral', ms=9)
l3 = plt.plot(x, min2, '--', label='minmax: min assignment', linewidth=3, color='coral', ms=9)
# l5 = plt.plot(x, mean2, '--', label='', linewidth=1, color='red', ms=9)
l2 = plt.plot(x, max1, '-', label='overall: max assignment', linewidth=3, color='royalblue', ms=9)
l1 = plt.plot(x, min1, '--', label='overall: min assignment', linewidth=3, color='royalblue', ms=9)
# l6 = plt.plot(x, mean1, '-', label='', linewidth=1, color='blue', ms=9)

plt.fill_between(x, max1, min1, color='cornflowerblue', alpha=0.5, linewidth=0)
plt.fill_between(x, max2, min2, color='darksalmon', alpha=0.5, linewidth=0)

plt.tick_params(labelsize=30)  #### fontsize for ticks on y and x-axis


######## change 'size' : 30
plt.xlabel('Instance ID', {'family': 'Times New Roman', 'weight': 'bold', 'size': 40})
plt.ylabel('Number of Assigned Customers', {'family': 'Times New Roman', 'weight': 'bold', 'size': 40})
plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 40}, loc="upper left", frameon=False)
plt.grid(ls='--')
plt.tight_layout()
plt.savefig('ass_overall_and_minmax_{}_{}.pdf'.format(n, m), dpi=100)
plt.show()
