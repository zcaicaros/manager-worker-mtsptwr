import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl

# mpl.style.use('classic')

x1 = np.arange(1, 101)
min_50_5 = np.load('../manager/counts/count_min_50_5.npy')
max_50_5 = np.load('../manager/counts/count_max_50_5.npy')
min_150_5 = np.load('../manager/counts/count_min_150_5.npy')
max_150_5 = np.load('../manager/counts/count_max_150_5.npy')
'''
min_50_10=np.load('./count_min_50_10.npy')
max_50_10=np.load('./count_max_50_10.npy')
min_150_10=np.load('./count_min_150_10.npy')
max_150_10=np.load('./count_max_150_10.npy')
'''
fig, ax = plt.subplots(figsize=(16, 11.6))
ax.set_xlim(-3, 103)
l4 = plt.plot(x1, max_150_5, '-', label='n=150, m=5:max', linewidth=3, color='coral', ms=9)
l3 = plt.plot(x1, min_150_5, '--', label='n=150, m=5:min', linewidth=3, color='coral', ms=9)
l2 = plt.plot(x1, max_50_5, '-', label='n=50, m=5:max', linewidth=3, color='royalblue', ms=9)
l1 = plt.plot(x1, min_50_5, '--', label='n=50, m=5:min', linewidth=3, color='royalblue', ms=9)

plt.fill_between(x1, max_50_5, min_50_5, color='cornflowerblue', alpha=0.5, linewidth=0)
plt.fill_between(x1, max_150_5, min_150_5, color='darksalmon', alpha=0.5, linewidth=0)

plt.tick_params(labelsize=30)  #### fontsize for ticks on y and x-axis
# l5=plt.plot(x1,min_50_10,'--',label='',linewidth=1,color='coral',ms=9)
# l6=plt.plot(x1,max_50_10,'-',label='',linewidth=1,color='coral',ms=9)
# l7=plt.plot(x1,min_150_10,'--',label='',linewidth=1,color='tomato',ms=9)
# l8=plt.plot(x1,max_150_10,'-',label='',linewidth=1,color='tomato',ms=9)
# plt.plot(x1,y1,'^','darksalmon',x2,y2,'o',color='cornflowerblue',ms=9)
# plt.plot(x1,y1,'go',x3,y3,'gx')
# plt.xticks([50,100,150,200,300,400,500],[r'$50$', r'$100$', r'$150$', r'$200$', r'$300$',r'$400$',r'$500$'])
# plt.title('The  Compensation Study', {'family' : 'Times New Roman','weight':'bold', 'size'   : 20})


######## change 'size' : 30
plt.xlabel('Instance ID', {'family': 'Times New Roman', 'weight': 'bold', 'size': 40})
plt.ylabel('Number of Assigned Customers', {'family': 'Times New Roman', 'weight': 'bold', 'size': 40})
plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 40}, loc="upper left", frameon=False)
plt.grid(ls='--')
plt.tight_layout()
plt.savefig('ass.pdf', dpi=100)
# plt.show()
