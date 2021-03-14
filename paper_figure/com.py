# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
#mpl.style.use('classic')
x1=[50,100,150,200,300,400,500]
y1=[3.56,3.82,4.31,5.10,6.03,12.12,15.13]
x2=[50,100,150,200,300,400,500]
y2=[3.81,4.65,4.97,6.04,14.31,20.79,27.61]
x3=[50,100,150,200,300,400,500]
y3=[3.250,4.070,4.980,6.860,15.340,24.112,30.55]
x4=[50,100,150,200,300,400,500]
y4=[4.070,6.860,15.340,24.112,35.973,44.655,49.637]
plt.figure(figsize=(16, 11.6))
l1=plt.plot(x1,y1,'^-',label='WA supervised by MA, m=10',linewidth=3,color='coral',ms=9)
l2=plt.plot(x2,y2,'o-',label='WA supervised by MA, m=5',linewidth=3,color='royalblue',ms=9)
l3=plt.plot(x3,y3,'^--',label='WA unsupervised, m=10',linewidth=3,color='coral',ms=9)
l4=plt.plot(x4,y4,'o--',label='WA unsupervised, m=5',linewidth=3,color='royalblue',ms=9)



#plt.plot(x1,y1,'^','darksalmon',x2,y2,'o',color='cornflowerblue',ms=9)
#plt.plot(x1,y1,'go',x3,y3,'gx')

#### fontsize for ticks on y and x-axis
plt.tick_params(labelsize=30)
plt.xticks([50,100,150,200,300,400,500],[r'$50$', r'$100$', r'$150$', r'$200$', r'$300$',r'$400$',r'$500$'])
#plt.title('The  Compensation Study', {'family' : 'Times New Roman','weight':'bold', 'size'   : 20})

######## change 'size' : 30
plt.xlabel('Different sizes of mTSPTWR', {'family' : 'Times New Roman', 'weight':'bold','size': 40})
plt.ylabel('Hybrid Cost', {'family' : 'Times New Roman','weight':'bold', 'size': 40})
plt.legend(prop={'family' : 'Times New Roman','weight':'bold', 'size': 40}, loc="upper left", frameon=False)
plt.grid(ls='--')
plt.tight_layout()
plt.savefig('com.pdf', dpi=100)
# plt.show()