import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
#mpl.style.use('classic')

labels = ['50', '150','300','500']
SH_5 = [56.73, 71.83,79.19,86.95]
MH_5 = [3.75, 4.97,14.31,27.61]
SH_10 = [69.08,85.37,87.52,92.11]
MH_10 = [3.56,4.31,5.85,15.13]
x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(16, 11.6))
plt.tick_params(labelsize=25)
#plt.grid(axis="y",ls='--')
rects1 = ax.bar(x - width/2, SH_5, width,color='cornflowerblue', label='SH, m=5')
rects2 = ax.bar(x - width/2, MH_5, width, color='royalblue',label='MH, m=5')
rects3 = ax.bar(x + width/2, SH_10, width, color='darksalmon',label='SH, m=10')
rects4 = ax.bar(x + width/2, MH_10, width, color='coral',label='MH, m=10')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim(0, 119)

######## change fontsize =30
ax.set_ylabel('Hybrid Cost',fontsize=40,fontweight='bold',fontname='Times New Roman')
ax.set_xlabel('Different sizes of mTSPTWR',fontsize=40,fontweight='bold',fontname='Times New Roman')
#ax.set_title("Single-head vs. Multi-head on different mTSPTWR",loc="center",fontsize=20,fontweight='bold',fontname='Times New Roman')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.tick_params(labelsize=30)

####### change 'size'=30
ax.legend(frameon=False,loc='upper left',prop={'family' : 'Times New Roman','weight':'bold', 'size': 40})


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=38,fontweight='bold',fontname='Times New Roman') #### fontsize, weight for the number above the bars


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.savefig('ab.pdf', dpi=100)
# plt.show()