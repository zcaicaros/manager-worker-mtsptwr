import matplotlib.pyplot as plt


m_5_tourlength = [4.76, 4.92, 5.13, 4.88, 4.93]
m_5_rej = [1, 0.5, 0.3, 0.1, 0]
m_10_tourlength = [4.23, 4.18, 4.29, 4.25, 4.22]
m_10_rej = [0.3, 0.2, 0.1, 0.1, 0]
beta = [10, 30, 50, 70, 100]

fig, ax1 = plt.subplots(figsize=(16, 11.6))
# tourlength
line1, = ax1.plot(beta, m_5_tourlength, color='darksalmon', linestyle='-', label='tour length', linewidth=3)
# p1 = ax1.scatter(beta, m_5_tourlength,color = 'red',marker = 'v',s = 30,label = 'tour length')

# rej
ax2 = ax1.twinx()
line2, = ax2.plot(beta, m_5_rej, color='royalblue', linestyle='--', label='rejectin rate', linewidth=3)
# p2 = ax2.scatter(beta, m_5_rej, color = 'blue',marker = 'o',s = 30,label = 'rejection rate')

# beta
ax1.set_xlim([10, 100])
ax1.set_ylim([4, 5.5])
ax2.set_ylim([0, 1])

ax1.set_xlabel(r'$\beta$', {'family': 'Times New Roman', 'weight': 'bold', 'size': 40})
ax1.set_ylabel("Tour Length", {'family': 'Times New Roman', 'weight': 'bold', 'size': 40})
ax2.set_ylabel("Rejection Rate (%)", {'family': 'Times New Roman', 'weight': 'bold', 'size': 40})
# ax1.set_xlabel("Instance ID", fontsize=12)
# ax1.set_title(r"The influence on tour length and rejection rate for different $\beta$",fontsize = 14)

# 双Y轴标签颜色设置
# ax1.yaxis.label.set_color(line1.get_color())
# ax2.yaxis.label.set_color(line2.get_color())

# 双Y轴刻度颜色设置
ax1.tick_params(axis='y', colors=line1.get_color(), labelsize=30)
ax1.tick_params(axis='x', labelsize=30)
ax2.tick_params(axis='y', colors=line2.get_color(), labelsize=30)

# 图例设置
# plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 40}, loc="upper left", frameon=False)
plt.legend(handles=[line1, line2], prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 40}, frameon=True)
ax1.grid(axis='both')
plt.tight_layout()
plt.savefig('beta_study.pdf', dpi=100)
plt.show()
