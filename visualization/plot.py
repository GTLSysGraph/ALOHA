import matplotlib.pyplot as plt

#折线图
# x = [0.0,     0.05,   0.1,     0.15,        0.2,        0.25]#点的横坐标
# k1 = [0.8590, 0.8227, 0.7648,  0.7176,      0.6397,     0.5592]#线1的纵坐标
# k2 = [0.8490, 0.8144, 0.7618,  0.7152,      0.6502,     0.5812]#线2的纵坐标
# plt.plot(x,k1,'s-',color = '#FF6A6A',label="ptb")#s-:方形
# plt.plot(x,k2,'o-',color = '#32CD32',label="refine")#o-:圆形
# plt.xlabel("Perturbation Ratio")#横坐标名字
# plt.ylabel("accuracy")#纵坐标名字
# plt.legend(loc = "best")#图例

# # 保存图
# plt.savefig('graph4recon_cora.pdf', dpi=1280)

x =  [0.0,     10,     20,      40,          60,         80]#点的横坐标
k1 = [0.7720, 0.7807, 0.7857,  0.7840,      0.7840,     0.7820]#线1的纵坐标
plt.plot(x,k1,'s-',color = '#9932CC',label="$\lambda$")#s-:方形
plt.xlabel("$\lambda$")#横坐标名字
plt.ylabel("accuracy")#纵坐标名字
plt.legend(loc = "best")#图例

# 保存图
plt.savefig('lamda_pubmed_0.25.pdf', dpi=1280)


# x =  [0.0,     0.1,     0.5,      1,          10,         20]#点的横坐标
# k1 = [0.67500, 0.7607, 0.7833,  0.7781,      0.7670,     0.75820]#线1的纵坐标
# plt.plot(x,k1,'s-',color = '#87CEFA',label="$gamma$")#s-:方形
# plt.xlabel("$gamma$")#横坐标名字
# plt.ylabel("accuracy")#纵坐标名字
# plt.legend(loc = "best")#图例

# # 保存图
# plt.savefig('gamma_pubmed_0.25.pdf', dpi=1280)