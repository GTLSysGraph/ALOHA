import matplotlib.pyplot as plt

# 字体大小 图例 标签
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size' : 15,
       }

plt.xticks(fontsize=15, fontproperties = 'Times New Roman') #设置x轴刻度字体大小
plt.yticks(fontsize=15, fontproperties = 'Times New Roman') #设置y轴刻度字体大小


# graph4recon_cora
# x      = [0.0,     0.05,   0.1,     0.15,        0.2,        0.25]#点的横坐标
# ptb    = [0.8590, 0.8227, 0.7648,  0.7176,      0.6397,     0.5592]#线1的纵坐标
# refine = [0.8480, 0.8144, 0.7618,  0.7152,      0.6502,     0.5812]#线2的纵坐标
# plt.plot(x,ptb,    's--', color = '#FA8072', label="$\mathcal{A}_{ptb}$",     linewidth=3)#s-:方形
# plt.plot(x,refine, 'o--', color = '#3CB371', label="$\mathcal{A}_{refine}$",  linewidth=3)#o-:圆形
# plt.xlabel("Perturbation Ratio(%)", font)#横坐标名字
# plt.ylabel("Accuracy",              font)#纵坐标名字
# plt.legend(prop=font, loc = "best")#图例
# plt.savefig('graph4recon_cora.pdf', dpi=1280) # 保存图


# graph4recon_citeseer
# x      = [0.0,       0.05,   0.1,      0.15,         0.20,         0.25]#点的横坐标
# ptb    = [0.7540,   0.7520,  0.7320,   0.6880,       0.6670,       0.6590]#线1的纵坐标
# refine = [0.7560,   0.7540,  0.7340,   0.6940,       0.6710,       0.6630]#线2的纵坐标
# plt.plot(x,ptb,    's--', color = '#FA8072', label="$\mathcal{A}_{ptb}$",     linewidth=3)#s-:方形
# plt.plot(x,refine, 'o--', color = '#3CB371', label="$\mathcal{A}_{refine}$",  linewidth=3)#o-:圆形
# plt.xlabel("Perturbation Ratio(%)",font)#横坐标名字
# plt.ylabel("Accuracy",             font)#纵坐标名字
# plt.legend(prop=font, loc = "best")#图例
# plt.savefig('graph4recon_citeseer.pdf', dpi=1280) # 保存图





# gamma sensitivity
# x     = [0.0,       10,        20,       40,           60,          80]#点的横坐标
# gamma = [0.7720,    0.7807,    0.7857,   0.7840,       0.7840,      0.7820]#线1的纵坐标
# plt.plot(x,gamma,   's--',  color = '#9932CC',   label="$\lambda$",   linewidth=3)#s-:方形
# plt.xlabel("$\lambda$", font)#横坐标名字
# plt.ylabel("Accuracy",  font)#纵坐标名字
# plt.legend(prop=font, loc = "best")#图例
# plt.savefig('Pubmed_gamma_0.25.pdf', dpi=1280) # 保存图


# beta sensitivity
# x    = [0.0,     0.1,     0.5,      1,          10,         20]#点的横坐标
# beta = [0.67500, 0.7607, 0.7833,  0.7781,      0.7670,     0.7582]#线1的纵坐标
# plt.plot(x,beta,     'o--',   color = '#1E90FF',   label="$\\beta$",   linewidth=3)#s-:方形
# plt.xlabel("$\\beta$", font)#横坐标名字
# plt.ylabel("Accuracy", font)#纵坐标名字
# plt.legend(prop=font, loc = "best")#图例
# plt.savefig('Pubmed_beta_0.25.pdf', dpi=1280) # 保存图



# DICE Cora
# x            = [0.0,      0.1,     0.2,      0.3,          0.4,         0.5]#点的横坐标
# acc_GAE      = [0.8134,  0.7763,  0.7200,   0.7069,       0.6580,      0.6219]#线1的纵坐标
# acc_VGAE     = [0.8346,  0.7666,  0.7486,   0.7163,       0.6633,      0.6122]#线1的纵坐标
# acc_S2GAE    = [0.8269,  0.7719,  0.7204,   0.6830,       0.6260,      0.5600]#线1的纵坐标
# acc_MaskGAE  = [0.8449,  0.8223,  0.7717,   0.7577,       0.7238,      0.6826]#线1的纵坐标
# acc_GraphMAE = [0.8480,  0.8242,  0.7817,   0.7574,       0.7220,      0.6668]#线1的纵坐标
# acc_SPMGAE   = [0.8578,  0.8346,  0.8101,   0.7801,       0.7475,      0.7123]#线1的纵坐标
# plt.plot(x,acc_GAE,     'o--',   color = '#5F9EA0',  label="GAE",         linewidth=2)#s-:方形
# plt.plot(x,acc_VGAE,    'v--',   color = '#FFD700',  label="VGAE",        linewidth=2)#s-:方形
# plt.plot(x,acc_S2GAE,   'p--',   color = '#EE82EE',  label="S2GAE",       linewidth=2)#s-:方形
# plt.plot(x,acc_MaskGAE, 'h--',   color = '#FF6A6A',  label="MaskGAE",     linewidth=2)#s-:方形
# plt.plot(x,acc_GraphMAE,'d--',   color = '#87CEFA',  label="GraphMAE",    linewidth=2)#s-:方形
# plt.plot(x,acc_SPMGAE,  's--',   color = '#4EEE94',  label="SPMGAE",      linewidth=2)#s-:方形
# plt.xlabel("Perturbation Rate(%)", font)#横坐标名字
# plt.ylabel("Accuracy",             font)#纵坐标名字
# plt.legend(prop=font, loc = "best")#图例
# plt.savefig('./Cora_DICE.pdf', dpi=1280)


# random Cora
# x            = [0.0,      0.1,     0.2,      0.3,          0.4,         0.5       ]#点的横坐标
# acc_GAE      = [0.8118,   0.7931,  0.7591,   0.7175,       0.7131,      0.6626    ]#线1的纵坐标
# acc_VGAE     = [0.8329,   0.7852,  0.7602,   0.7430,       0.7201,      0.6817    ]#线1的纵坐标
# acc_S2GAE    = [0.8269,   0.7803,  0.7293,   0.6891,       0.6488,      0.6302    ]#线1的纵坐标
# acc_MaskGAE  = [0.8426,   0.8202,  0.8063,   0.7893,       0.7575,      0.7437    ]#线1的纵坐标
# acc_GraphMAE = [0.8480,   0.8286,  0.8060,   0.7828,       0.7779,      0.7660    ]#线1的纵坐标
# acc_SPMGAE   = [0.8578,   0.8400,  0.8193,   0.8090,       0.7919,      0.7800    ]#线1的纵坐标
# plt.plot(x,acc_GAE,     'o--',   color = '#5F9EA0',  label="GAE",         linewidth=2)#s-:方形
# plt.plot(x,acc_VGAE,    'v--',   color = '#FFD700',  label="VGAE",        linewidth=2)#s-:方形
# plt.plot(x,acc_S2GAE,   'p--',   color = '#EE82EE',  label="S2GAE",       linewidth=2)#s-:方形
# plt.plot(x,acc_MaskGAE, 'h--',   color = '#FF6A6A',  label="MaskGAE",     linewidth=2)#s-:方形
# plt.plot(x,acc_GraphMAE,'d--',   color = '#87CEFA',  label="GraphMAE",    linewidth=2)#s-:方形
# plt.plot(x,acc_SPMGAE,  's--',   color = '#4EEE94',  label="SPMGAE",      linewidth=2)#s-:方形
# plt.xlabel("Perturbation Rate(%)", font)#横坐标名字
# plt.ylabel("Accuracy",             font)#纵坐标名字
# plt.legend(prop=font, loc = "best")#图例
# plt.savefig('./Cora_random.pdf', dpi=1280)


# nettack Cora
# x            = [0.0,      1.0,     2.0,      3.0,          4.0,         5.0       ]#点的横坐标
# acc_GAE      = [0.8205,   0.7867,  0.7265,   0.6855,       0.6386,      0.5651    ]#线1的纵坐标
# acc_VGAE     = [0.8386,   0.7843,  0.7120,   0.6819,       0.6157,      0.5542    ]#线1的纵坐标
# acc_S2GAE    = [0.8152,   0.7710,  0.6827,   0.6425,       0.5742,      0.5180    ]#线1的纵坐标
# acc_MaskGAE  = [0.8205,   0.7831,  0.7410,   0.6952,       0.6410,      0.6265    ]#线1的纵坐标
# acc_GraphMAE = [0.8313,   0.7771,  0.7500,   0.7108,       0.6566,      0.6235    ]#线1的纵坐标
# acc_SPMGAE   = [0.8283,   0.7831,  0.7771,   0.7440,       0.7001,      0.6998    ]#线1的纵坐标
# plt.plot(x,acc_GAE,     'o--',   color = '#5F9EA0',  label="GAE",         linewidth=2)#s-:方形
# plt.plot(x,acc_VGAE,    'v--',   color = '#FFD700',  label="VGAE",        linewidth=2)#s-:方形
# plt.plot(x,acc_S2GAE,   'p--',   color = '#EE82EE',  label="S2GAE",       linewidth=2)#s-:方形
# plt.plot(x,acc_MaskGAE, 'h--',   color = '#FF6A6A',  label="MaskGAE",     linewidth=2)#s-:方形
# plt.plot(x,acc_GraphMAE,'d--',   color = '#87CEFA',  label="GraphMAE",    linewidth=2)#s-:方形
# plt.plot(x,acc_SPMGAE,  's--',   color = '#4EEE94',  label="SPMGAE",      linewidth=2)#s-:方形
# plt.xlabel("Number of Perturbations Per Node",  font)#横坐标名字
# plt.ylabel("Accuracy",                          font)#纵坐标名字
# plt.legend(prop=font, loc = "best")#图例
# plt.savefig('./Cora_nettack.pdf', dpi=1280)


# nettack Citeseer
x            = [0.0,      1.0,     2.0,      3.0,          4.0,         5.0       ]#点的横坐标
acc_GAE      = [0.8254,   0.7778,  0.6143,   0.5857,       0.5270,      0.4698    ]#线1的纵坐标
acc_VGAE     = [0.8302,   0.7825,  0.6317,   0.5778,       0.5667,      0.4397    ]#线1的纵坐标
acc_S2GAE    = [0.8201,   0.7671,  0.7460,   0.5873,       0.5820,      0.5608    ]#线1的纵坐标
acc_MaskGAE  = [0.8254,   0.8079,  0.7063,   0.6317,       0.6127,      0.4381    ]#线1的纵坐标
acc_GraphMAE = [0.8254,   0.8135,  0.7817,   0.6786,       0.6429,      0.6111    ]#线1的纵坐标
acc_SPMGAE   = [0.8254,   0.8214,  0.8214,   0.8170,       0.7897,      0.7817    ]#线1的纵坐标
plt.plot(x,acc_GAE,     'o--',   color = '#5F9EA0',  label="GAE",         linewidth=2)#s-:方形
plt.plot(x,acc_VGAE,    'v--',   color = '#FFD700',  label="VGAE",        linewidth=2)#s-:方形
plt.plot(x,acc_S2GAE,   'p--',   color = '#EE82EE',  label="S2GAE",       linewidth=2)#s-:方形
plt.plot(x,acc_MaskGAE, 'h--',   color = '#FF6A6A',  label="MaskGAE",     linewidth=2)#s-:方形
plt.plot(x,acc_GraphMAE,'d--',   color = '#87CEFA',  label="GraphMAE",    linewidth=2)#s-:方形
plt.plot(x,acc_SPMGAE,  's--',   color = '#4EEE94',  label="SPMGAE",      linewidth=2)#s-:方形
plt.xlabel("Number of Perturbations Per Node",  font)#横坐标名字
plt.ylabel("Accuracy",                          font)#纵坐标名字
plt.legend(prop=font, loc = "best")#图例
plt.savefig('./Citeseer_nettack.pdf', dpi=1280)