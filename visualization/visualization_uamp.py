import sys
import os.path as osp
import os

sys.path.append(os.path.join('/home/songsh/', 'GCL'))

from  datasets_pyg.data_pyg import get_dataset
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size' : 20,
       }


dataset_name = 'Cora'
attack       = 'Meta_Self-0.2'

path = osp.expanduser('/home/songsh/GCL/datasets_pyg')
path = osp.expanduser('/home/songsh/GCL/datasets_pyg')
if dataset_name.split('-')[0] == 'Attack':
    attackmethod = attack.split('-')[0]
    attackptb    = attack.split('-')[1]
    path = osp.expanduser('/home/songsh/GCL/datasets_pyg/Attack_data')
    dataset = get_dataset(path, dataset_name, attackmethod, attackptb)
else:
    path = osp.join(path, dataset_name)
    dataset = get_dataset(path, dataset_name)

data = dataset[0]

##### 原始数据集可视化
embd_x = umap.UMAP().fit_transform(data.x.numpy())
palette = {}

for n, y in enumerate(set(data.y.numpy())):
    palette[y] = f'C{n}'
    
plt.figure(figsize=(10, 10))
# plt.scatter(x=embd_x.T[0], y=embd_x.T[1],c=data.y.cpu().numpy(),marker='o')

sns.set_theme(style="white",font='Times New Roman',font_scale=2.5) # font_scale可以调整横纵坐标字体大小
sns.scatterplot(x=embd_x.T[0], y=embd_x.T[1], hue=data.y.cpu().numpy(),palette=palette)

# plt.ylabel('y', font={'family':'Times New Roman', 'size':16})
# plt.xlabel('x', font={'family':'Times New Roman', 'size':16})

plt.legend(loc = 3, prop=font)
plt.savefig("Cora2.pdf", dpi=1280)
