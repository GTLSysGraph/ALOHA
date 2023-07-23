import sys
import os.path as osp
import os
import networkx as nx
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join('/home/songsh/', 'GCL'))

from  datasets_pyg.data_pyg import get_dataset
from torch_geometric.loader import NeighborLoader





# G = nx.star_graph(20)
# pos = nx.spring_layout(G) #布局为中心放射状
# colors = range(20)
# nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors,
#         width=4, edge_cmap=plt.cm.Blues, with_labels=False)
# plt.savefig('xx.png', dpi=1280)
# plt.show()

# 加载Cora数据集
dataset_name = 'CiteSeer'
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



dataset.data.n_id = torch.arange(dataset.data.num_nodes)

# 全图
num_nodes = 20 # 需要可视化的节点数
graph = nx.Graph() # 创建一个图
edge_index = dataset.data.edge_index.T # 边信息

# 将Cora的边信息添加到nx图中
for i in range(num_nodes):
    graph.add_edge(edge_index[i][0].item(), edge_index[i][1].item())
    
# 计算每个节点的位置信息，采用kamada_kawai_layout布局方式
# pos=nx.circular_layout(graph)          # 生成圆形节点布局
# pos=nx.random_layout(graph)            # 生成随机节点布局
pos=nx.shell_layout(graph)             # 生成同心圆节点布局
# pos=nx.spring_layout(graph)            # 利用Fruchterman-Reingold force-directed算法生成节点布局
# pos=nx.spectral_layout(graph)          # 利用图拉普拉斯特征向量生成节点布局
# pos=nx.kamada_kawai_layout(graph)      # 使用Kamada-Kawai路径长度代价函数生成布局


# Cora有7个类别，对应7个颜色 参考rgb颜色表修改
# color = ['red',     'orange', 'blue',           'green',      'yellow',    'pink', 'darkviolet']
color =   ['#FA8072', 'orange', 'CornflowerBlue', '#3CB371',    '#FFEC8B',   'pink', 'Plum']

# 每个节点对应的颜色
node_color = [color[dataset.data.y[i]] for i in graph.nodes]
# 绘制图
nx.draw_networkx(graph, pos, font_family = 'Times New Roman', node_size=100, node_color=node_color,style='solid',alpha=1.0, edge_color='#DCDCDC',with_labels=False)

# 保存图
plt.savefig('a.pdf', dpi=1280)

#################################################################################################################################

# draw子图
# color =   ['#FA8072', 'orange', 'CornflowerBlue', '#3CB371',    '#FFEC8B',   'pink', 'Plum']
# num_neighbors = [100]
# input_nodes = torch.tensor([7])
# for s in NeighborLoader(dataset[0], num_neighbors=num_neighbors,input_nodes=input_nodes):

#     graph_sub = nx.Graph() # 创建一个图
#     edge_index = s.edge_index.T # 边信息

#     # 将Cora的边信息添加到nx图中
#     for i in range(s.num_edges):
#         graph_sub.add_edge(edge_index[i][0].item(), edge_index[i][1].item())

#     s_colors = [color[dataset.data.y[i]] for i in s.n_id]

#     nx.draw_networkx(graph_sub, nx.shell_layout(graph_sub), node_size=800, node_color=s_colors,style='solid',alpha=1.0,  edge_color='#DCDCDC',with_labels=False)
#     plt.savefig('b.pdf', dpi=1280)

