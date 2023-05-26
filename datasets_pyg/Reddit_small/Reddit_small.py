import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
import json
import scipy
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.sparse
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data

class Reddit_small(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):  

        # 数据的下载和处理过程在父类中调用实现
        super(Reddit_small, self).__init__(root, transform, pre_transform)

        # 加载数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')
    

    # 将函数修饰为类属性
    @property
    def raw_file_names(self):
        return ['adj_full.npz','adj_train.npz','class_map.json','feats.npy','role.json']

    @property
    def processed_file_names(self):
        return ['data_reddit_small.pt']


    def download(self):
        # download to self.raw_dir
        pass

    def process(self):
        prefix = self.raw_dir

        adj_full = scipy.sparse.load_npz('{}/adj_full.npz'.format(prefix)).astype(np.bool)
        adj_full = self.sparse_mx_to_torch_sparse_tensor(adj_full)
        num_nodes=adj_full.size(0)
        edge_index, _ = add_self_loops(adj_full.coalesce().indices(), num_nodes=adj_full.size(0))

        adj_train = scipy.sparse.load_npz('{}/adj_train.npz'.format(prefix)).astype(np.bool)
        train_nid = np.array(list(set(adj_train.nonzero()[0])))

        role = json.load(open('{}/role.json'.format(prefix)))
        mask = np.zeros((num_nodes,), dtype=bool)
        train_mask = mask.copy()
        train_mask[role['tr']] = True
        val_mask = mask.copy()
        val_mask[role['va']] = True
        test_mask = mask.copy()
        test_mask[role['te']] = True
        feats = np.load('{}/feats.npy'.format(prefix))
        scaler = StandardScaler()
        scaler.fit(feats[train_nid])
        feats = scaler.transform(feats)
        feats = torch.tensor(feats, dtype=torch.float)

        class_map = json.load(open('{}/class_map.json'.format(prefix)))
        class_map = {int(k): v for k, v in class_map.items()}

        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_nodes,))
        for k, v in class_map.items():
            class_arr[k] = v

    
        data = Data(x=feats, edge_index=edge_index, y=torch.tensor(class_arr, dtype=torch.long))
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
                
        data = data if self.pre_transform is None else self.pre_transform(data)
        # 这里的save方式以及路径需要对应构造函数中的load操作
        torch.save(self.collate([data]), self.processed_paths[0])


    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
        sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
        sparseconcat=torch.cat((sparserow, sparsecol),1)
        sparsedata=torch.FloatTensor(sparse_mx.data)
        return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))