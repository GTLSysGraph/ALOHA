# <center>ALOHA : GAE & GCL Benchmark and Toolkit</center>
Hello, I'll talk more about that later.
## 🌱 Highlighted Features
* Attacked dataset
* Both dgl & pyg are applicable
* You can use `bash train_all_attack.sh MODEL DATASET` to get model performance on datasets attacked by different methods.

## 🌵 Dependencies
We recommend using ALOHA on Linux systems (e.g. Ubuntu and CentOS). Other systems (e.g., Windows and macOS) have not been tested.

### **Pytorch**
ALOHA is built based on PyTorch. You can install PyTorch following the instruction in PyTorch. For example:
```
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```
### **PyG**
PyG (PyTorch Geometric) is a library built upon  PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.

Select supported libraries based on your cuda version and python version,you can download from 
```
https://pytorch-geometric.com/whl/
```
Then,like
```
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.8-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric
```
### **DGL**
DGL is an easy-to-use, high performance and scalable Python package for deep learning on graphs.

you can install DGL via:
```
conda install -c dglteam dgl-cuda10.2==0.6.1
```

Of course, we also provide a complete library installation list, You can install it directly via:
```
pip install -r requirements.txt
```

## 🍂 References
Go!