from torch_geometric.data import InMemoryDataset

class ConcatDataset(InMemoryDataset):
    r"""
    PyG Dataset class for merging multiple Dataset objects into one.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.__indices__ = None
        self.__data_list__ = []
        for dataset in datasets:
            self.__data_list__.extend(list(dataset))
        self.data, self.slices = self.collate(self.__data_list__)