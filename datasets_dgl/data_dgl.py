from datasets_dgl.datasets_file.cora import CoraDataset
from datasets_dgl.datasets_file.citeseer import CiteseerDataset
from datasets_dgl.datasets_file.pubmed import PubmedDataset
from datasets_dgl.datasets_file.polblogs import PolblogsDataset

from dgl.data import CoraGraphDataset
from dgl.data import CiteseerGraphDataset
from dgl.data import PubmedGraphDataset

def load_data(args):
    if args.data in ['Attack-Cora']:
        return CoraDataset(args)
    elif args.data in ['Attack-Citeseer']:
        return CiteseerDataset(args)
    elif args.data in ['Attack-Pubmed']:
        return PubmedDataset(args)
    elif args.data in ['Attack-Polblogs']:
        return PolblogsDataset(args)
    elif args.data in ['Cora']:
        return CoraGraphDataset()
    elif args.data in ['Citeseer']:
        return CiteseerGraphDataset()
    elif args.data in ['Pubmed']:
        return PubmedGraphDataset()
    else:
        raise Exception('Unknown dataset!')