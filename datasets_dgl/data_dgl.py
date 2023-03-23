from datasets_dgl.datasets_file.cora import CoraDataset
from datasets_dgl.datasets_file.citeseer import CiteseerDataset
from datasets_dgl.datasets_file.pubmed import PubmedDataset
from datasets_dgl.datasets_file.polblogs import PolblogsDataset

def load_data(args):
    if args.data in ['Cora']:
        return CoraDataset(args)
    elif args.data in ['Citeseer']:
        return CiteseerDataset(args)
    elif args.data in ['Pubmed']:
        return PubmedDataset(args)
    elif args.data in ['Polblogs']:
        return PolblogsDataset(args)
    else:
        raise Exception('Unknown dataset!')