from     datasets_graphsaint.data_graphsaint import *
from     datasets_dgl.data_dgl import *

from     easydict        import EasyDict
from     .build_easydict import *
from     .model import GAT
from     .train_eval_tranductive import train_tranductive
from     .train_eval_inductive   import train_inductive
import   torch





def Train_GAT_nodecls(margs):
    #########################
    device = "cuda" if torch.cuda.is_available() else "cpu"
  
    dataset_name = margs.dataset
    if margs.mode in ['tranductive' , 'mini_batch']:
        if dataset_name.split('-')[0] == 'Attack':
            # dataset_name = dataset_name.split('-')[1]
            DATASET = EasyDict()
            DATASET.ATTACK = {
                "data":dataset_name,
                "attack":margs.attack.split('-')[0],
                "ptb_rate":margs.attack.split('-')[1]
            }
            # now just attack use
            dataset  = load_attack_data(DATASET['ATTACK'])
            graph = dataset.graph    
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
        else:
            if dataset_name in ['Cora','Pubmed','Citeseer']:
                dataset  = load_data(dataset_name)
                graph = dataset[0]
            elif dataset_name in ['ogbn-arxiv','ogbn-arxiv_undirected','reddit','ppi','yelp', 'amazon']:   
                multilabel_data = set(['ppi', 'yelp', 'amazon'])
                multilabel = dataset_name in multilabel_data
                if multilabel == True:
                    raise Exception('not realise multilabel, loss should be BCE loss, will realise if use')
                dataset  = load_GraphSAINT_data(dataset_name, multilabel)
                graph = dataset.g

            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
        
        num_classes = dataset.num_classes
        num_features = graph.ndata['feat'].shape[1]
    elif margs.mode in ['inductive']:
            (
                train_dataloader,
                valid_dataloader, 
                test_dataloader, 
                eval_train_dataloader, 
                num_features, 
                num_classes
            ) = load_inductive_dataset(dataset_name)
    else:
        raise Exception('Unknown mode!')

    ##########################
    
    MDT           = build_easydict()
    args         = MDT['MODEL']['PARAM']

    # create GAT model
    in_size = num_features
    out_size = num_classes
    model = GAT(args.num_layers, in_size, args.n_hidden, out_size, args.heads).to(device)

    # seeds
    seeds = args.seeds

    if margs.mode == 'tranductive':
        train_tranductive(seeds,graph, model, device)
    elif margs.mode == 'inductive':
        assert dataset_name == 'ppi'
        train_inductive(seeds,train_dataloader, valid_dataloader, test_dataloader,model, device)
        










