import   torch
from     easydict        import EasyDict
from     datasets_dgl.data_dgl import *
from     model_zoo.GAE.GraphMAE2.utils import *
from     model_zoo.GAE.GraphMAE2.build_easydict import *


def Train_GraphMAE2_nodecls(margs):
    #########################
    device = f"cuda" if torch.cuda.is_available() else "cpu"
  
    dataset_name = margs.dataset
    if margs.mode in ['tranductive' , 'mini_batch']:
        if dataset_name.split('-',1)[0] == 'Attack':
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
            graph = dgl.add_self_loop(graph) # graphmae + self loop这结果也太好了，分析一下，有点意思
        else:
            if dataset_name in ['Cora','Pubmed','Citeseer','Cora_ml']:
                dataset  = load_data(dataset_name)
                graph = dataset[0]
            # elif dataset_name in ['ogbn-arxiv','ogbn-arxiv_undirected','reddit','ppi','yelp', 'amazon']:   
            #     multilabel_data = set(['ppi', 'yelp', 'amazon'])
            #     multilabel = dataset_name in multilabel_data
            #     if multilabel == True:
            #         raise Exception('not realise multilabel, loss should be BCE loss, will realise if use')
            #     dataset  = load_GraphSAINT_data(dataset_name, multilabel)
            #     graph = dataset.g
            #     graph = dgl.to_bidirected(graph)
            
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)

        num_classes = dataset.num_classes
        num_features = graph.ndata['feat'].shape[1]
    else:
        raise Exception('inductive not realise, wait...')

    MDT = build_easydict()
    param         = MDT['MODEL']['PARAM']
    if param.use_cfg:
        param = load_best_configs(param, dataset_name.split('-',1)[1].lower() if dataset_name.split('-',1)[0] == 'Attack' else dataset_name.lower())
    print(param)

    seeds         = param.seeds
    max_epoch     = param.max_epoch
    max_epoch_f   = param.max_epoch_f
    num_hidden    = param.num_hidden
    num_layers    = param.num_layers
    encoder_type  = param.encoder
    decoder_type  = param.decoder
    replace_rate  = param.replace_rate

    optim_type    = param.optimizer 
    loss_fn       = param.loss_fn

    lr             = param.lr
    weight_decay   = param.weight_decay
    lr_f           = param.lr_f
    weight_decay_f = param.weight_decay_f
    linear_prob    = param.linear_prob
    load_model     = param.load_model
    save_model     = param.save_model
    logs           = param.logging

    # mini batch use
    use_scheduler  = param.scheduler
    batch_size       = param.batch_size
    batch_size_f     = param.batch_size_f

    param.num_features = num_features

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        
        model = model.to(device)
        model.eval()

        final_acc, estp_acc = linear_probing_full_batch(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")