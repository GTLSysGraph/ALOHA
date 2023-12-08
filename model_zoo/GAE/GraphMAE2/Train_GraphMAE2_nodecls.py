import   torch
from     easydict        import EasyDict
from     datasets_dgl.data_dgl import *
from     model_zoo.GAE.GraphMAE2.utils import *
from     model_zoo.GAE.GraphMAE2.build_easydict import *
from     model_zoo.GAE.GraphMAE2.evaluation_tranductive import *
from     model_zoo.GAE.GraphMAE2.evaluation_mini_batch  import *
from     model_zoo.GAE.GraphMAE2.models import build_model


def pretrain_mini_batch(model, graph, optimizer, batch_size, max_epoch, device, use_scheduler):
    logging.info("start training GraphMAE2 mini batch node classification..")

    model = model.to(device)

    # dataloader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.NodeDataLoader(
            graph,torch.arange(0, graph.num_nodes()), sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=1)

    logging.info(f"After creating dataloader: Memory: {show_occupied_memory():.2f} MB")
    if use_scheduler and max_epoch > 0:
        logging.info("Use scheduler")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    # train
    for epoch in range(max_epoch):
        epoch_iter = tqdm(dataloader)
        loss_list = []
        for input_nodes, output_nodes, _ in epoch_iter:
            model.train()
            subgraph = dgl.node_subgraph(graph, input_nodes).to(device)
            subgraph = subgraph.to(device)
            loss = model(subgraph, subgraph.ndata["feat"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            epoch_iter.set_description(f"train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
            loss_list.append(loss.item())
            
        if scheduler is not None:
            scheduler.step()

        # torch.save(model.state_dict(), os.path.join(model_dir, model_name))
        print(f"# Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}, Memory: {show_occupied_memory():.2f} MB")
    return model




def pretrain_tranductive(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss = model(graph, x, targets=target_nodes)

        loss_dict = {"loss": loss.item()}
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    return model



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

        model = build_model(param)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        if not load_model:
            if margs.mode == 'tranductive':
                model = pretrain_tranductive(model, graph, graph.ndata["feat"], optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            elif margs.mode == 'mini_batch':
                model = pretrain_mini_batch(model, graph, optimizer, batch_size, max_epoch, device, use_scheduler)


        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        
        model = model.to(device)
        model.eval()

        if margs.mode == 'tranductive':
            final_acc, estp_acc = node_classification_evaluation(model, graph, graph.ndata['feat'], num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        elif margs.mode == 'mini_batch':
            final_acc = evaluete_mini_batch(model, graph, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, device, batch_size=batch_size_f, shuffle=True)
            estp_acc  = final_acc


        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")