
from     easydict        import EasyDict
import   torch
from     model_zoo.GAE.GraphMAE.utils          import *
from     model_zoo.GAE.GraphMAE.build_easydict import *
from     model_zoo.GAE.GraphMAE.evaluation_tranductive     import * 
from     model_zoo.GAE.GraphMAE.evaluation_inductive       import * 
from     model_zoo.GAE.GraphMAE.evaluation_mini_batch       import * 
from     datasets_graphsaint.data_graphsaint import *
from     datasets_dgl.data_dgl import *

import   logging
from     tqdm import tqdm
from     model_zoo.GAE.GraphMAE.models import build_model
from     sampler.SAINTSampler import SAINTNodeSampler, SAINTEdgeSampler, SAINTRandomWalkSampler

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


# def pretrain_mini_batch(model, graph, optimizer, max_epoch, batch_size,device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    
#     # 用全部的还是只用train
#     # train_index = graph.ndata['train_mask'].nonzero().squeeze()
#     # torch.arange(0, graph.num_nodes())

#     # base sample 

#     # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
#     # dataloader = dgl.dataloading.NodeDataLoader(
#     #         graph,torch.arange(0, graph.num_nodes()), sampler,
#     #         batch_size=batch_size,
#     #         shuffle=True,
#     #         drop_last=False,
#     #         num_workers=1)
#     # 
#     # for input_nodes, output_nodes, _ in epoch_iter:
#     #     model.train()
#     #     subgraph = dgl.node_subgraph(graph, input_nodes).to(device)


#     # saint sample 用dgl提供的SAINTSampler

#     # num_iters = 1000
#     # sampler = SAINTSampler(
#     #             mode='node',                      # Can be 'node', 'edge' or 'walk'
#     #             budget=2000,
#     #             prefetch_ndata=['feat', 'label']  # optionally, specify data to prefetch
#     #         )
#     # dataloader = DataLoader(graph, torch.arange(num_iters), sampler, num_workers=1)
#     # for subgraph in epoch_iter:

#     # sampler文件里的 SAINTSampler
#     train_nid = graph.ndata['train_mask'].nonzero().squeeze()
#     subg_iter = SAINTNodeSampler(6000, 'Reddit', graph,
#                                     train_nid, 50)
    
#     logging.info("start mini batch training..")

#     total_epoch = tqdm(range(max_epoch))
#     for epoch in total_epoch:
#         loss_list = []
#         for _, subgraph in enumerate(subg_iter):
#             subgraph = subgraph.to(device)
#             model.train()
#             loss, loss_dict = model(subgraph, subgraph.ndata["feat"])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             loss_list.append(loss.item())

#             if scheduler is not None:
#                 scheduler.step()

#         train_loss = np.mean(loss_list)
#         total_epoch.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f}")
#         if logger is not None:
#             loss_dict["lr"] = get_current_lr(optimizer)
#             logger.note(loss_dict, step=epoch)
#     return model

def pretrain_mini_batch(model, graph, optimizer, batch_size, max_epoch, device, use_scheduler):
    logging.info("start training SPMGAE mini batch node classification..")

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
            loss, loss_dict = model(subgraph, subgraph.ndata["feat"])
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




def pretrain_inductive(model, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    train_loader, val_loader, test_loader, eval_train_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))

    if isinstance(train_loader, list) and len(train_loader) ==1:
        train_loader = [train_loader[0].to(device)]
        eval_train_loader = train_loader
    if isinstance(val_loader, list) and len(val_loader) == 1:
        val_loader = [val_loader[0].to(device)]
        test_loader = val_loader

    for epoch in epoch_iter:
        model.train()
        loss_list = []

        for subgraph in train_loader:
            subgraph = subgraph.to(device)
            loss, loss_dict = model(subgraph, subgraph.ndata["feat"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
        epoch_iter.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)
        
        if epoch == (max_epoch//2):
            evaluete(model, (eval_train_loader, val_loader, test_loader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)
    return model




def pretrain_tranductive(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)
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

    # return best_model
    return model



def Train_GraphMAE_nodecls(margs):
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
    
    MDT = build_easydict()
    param         = MDT['MODEL']['PARAM']
    if param.use_cfg:
        param = load_best_configs(param, dataset_name.split('-',1)[1].lower() if dataset_name.split('-',1)[0] == 'Attack' else dataset_name.lower() , "./model_zoo/GAE/GraphMAE/configs.yml")
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
    use_scheduler  = param.scheduler
    batch_size       = param.batch_size
    batch_size_f     = param.batch_size_f

    param.num_features = num_features

    acc_list = []
    estp_acc_list = []

    for i, seed in enumerate(seeds):
        print(f"####### Run {i+1} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name.lower()}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(param)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None


        if not load_model:
            if margs.mode == 'tranductive':
                model = pretrain_tranductive(model, graph, graph.ndata["feat"], optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            elif margs.mode == 'inductive':
                model = pretrain_inductive(model, (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            elif margs.mode == 'mini_batch':
                # model = pretrain_mini_batch(model, graph, optimizer, max_epoch, batch_size, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
                model = pretrain_mini_batch(model, graph, optimizer, batch_size, max_epoch, device, use_scheduler)
         


        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()


        if margs.mode == 'tranductive':
            final_acc, estp_acc = node_classification_evaluation(model, graph, graph.ndata['feat'], num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        elif margs.mode == 'inductive':
            final_acc, estp_acc = evaluete(model, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
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

