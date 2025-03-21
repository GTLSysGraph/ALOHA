
from     easydict        import EasyDict
import   torch
from     model_zoo.GAE.SPMGAE.utils          import *
from     model_zoo.GAE.SPMGAE.build_easydict import *
from     model_zoo.GAE.SPMGAE.evaluation_tranductive     import * 
from     model_zoo.GAE.SPMGAE.evaluation_inductive       import * 
from     model_zoo.GAE.SPMGAE.evaluation_mini_batch       import * 
from     datasets_graphsaint.data_graphsaint import *
from     datasets_dgl.data_dgl import *

import   logging
from     tqdm import tqdm
from     model_zoo.GAE.SPMGAE.models import build_model
from     sampler.SAINTSampler import SAINTNodeSampler, SAINTEdgeSampler, SAINTRandomWalkSampler
import torch.nn.functional as F
from .utils import compute_edge_sim

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)





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




def pretrain_tranductive(param, model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training SPMGAE node classification..")
    graph = graph.to(device)
    x = feat.to(device)

    ###########################################################
    n_node = graph.num_nodes()

    edges_sim       = compute_edge_sim(graph.edges(), x, sim_mode=param.sim_mode)
    ref_edges_index = torch.where(edges_sim >= param.keep_threshold)
    del_edges_index = torch.where(edges_sim <  param.dele_threshold)
    edges           = torch.stack((graph.edges()[0],graph.edges()[1]),dim=0)

    ref_edges       = edges[:, ref_edges_index[0]]
    del_edges       = edges[:, del_edges_index[0]]
    graph_refine    = dgl.graph((ref_edges[0],ref_edges[1]), num_nodes=n_node).to(device)
    graph_refine = graph_refine.remove_self_loop()
    graph_refine = graph_refine.add_self_loop()

    print('num  edges : {}'.format(graph.num_edges()))
    print('keep edges : {}'.format(len(ref_edges_index[0])))
    print('del  edges : {}'.format(len(del_edges_index[0])))

    ############################################################

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()

        # loss, loss_dict = model(graph, x, epoch)
        loss, loss_dict = model(graph_refine, del_edges, x, epoch)

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



def Train_SPMGAE_nodecls(margs):
    #########################
    device = f"cuda" if torch.cuda.is_available() else "cpu"
  
    dataset_name = margs.dataset
    if margs.mode in ['tranductive']:
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
            graph = dgl.add_self_loop(graph) # graphmae + self loop这结果也太好了，分析一下，有点意思
        else:
            if dataset_name in ['Cora','Pubmed','Citeseer','CoauthorCS']:
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
    
    MDT = build_easydict()
    param         = MDT['MODEL']['PARAM']
    if param.use_cfg:
        param = load_best_configs(param, dataset_name.split('-')[1].lower() if dataset_name.split('-')[0] == 'Attack' else dataset_name.lower() , "./model_zoo/GAE/SPMGAE/configs.yml")


    # ##################################################################################################################
    # # grid search
    # grid_num = 0 
    # # for gamma in [0, 1, 10]:
    # #     for beta in [0.0, 0.001, 0.1]:
    # for decay in [0.2, 0.8]:
    # #           for add_rate in [0.2, 0.4, 0.6, 0.8]:
    #     for threshold in [[0.04, 0.02], [0.04, 0.04],[0.06,0.02], [0.06,0.04],[0.08, 0.02],[0.08, 0.04],[0.1, 0.02],[0.1, 0.04]]:
    #         for type_graph4recon in ['refine', 'ptb']:
    #             for num_hidden in [256, 512, 1024]:
    #                 grid_num += 1
    #                 # param.gamma              = gamma
    #                 # param.beta               = beta
    #                 param.decay              = decay
    #                 # param.add_rate           = add_rate
    #                 param.num_hidden         = num_hidden
    #                 param.type_graph4recon   = type_graph4recon
    #                 param.keep_threshold     = threshold[0]
    #                 param.dele_threshold     = threshold[1]
    #                 print('############################################################################')
    #                 print('grid search at experiment {}'.format(grid_num))
    #                 print('************************************************')
    #                 print('gamma                : {}'.format(param.gamma))
    #                 print('beta                 : {}'.format(param.beta))
    #                 print('add_rate             : {}'.format(param.add_rate))
    #                 print('decay                : {}'.format(param.decay))
    #                 print('num_hidden           : {}'.format(param.num_hidden))
    #                 print('keep_threshold       : {}'.format(param.keep_threshold))
    #                 print('dele_threshold       : {}'.format(param.dele_threshold))
    #                 print('type_graph4recon     : {}'.format(param.type_graph4recon))
    #                 print('************************************************')
    #                 ###############################################################################################################

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
    batch_size     = param.batch_size
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
                model = pretrain_tranductive(param, model, graph, graph.ndata["feat"], optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            elif margs.mode == 'inductive':
                model = pretrain_inductive(model, (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
        


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
    

        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")

