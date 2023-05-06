
from     easydict        import EasyDict
import   torch
from     model_zoo.GAE.RobustGAE.utils          import *
from     model_zoo.GAE.RobustGAE.build_easydict import *
from     model_zoo.GAE.RobustGAE.evaluation     import * 
from     datasets_dgl.data_dgl import *
from     datasets_dgl.utils import *


import   logging
from     tqdm import tqdm
from     model_zoo.GAE.RobustGAE.models import build_model
import  ipdb

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def pretrain_inductive(model, dstname ,graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
            graph, torch.arange(0, graph.num_nodes()), sampler,
            batch_size=2048,
            shuffle=True,
            drop_last=False,
            num_workers=1)

    logging.info("start training..")

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []

        dl = tqdm(dataloader)
        for input_nodes, output_nodes, blocks in dl:
            subgraph = dgl.node_subgraph(graph, input_nodes,store_ids=True)
            perturbed_adj_sparse = to_scipy(subgraph.adj())
            # print('===get perturbed edges===')
            jt = 0.03
            if dstname == 'polblogs':
                jt = 0
            adj_pre, removed_cnt = preprocess_adj(feat[subgraph.ndata[dgl.NID]], perturbed_adj_sparse,  threshold=jt)
            dl.set_description('removed %s edges in the original graph' % removed_cnt)
            adj_delete = perturbed_adj_sparse - adj_pre
            adj_delete = torch.tensor(adj_delete.todense())
            graph_processed = dgl.from_scipy(adj_pre).to(device)
            x = feat[subgraph.ndata[dgl.NID]].to(device)
            ##################

            loss, loss_dict = model(graph_processed, adj_delete, x, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {train_loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 1 == 0:
            node_classification_evaluation(model, graph, feat, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # return best_model
    return model


def pretrain_tranductive(model, dstname ,graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):

    ###################
    perturbed_adj_sparse = to_scipy(graph.adj())
    print('===get perturbed edges===')
    jt = 0.03
    if dstname == 'polblogs':
        jt = 0
    adj_pre,removed_cnt = preprocess_adj(feat, perturbed_adj_sparse,  threshold=jt)
    print('removed %s edges in the original graph' % removed_cnt)
    adj_delete = perturbed_adj_sparse - adj_pre
    adj_delete = torch.tensor(adj_delete.todense())
    graph_processed = dgl.from_scipy(adj_pre)
    ##################
    
    logging.info("start training..")
    graph_processed = graph_processed.to(device)
    x = feat.to(device)


    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph_processed, adj_delete, x, epoch)
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



def Train_RobustGAE_nodecls(margs):
    #########################
    if margs.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{margs.gpu_id}" if torch.cuda.is_available() else "cpu"

    mode = margs.mode
    dataset_name = margs.dataset

    DATASET = EasyDict()
    if dataset_name.split('-')[0] == 'Attack':
        # dataset_name = dataset_name.split('-')[1]
        DATASET.ATTACK = EasyDict()
        DATASET.ATTACK.PARAM = {
            "data":dataset_name,
            "attack":margs.attack.split('-')[0],
            "ptb_rate":margs.attack.split('-')[1]
        }
        # now just attack use
        dataset  = load_data(DATASET['ATTACK']['PARAM'])
        graph = dataset.graph
    else:
        DATASET.PARAM = {
            "data":dataset_name,
        }
        dataset  = load_data(DATASET['PARAM'])
        graph = dataset[0]
        # 注意 有的数据集没有加自环

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    

    dstname =  dataset_name.split('-')[1].lower() if dataset_name.split('-')[0] == 'Attack' else dataset_name.lower()
    num_classes = dataset.num_classes
    num_features = graph.ndata['feat'].shape[1]
    #######################
    
    MDT = build_easydict_nodecls()
    param         = MDT['MODEL']['PARAM']
    if param.use_cfg:
        param = load_best_configs(param, dstname , "./model_zoo/GAE/RobustGAE/configs.yml")

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
    param.num_features = num_features


    # ###################

    # perturbed_adj_sparse = to_scipy(graph.adj())
    
    # print('===get perturbed edges===')
    # jt = 0.03
    # if dstname == 'polblogs':
    #     jt = 0
    # adj_pre = preprocess_adj(graph.ndata['feat'], perturbed_adj_sparse,  threshold=jt)
    # adj_delete = perturbed_adj_sparse - adj_pre
    # adj_delete = torch.tensor(adj_delete.todense())
    # graph_processed = dgl.from_scipy(adj_pre)
    # ##################

    acc_list = []
    estp_acc_list = []

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name.lower()}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(param)
        model.to(device)
        model.register()
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None


        x = graph.ndata["feat"]

        if not load_model:
            if mode == 'tranductive':
                model = pretrain_tranductive(model, dstname, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            else:
                model = pretrain_inductive(model, dstname, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            # model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")

