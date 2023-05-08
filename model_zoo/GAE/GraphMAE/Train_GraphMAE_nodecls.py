
from     easydict        import EasyDict
import   torch
from     model_zoo.GAE.GraphMAE.utils          import *
from     model_zoo.GAE.GraphMAE.build_easydict import *
from     model_zoo.GAE.GraphMAE.evaluation_tranductive     import * 
from     model_zoo.GAE.GraphMAE.evaluation_inductive       import * 
from     datasets_dgl.data_dgl import *

import   logging
from     tqdm import tqdm
from     model_zoo.GAE.GraphMAE.models import build_model


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
    if margs.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{margs.gpu_id}" if torch.cuda.is_available() else "cpu"

    dataset_name = margs.dataset
    DATASET = EasyDict()
    if margs.mode == 'tranductive':
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
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph) # graphmae + self loop这结果也太好了，分析一下，有点意思

        else:
            DATASET.PARAM = {
                "data":dataset_name,
            }
            dataset  = load_data(DATASET['PARAM'])
            if dataset_name == 'ogbn-arxiv':
                graph = process_OGB(dataset)
            else:   
                graph = dataset[0]
                graph = dgl.remove_self_loop(graph)
                graph = dgl.add_self_loop(graph) # graphmae + self loop这结果也太好了，分析一下，有点意思

        num_classes = dataset.num_classes
        num_features = graph.ndata['feat'].shape[1]

    elif margs.mode == 'inductive':
            (
                train_dataloader,
                valid_dataloader, 
                test_dataloader, 
                eval_train_dataloader, 
                num_features, 
                num_classes
            ) = load_inductive_dataset(dataset_name)
    
    ##########################
    
    MDT = build_easydict_nodecls()
    param         = MDT['MODEL']['PARAM']
    if param.use_cfg:
        param = load_best_configs(param, dataset_name.split('-')[1].lower() if dataset_name.split('-')[0] == 'Attack' else dataset_name.lower() , "./model_zoo/GAE/GraphMAE/configs.yml")

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
                x = graph.ndata["feat"]
                model = pretrain_tranductive(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            elif margs.mode == 'inductive':
                model = pretrain_inductive(model, (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            # model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        if margs.mode == 'tranductive':
            final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
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

