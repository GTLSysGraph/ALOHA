
from     easydict        import EasyDict
import   torch
from     model_zoo.GAE.DiffMGAE.utils          import *
from     model_zoo.GAE.DiffMGAE.build_easydict import *
from     model_zoo.GAE.DiffMGAE.evaluation     import * 
from     datasets_dgl.data_dgl import *
from     datasets_graphsaint.data_graphsaint import *

import   logging
from     tqdm import tqdm
from     model_zoo.GAE.DiffMGAE.models import build_model
import  ipdb

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def pretrain_tranductive(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training DiffGMAE node classification..")
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



def Train_DiffMGAE_nodecls(margs):
    #########################
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode = margs.mode
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
            graph = dgl.add_self_loop(graph) # graphmae + self loop这结果也太好了，分析一下，有点意思
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
    
    dstname =  dataset_name.split('-')[1].lower() if dataset_name.split('-')[0] == 'Attack' else dataset_name.lower()
    num_classes = dataset.num_classes
    num_features = graph.ndata['feat'].shape[1]
    #######################
    
    MDT = build_easydict()
    param         = MDT['MODEL']['PARAM']
    if param.use_cfg:
        param = load_best_configs(param, dstname , "./model_zoo/GAE/DiffMGAE/configs.yml")

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
    param.num_nodes    = graph.num_nodes()


    ##################################################################################################################
    # grid search
    grid_num = 0 
    for beta_schedule in ['linear','cosine','quadratic','sigmoid']:
        for remask_rate in [0.4, 0.5, 0.6, 0.8, 1.0, 0.2]:
            for lamda_loss in [0.1 ,0.4, 0.8, 0, 1]:
                for lamda_neg_ratio in [-10, 0, 1, 10]:
                    grid_num += 1
                    param.remask_rate = remask_rate
                    param.beta_schedule = beta_schedule
                    param.lamda_loss = lamda_loss
                    param.lamda_neg_ratio = lamda_neg_ratio
                    print('###################################################################################################################################################')
                    print('grid search at experiment {}'.format(grid_num))
                    print('************************************************')
                    print('remask_rate: {}'.format(param.remask_rate))
                    print('beta_schedule: {}'.format(param.beta_schedule))
                    print('lamda_loss: {}'.format(param.lamda_loss))
                    print('lamda_neg_ratio: {}'.format(param.lamda_neg_ratio))
                    print('************************************************')
    ###############################################################################################################
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


                        x = graph.ndata["feat"]

                        if not load_model:
                            if mode == 'tranductive':
                                model = pretrain_tranductive(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
                            else:
                                print("waiting...")

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
                    print('###################################################################################################################################################')
                    