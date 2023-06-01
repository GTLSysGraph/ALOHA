from     model_zoo.GAE.VGAE.Train_VGAE_linkcls import Train_VGAE_linkcls_usemd
from     model_zoo.GAE.VGAE.model import *
from     model_zoo.GAE.VGAE.build_easydict import *
from     easydict        import EasyDict
from     datasets_dgl.data_dgl import *
from    .utils import create_optimizer, accuracy
from     tqdm import tqdm
import copy

def Train_VGAE_nodecls(margs):
    # preprocessed
    #######################
    if margs.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{margs.gpu_id}" if torch.cuda.is_available() else "cpu"

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
            dataset  = load_data(dataset_name)
            if dataset_name == 'ogbn-arxiv':
                graph = process_OGB(dataset)
            else:   
                graph = dataset[0]
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
    

    feat = graph.ndata['feat']
    num_classes = dataset.num_classes
    num_features = graph.ndata['feat'].shape[1]
    #######################
    # cora raw 在这个数据集下的表现反而不如meta 0.2好 ，链接预测 why？
    # dataset = CoraGraphDataset()
    # graph = dataset[0]

    MDT = build_easydict()
    args1  = MDT['MODEL']['PARAM']
    vgae_model = VGAEModel(num_features, args1.hidden1, args1.hidden2, device)
    # pretrain
    Train_VGAE_linkcls_usemd(args1, vgae_model, graph, device)
    # nodecls
    model = vgae_model.to(device)
    args2=  MDT['MODEL']['NODECLS']
    Train_VGAE_nodeclas_usemd(args2, model , graph, feat, num_classes, device)







def Train_VGAE_nodeclas_usemd(args2, model, graph, x, num_classes, device, mute=False):
    lr_f            = args2.lr_f
    weight_decay_f  = args2.weight_decay_f
    max_epoch_f     = args2.max_epoch_f
    linear_prob     = args2.linear_prob

    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        raise Exception('not support finetune!')
        # 这里要注意和GraphMAE不一样的地方，GraphMAE里的encoder是一个类，而这里不是，是bound method，所以并不能deepcopy，无法求导，所以VGAE这里只用linear验证吧，linear_prob不可用
        # 注意这里改变了encoder的结构encoder和model.encoder应该是分开的，但是这里是关联的，encoder变了，model.encoder也变，应该deepcopy新的模型分开形成独立个体，原始代码写的有点问题
        # encoder = copy.deepcopy(model.encoder)
        # encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, optimizer_f, max_epoch_f, device, mute)
    return final_acc, estp_acc



class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits


def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc

