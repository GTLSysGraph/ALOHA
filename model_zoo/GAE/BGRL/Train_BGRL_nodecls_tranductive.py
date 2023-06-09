from torch.utils.tensorboard import SummaryWriter
import os.path as osp
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from datasets_pyg.data_pyg import get_dataset 
from .build_easydict import *
from .bgrl.transformers import *
from .bgrl.models import *
from .bgrl.predictors import *
from .bgrl.bgrl import *
from .bgrl.scheduler import *
from .bgrl.logistic_regression_eval import *
from utils import set_seed,load_best_configs,index_to_mask
from tqdm import tqdm
import os

def Train_BGRL_nodecls_tranductive(margs):
    dataset_name = margs.dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ])

    path = osp.expanduser('/home/songsh/GCL/datasets_pyg')
    if dataset_name.split('-')[0] == 'Attack':
        attackmethod = margs.attack.split('-')[0]
        attackptb    = margs.attack.split('-')[1]
        path = osp.expanduser('/home/songsh/GCL/datasets_pyg/Attack_data')
        dataset = get_dataset(path, dataset_name, attackmethod, attackptb)
    else:
        path = osp.join(path, dataset_name)
        dataset = get_dataset(path, dataset_name)


    if dataset_name == 'ogbn-arxiv':
        data = transform(dataset[0])
        split_idx = dataset.get_idx_split()
        # data.train_nodes = split_idx['train']
        # data.val_nodes = split_idx['valid']
        # data.test_nodes = split_idx['test']
        data.train_mask = index_to_mask(split_idx['train'],size=data.num_nodes)      
        data.val_mask   = index_to_mask(split_idx['valid'],size=data.num_nodes)    
        data.test_mask  = index_to_mask(split_idx['test'] ,size=data.num_nodes)    

    elif dataset_name in ['Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo']:
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data) # 这些数据集没有mask划分 划分完会有train_mask=[18333], val_mask=[18333], test_mask=[18333]  'Coauthor-CS'为例
    else:
        data = transform(dataset[0])

    data = data.cpu()

    MDT = build_easydict_nodecls()
    FLAGS = MDT['MODEL']['PARAM']

    if FLAGS.use_cfg:
        FLAGS = load_best_configs(FLAGS, dataset_name, "./model_zoo/GAE/BGRL/configs.yml")
    print(FLAGS)
    print(data)

    # set random seed
    if FLAGS.model_seed is not None:
        print('Random seed set to {}.'.format(FLAGS.model_seed))
        set_seed(FLAGS.model_seed)


    if dataset_name != 'WikiCS':
        num_eval_splits = FLAGS.num_eval_splits
    else:
        std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
        data.x = (data.x - mean) / std
        data.edge_index = to_undirected(data.edge_index)
        train_masks, val_masks, test_masks =  np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)
        dataset = [data]
        num_eval_splits = data.train_mask.shape[1]
    

    # 先判断一下是否存在已经训练好的encoder，如果存在就不训练了，直接evalute
    if os.path.exists(os.path.join(FLAGS.logdir, 'bgrl-{}.pt'.format(dataset_name))):
        if dataset_name != 'WikiCS':
            # 随机划分，其实这里是对全图的一部分随机节点进行评估，如果我们想对特定的test_mask进行评估可以用Wikics的方式，只不过只用一组[train_masks, val_masks, test_masks]就可以了,这里实现了一个fit_logistic_regression_fix_split，如果不想随机划分在下面替换就可以
            EVAL_BGRL_nodecls(dataset_name, data, [] , FLAGS, device)
        else:
            # 因为wikics有20中不同的划分
            EVAL_BGRL_nodecls(dataset_name, data, [train_masks, val_masks, test_masks] , FLAGS, device)
        return


    data = data.to(device)  # permanently move in gpu memory
    

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)   # 512, 256, 128
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = BGRL(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    # setup tensorboard and make custom layout
    writer = SummaryWriter(FLAGS.logdir)
    layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(num_eval_splits)]]}}
    writer.add_custom_scalars(layout)


    def train(step):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()

        x1, x2 = transform_1(data), transform_2(data)

        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)

        loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        loss.backward()

        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

        # log scalars
        writer.add_scalar('params/lr', lr, step)
        writer.add_scalar('params/mm', mm, step)
        writer.add_scalar('train/loss', loss, step)

    def eval(epoch):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        representations, labels = compute_representations(tmp_encoder, data, device)

        if dataset_name != 'WikiCS':
            # 这里实现了一个fit_logistic_regression_fix_split，如果想随机划分在下面替换就可以
            scores = fit_logistic_regression_fix_split(representations.cpu().numpy(), labels.cpu().numpy(), data.train_mask.cpu(), data.val_mask.cpu(), data.test_mask.cpu(),repeat=FLAGS.num_eval_splits)
            # scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),data_random_seed=FLAGS.data_seed, repeat=FLAGS.num_eval_splits)
        else:
            scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(),
                                                           train_masks, val_masks, test_masks)

        for i, score in enumerate(scores):
            writer.add_scalar(f'accuracy/test_{i}', score, epoch)



    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        train(epoch-1)
        if epoch % FLAGS.eval_epochs == 0:
            eval(epoch)
    
    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, 'bgrl-{}.pt'.format(dataset_name)))


    if dataset_name != 'WikiCS':
        EVAL_BGRL_nodecls(dataset_name, data, [] , FLAGS, device)
    else:
        EVAL_BGRL_nodecls(dataset_name, data, [train_masks, val_masks, test_masks] , FLAGS, device)




    
def EVAL_BGRL_nodecls(dataset_name, data, mask ,FLAGS, device):
    print('Using {} for evaluation.'.format(device))
    if dataset_name == 'WikiCS':
        train_masks, val_masks, test_masks = mask
    # data = dataset[0]
    data = data.to(device)

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True) # 512, 256, 128
    load_trained_encoder(encoder, FLAGS.ckpt_path, device)
    encoder.eval()

    # compute representations
    representations, labels = compute_representations(encoder, data, device)

    if dataset_name != 'WikiCS':
        # 这里实现了一个fit_logistic_regression_fix_split，想随机划分在下面替换就可以
        score = fit_logistic_regression_fix_split(representations.cpu().numpy(), labels.cpu().numpy(), data.train_mask.cpu(), data.val_mask.cpu(), data.test_mask.cpu())[0]
        # score = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy())[0]
    else:
        scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(),
                                                       train_masks, val_masks, test_masks)
        score = np.mean(scores)

    print('Test score: %.5f' %score)



