import torch
from utils import set_seed,load_best_configs
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from .build_easydict import *
from datasets_pyg.data_pyg import get_dataset 
import os.path as osp
from torch_geometric.loader import DataLoader
from .bgrl.data import *
from .bgrl.transformers import *
from .bgrl.models import *
from .bgrl.predictors import *
from .bgrl.bgrl import *
from .bgrl.scheduler import *
from .bgrl.linear_eval_ppi import *
from tqdm import tqdm
import os

def Train_BGRL_nodecls_inductive(margs):
    dataset_name = margs.dataset
    assert margs.dataset == 'PPI'
    assert margs.mode =='inductive'

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MDT = build_easydict_nodecls()
    FLAGS = MDT['MODEL']['PARAM']

    if FLAGS.use_cfg:
            FLAGS = load_best_configs(FLAGS, dataset_name, "./model_zoo/GAE/BGRL/configs.yml")
    print(FLAGS)

    # set random seed
    if FLAGS.model_seed is not None:
        print('Random seed set to {}.'.format(FLAGS.model_seed))
        set_seed(FLAGS.model_seed)

    # setup tensorboard
    writer = SummaryWriter(FLAGS.logdir)

    # load data
    path = osp.expanduser('/home/songsh/GCL/datasets_pyg/')
    path = osp.join(path, dataset_name)
    train_dataset,val_dataset,test_dataset = get_dataset(path, dataset_name)


    # 先判断一下是否存在已经训练好的encoder，如果存在就不训练了，直接evalute
    if os.path.exists(os.path.join(FLAGS.logdir, 'bgrl-{}.pt'.format(dataset_name))):
        EVAL_BGRL_nodecls([train_dataset,val_dataset,test_dataset], FLAGS, device)
        return


    # train BGRL using both train and val splits
    train_loader = DataLoader(ConcatDataset([train_dataset, val_dataset]), batch_size=FLAGS.batch_size, shuffle=True,
                              num_workers=FLAGS.num_workers)

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)

    # build networks
    input_size, representation_size = train_dataset.num_node_features, 512
    encoder = GraphSAGE_GCN(input_size, 512, 512)
    # encoder =GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = BGRL(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=0., weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_steps, FLAGS.steps)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.steps)

    def train(data, step):
        model.train()

        # move data to gpu and transform
        data = data.to(device)
        x1, x2 = transform_1(data), transform_2(data)

        # update learning rate
        lr = lr_scheduler.get(step)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()
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


    def eval(step):
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()

        train_data = compute_representations(tmp_encoder, train_dataset, device)
        val_data = compute_representations(tmp_encoder, val_dataset, device)
        test_data = compute_representations(tmp_encoder, test_dataset, device)

        val_f1, test_f1 = ppi_train_linear_layer(train_dataset.num_classes, train_data, val_data, test_data, device)
        writer.add_scalar('accuracy/val', val_f1, step)
        writer.add_scalar('accuracy/test', test_f1, step)

    train_iter = iter(train_loader)

    for step in tqdm(range(1, FLAGS.steps + 1)):
        data = next(train_iter, None)
        if data is None:
            train_iter = iter(train_loader)
            data = next(train_iter, None)

        train(data, step)

        if step % FLAGS.eval_steps == 0:
            eval(step)

    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, 'bgrl-{}.pt'.format(dataset_name)))
    EVAL_BGRL_nodecls([train_dataset,val_dataset,test_dataset], FLAGS, device)







def EVAL_BGRL_nodecls(dataloader ,FLAGS, device):
    train_dataset,val_dataset,test_dataset = dataloader

    # build networks

    input_size, representation_size = train_dataset.num_node_features, 512
    encoder = GraphSAGE_GCN(input_size, 512, 512)
    load_trained_encoder(encoder, FLAGS.ckpt_path, device)
    encoder.eval()

    # compute representations
    train_data = compute_representations(encoder, train_dataset, device)
    val_data = compute_representations(encoder, val_dataset, device)
    test_data = compute_representations(encoder, test_dataset, device)

    val_f1, test_f1 = ppi_train_linear_layer(train_dataset.num_classes, train_data, val_data, test_data, device)
    print('Test F1-score: %.5f' % test_f1)