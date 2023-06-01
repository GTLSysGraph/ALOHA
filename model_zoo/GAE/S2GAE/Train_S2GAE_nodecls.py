import torch
import time
from torch.utils.data import DataLoader
from model_zoo.GAE.S2GAE.build_easydict import *
import os.path as osp
import os
from datasets_pyg.data_pyg import get_dataset
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor
from model_zoo.GAE.S2GAE.utils import edgemask_um, edgemask_dm, do_edge_split_nc
import numpy as np
from model_zoo.GAE.S2GAE.model import LPDecoder_ogb as LPDecoder
from model_zoo.GAE.S2GAE.model import GCN_mgaev3 as GCN
from model_zoo.GAE.S2GAE.model import SAGE_mgaev2 as SAGE
from model_zoo.GAE.S2GAE.model import GIN_mgaev2 as GIN

from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from .evaluation_linear_probe import *

def random_edge_mask(args, edge_index, device, num_nodes):
    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * args.mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].to(device)

    edge_index_train, _ = add_self_loops(edge_index_train, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index_train).t()
    return adj, edge_index_train, edge_index_mask

def train(model, predictor, data, edge_index, optimizer, args):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    if args.mask_type == 'um':
        adj, _, pos_train_edge = edgemask_um(args.mask_ratio, edge_index, data.x.device, data.x.shape[0])
    else:
        adj, _, pos_train_edge = edgemask_dm(args.mask_ratio, edge_index, data.x.device, data.x.shape[0])
    adj = adj.to(data.x.device)

    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), args.batch_size,
                           shuffle=True)):
        optimizer.zero_grad()

        h = model(data.x, adj)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h, edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.x.shape[0], edge.size(), dtype=torch.long,
                             device=data.x.device)
        neg_out = predictor(h, edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples



@torch.no_grad()
def test(model, predictor, data, pos_test_edge, neg_test_edge, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.full_adj_t)

    pos_test_edge = pos_test_edge.to(data.x.device)
    neg_test_edge = neg_test_edge.to(data.x.device)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)
    test_auc = roc_auc_score(test_true, test_pred)
    return test_auc


def extract_feature_list_layer2(feature_list):
    xx_list = []
    xx_list.append(feature_list[-1])
    tmp_feat = torch.cat(feature_list, dim=-1)
    xx_list.append(tmp_feat)
    return xx_list


def accuracy(preds, labels):
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)


def test_classify(feature, labels, args):
    f1_mac = []
    f1_mic = []
    accs = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(feature):
        train_X, train_y = feature[train_index], labels[train_index]
        test_X, test_y = feature[test_index], labels[test_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        micro = f1_score(test_y, preds, average='micro')
        macro = f1_score(test_y, preds, average='macro')
        acc = accuracy(preds, test_y)
        accs.append(acc)
        f1_mac.append(macro)
        f1_mic.append(micro)
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    accs = np.array(accs)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    accs = np.mean(accs)
    print('Testing based on svm: ',
          'f1_micro=%.4f' % f1_mic,
          'f1_macro=%.4f' % f1_mac,
          'acc=%.4f' % accs)
    return f1_mic, f1_mac, accs




def Train_S2GAE_nodecls(margs):
    dataset_name = margs.dataset
    device = torch.device('cuda:{}'.format(margs.gpu_id) if torch.cuda.is_available() else 'cpu')

    path = osp.expanduser('/home/songsh/GCL/datasets_pyg')
    if dataset_name.split('-')[0] == 'Attack':
        attackmethod = margs.attack.split('-')[0]
        attackptb    = margs.attack.split('-')[1]
        path = osp.expanduser('/home/songsh/GCL/datasets_pyg/Attack_data')
        dataset = get_dataset(path, dataset_name, attackmethod, attackptb)
    else:
        path = osp.join(path, dataset_name)
        dataset = get_dataset(path, dataset_name)

    data = dataset[0]

    MDT = build_easydict_nodecls()
    config         = MDT['MODEL']['PARAM']

    if data.is_undirected():
        edge_index = data.edge_index
    else:
        print('### Input graph {} is directed'.format(margs.dataset))
        edge_index = to_undirected(data.edge_index)
    data.full_adj_t = SparseTensor.from_edge_index(edge_index).t()

    edge_index, test_edge, test_edge_neg = do_edge_split_nc(edge_index, data.x.shape[0])

    labels = data.y.view(-1)

    save_path_model = './model_zoo/GAE/S2GAE/weight/s2gaesvm-' + config['use_sage'] + '_{}_{}'.format(dataset_name,  config['mask_type']) + '_{}'.format(config['num_layers']) + '_hidd{}-{}-{}-{}'.format(config['hidden_channels'], config['mask_ratio'], config['decode_layers'], config['decode_channels']) + '_model.pth'
    save_path_predictor = './model_zoo/GAE/S2GAE/weight/s2gaesvm' + config['use_sage'] + '_{}_{}'.format(dataset_name,
                                                                          config['mask_type']) + '_{}'.format(
        config['num_layers']) + '_hidd{}-{}-{}-{}'.format(config['hidden_channels'], config['mask_ratio'], config['decode_layers'],
                                                     config['decode_channels']) + '_pred.pth'

    out2_dict = {0: 'last', 1: 'combine'}
    result_dict = out2_dict

    svm_result_final = np.zeros(shape=[config['runs'], len(out2_dict)])
    # Use training + validation edges for inference on test set.
    
    data = data.to(device)

    if config['use_sage'] == 'SAGE':
        model = SAGE(data.num_features, config['hidden_channels'],
                     config['hidden_channels'], config['num_layers'],
                     config['dropout']).to(device)
    elif config['use_sage'] == 'GIN':
        model = GIN(data.num_features, config['hidden_channels'],
                     config['hidden_channels'], config['num_layers'],
                     config['dropout']).to(device)
    else:
        model = GCN(data.num_features, config['hidden_channels'],
                    config['hidden_channels'], config['num_layers'],
                    config['dropout']).to(device)

    predictor = LPDecoder(config['hidden_channels'], config['decode_channels'], 1, config['num_layers'],
                              config['decode_layers'], config['dropout']).to(device)

    print('Start training with mask ratio={} # optimization edges={} / {}'.format( config['mask_ratio'],
                                                                             int( config['mask_ratio'] *
                                                                                 edge_index.shape[0]), edge_index.shape[0]))


    for run in range(config['runs']):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=config['lr'])

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + config['epochs']):
            t1 = time.time()
            loss = train(model, predictor, data, edge_index, optimizer,
                         config)
            t2 = time.time()
            #这里的test使用edge的预测情况做的auc
            auc_test = test(model, predictor, data, test_edge, test_edge_neg,
                           config['batch_size'])

            if auc_test > best_valid:
                best_valid = auc_test
                best_epoch = epoch
                torch.save(model.state_dict(), save_path_model)
                torch.save(predictor.state_dict(), save_path_predictor)
                cnt_wait = 0
            else:
                cnt_wait += 1

            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Best_epoch: {best_epoch:02d}, '
                  f'Best_valid: {100 * best_valid:.2f}%, '
                  f'Loss: {loss:.4f}, ')
            print('***************')
            if cnt_wait == 50:
                print('Early stop at {}'.format(epoch))
                break

        print('##### Testing on {}/{}'.format(run, config['runs']))

        model.load_state_dict(torch.load(save_path_model))
        predictor.load_state_dict(torch.load(save_path_predictor))
        feature = model(data.x, data.full_adj_t)
        feature = [feature_.detach() for feature_ in feature]
        
        feature_list = extract_feature_list_layer2(feature)
    
    #     for i, feature_tmp in enumerate(feature_list):
    #         # 改成只用测试集的结果而不是用全图 评估方式好像也要改一下
    #         f1_mic_svm, f1_mac_svm, acc_svm = test_classify(feature_tmp[data.test_mask].data.cpu().numpy(), labels[data.test_mask].data.cpu().numpy(),
    #                                                         margs)
    #         svm_result_final[run, i] = acc_svm
    #         print('**** SVM test acc on Run {}/{} for {} is F1-mic={} F1-mac={} acc={}'
    #               .format(run + 1, config['runs'], result_dict[i], f1_mic_svm, f1_mac_svm, acc_svm))

    # svm_result_final = np.array(svm_result_final)

    # if osp.exists(save_path_model):
    #     os.remove(save_path_model)
    #     os.remove(save_path_predictor)
    #     print('Successfully delete the saved models')

    # print('\n------- Print final result for SVM')
    # for i in range(len(out2_dict)):
    #     temp_resullt = svm_result_final[:, i]
    #     print('#### Final svm test result on {} is mean={} std={}'.format(result_dict[i], np.mean(temp_resullt),
    #                                                                       np.std(temp_resullt)))


        if dataset_name.split('-')[1] in ['Cora','Pubmed']:
            weight_decay_f =  1e-4
        elif dataset_name.split('-')[1] in ['Citeseer']:
            weight_decay_f = 0.01

        for i, feature_tmp in enumerate(feature_list):
            final_acc, estp_acc = node_classification_evaluation(data, feature_tmp, labels, dataset.num_classes, 0.01, weight_decay_f, 300, device)
            svm_result_final[run, i] = final_acc
            print('**** Linear probe test acc on Run {}/{} for {} is acc={} estp_acc={}'
                  .format(run + 1, config['runs'], result_dict[i], final_acc, estp_acc))


    svm_result_final = np.array(svm_result_final)

    if osp.exists(save_path_model):
        os.remove(save_path_model)
        os.remove(save_path_predictor)
        print('Successfully delete the saved models')

    print('\n------- Print final result for SVM')
    for i in range(len(out2_dict)):
        temp_resullt = svm_result_final[:, i]
        print('#### Final Linear probe test result on {} is mean={} std={}'.format(result_dict[i], np.mean(temp_resullt),
                                                                          np.std(temp_resullt)))