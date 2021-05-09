from my_transformers.transformers import BertConfig,BertModel,BertTokenizer
from my_transformers.transformers import AdamW

import copy
import time
import random
import argparse
import os
import json
import numpy as np
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data as data
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd

from dataset import loader


class Stable_PROTO(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.N, self.K, self.L = args.N, args.K, args.L
        self.max_length = args.max_length
        self.coder = BERT(args.N, args.max_length)
        self.hidden_size = 128
        self.bilstm = nn.LSTM(input_size=768, hidden_size=self.hidden_size, num_layers=1,
                              bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(args.max_length, 1)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, inputs):  # inputs: [N*K, max_length, 768]

        # ebd, (hn, cn) = self.bilstm(inputs)  # -> [N*K, max_length, 256]
        # outputs = self.linear(ebd.transpose(1, 2).contiguous()).squeeze(-1)  # -> [N*K, 128]
        pass

    def BiLSTM(self, inputs):  # inputs: [N*K, max_length, 768]

        ebd, (hn, cn) = self.bilstm(inputs)  # [N*K, max_length, 256]
        return ebd

    def FC(self, inputs):  # inputs: [N*K, max_length, 256]

        outputs = self.fc(inputs.transpose(1, 2).contiguous()).squeeze(-1)  # [N*K, 256]

        return outputs

    def loss(self, logits, label):
        loss_ce = self.cost(logits, label.view(-1))
        return loss_ce

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

class BERT(nn.Module):
    def __init__(self, N, max_length, blank_padding=True):
        super(BERT, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.n_way = N
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.pretrained_path = 'bert-base-uncased'
        if os.path.exists(self.pretrained_path):
            config = BertConfig.from_pretrained(os.path.join(self.pretrained_path, 'bert-base-uncased-config.json'))
            self.bert = BertModel.from_pretrained(
                os.path.join(self.pretrained_path, 'bert-base-uncased-pytorch_model.bin'), config=config)
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(self.pretrained_path, 'bert-base-uncased-vocab.txt'))
        else:
            self.bert = BertModel.from_pretrained(self.pretrained_path)
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        tokens, att_masks, outputs = [], [], []
        for _ in inputs:
            token, att_mask = self.tokenize(_)
            tokens.append(token)
            att_masks.append(att_mask)
        token = torch.cat([t for t in tokens], 0)  # [N*K,max_length]
        att_mask = torch.cat([a for a in att_masks], 0)  # [N*K,max_length]
        # sequence_output,pooled_output=self.bert(token,attention_mask=att_mask)
        sequence_output = self.bert(token, attention_mask=att_mask)  # [N*K,max_length,bert_size]
        # for i in range(token.size(0)):
        #     outputs.append(self.entity_start_state(head_poses[i], sequence_output[i]))
        # outputs = torch.cat([o for o in outputs], 0)
        # outputs = self.dropout(outputs)  # [N*K,bert_size*2]
        return sequence_output

    def entity_start_state(self, head_pos, sequence_output):  # 就是将BERT中两个实体前的标记位对应的输出拼接后输出作为整个句子的embedding。
        if head_pos[0] == -1 or head_pos[0] >= self.max_length:
            head_pos[0] = 0
            # raise Exception("[ERROR] no head entity")
        if head_pos[1] == -1 or head_pos[1] >= self.max_length:
            head_pos[1] = 0
            # raise Exception("[ERROR] no tail entity")
        res = torch.cat([sequence_output[head_pos[0]], sequence_output[head_pos[1]]], 0)
        return res.unsqueeze(0)

    def tokenize(self, inputs):
        tokens = inputs

        re_tokens, cur_pos = ['[CLS]', ], 0
        for token in tokens:
            token = token[0].lower()
            re_tokens += self.tokenizer.tokenize(token)
            cur_pos += 1
        re_tokens.append('[SEP]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)
        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1
        if self.cuda:
            indexed_tokens, att_mask = indexed_tokens.cuda(), att_mask.cuda()
        return indexed_tokens, att_mask  # both [1,max_length]


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def parse_args():
    parser = argparse.ArgumentParser(
            description="Few Shot Text Classification with Stable-PROTO")

    # data configuration
    parser.add_argument("--data_path", type=str, default="data/amazon.json", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="name of the dataset. Options: [20newsgroup, amazon, huffpost,reuters, rcv1, fewrel]")
    parser.add_argument("--n_train_class", type=int, default=10, help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5, help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=9, help="number of meta-test classes")

    parser.add_argument("--seed", type=int, default=330, help="seed")
    # parser.add_argument("--cuda", type=int, default=0, help="cuda device, -1 for cpu")

    parser.add_argument("--meta_lr", type=float, default=1, help="meta learning rate(out)")
    parser.add_argument("--task_lr", type=float, default=7e-2, help="meta learning rate(in)")

    parser.add_argument("--train_iter", type=int, default=10000, help="max num of training epochs")
    parser.add_argument("--val_iter", type=int, default=1000, help="max num of validating epochs")
    parser.add_argument("--test_iter", type=int, default=2000, help="max num of testing epochs")

    parser.add_argument("--train_task_step", type=int, default=5, help="max num of training task steps")
    parser.add_argument("--val_test_task_step", type=int, default=10, help="max num of val(test) task steps")

    parser.add_argument("--N", type=int, default=5, help="#classes for each task")
    parser.add_argument("--K", type=int, default=1, help="#support examples for each class for each task")
    parser.add_argument("--L", type=int, default=25, help="#query examples for each class for each task")
    parser.add_argument('--max_length', default=50, type=int, help='max length')

    parser.add_argument("--wv_path", type=str,
                        default='../pretrain_wordvec',
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default='../pretrain_wordvec/wiki.en.vec',
                        help=("Name of pretrained word embeddings."))

    parser.add_argument("--bert", type=bool, default=True, help="use bert or not")


    # # load bert embeddings for sent-level datasets (optional)
    # parser.add_argument("--n_workers", type=int, default=10,
    #                     help="Num. of cores used for loading data. Set this "
    #                     "to zero if you want to use all the cpus.")
    # # parser.add_argument("--bert_cache_dir", default=None, type=str,
    # #                     help=("path to the cache_dir of transformers"))
    # # parser.add_argument("--pretrained_bert", default=None, type=str,
    # #                     help=("path to the pre-trained bert embeddings."))
    #
    # # task configuration
    # parser.add_argument("--N", type=int, default=5,
    #                     help="#classes for each task")
    # parser.add_argument("--K", type=int, default=5,
    #                     help="#support examples for each class for each task")
    # parser.add_argument("--L", type=int, default=25,
    #                     help="#query examples for each class for each task")
    #
    # # train/test configuration
    # parser.add_argument("--train_epochs", type=int, default=1000,
    #                     help="max num of training epochs")
    # parser.add_argument("--train_episodes", type=int, default=100,
    #                     help="#tasks sampled during each training epoch")
    # parser.add_argument("--val_episodes", type=int, default=100,
    #                     help="#asks sampled during each validation epoch")
    # parser.add_argument("--test_episodes", type=int, default=1000,
    #                     help="#tasks sampled during each testing epoch")
    #
    # # settings for finetuning baseline
    # # parser.add_argument("--finetune_loss_type", type=str, default="softmax",
    # #                     help="type of loss for finetune top layer"
    # #                     "options: [softmax, dist]")
    # # parser.add_argument("--finetune_maxepochs", type=int, default=5000,
    # #                     help="number epochs to finetune each task for (inner loop)")
    # # parser.add_argument("--finetune_episodes", type=int, default=10,
    # #                     help="number tasks to finetune for (outer loop)")
    # # parser.add_argument("--finetune_split", default=0.8, type=float,
    # #                     help="percent of train data to allocate for val"
    # #                          "when mode is finetune")
    #
    # # model options
    # # parser.add_argument("--embedding", type=str, default="avg",
    # #                     help=("document embedding method. Options: "
    # #                           "[avg, tfidf, meta, oracle, cnn]"))
    # # parser.add_argument("--classifier", type=str, default="nn",
    # #                     help=("classifier. Options: [nn, proto, r2d2, mlp]"))
    # # parser.add_argument("--auxiliary", type=str, nargs="*", default=[],
    # #                     help=("auxiliary embeddings (used for fewrel). "
    # #                           "Options: [pos, ent]"))
    #
    # # cnn configuration
    # # parser.add_argument("--cnn_num_filters", type=int, default=50,
    # #                     help="Num of filters per filter size [default: 50]")
    # # parser.add_argument("--cnn_filter_sizes", type=int, nargs="+",
    # #                     default=[3, 4, 5],
    # #                     help="Filter sizes [default: 3]")
    #
    # # nn configuration
    # # parser.add_argument("--nn_distance", type=str, default="l2",
    # #                     help=("distance for nearest neighbour. "
    # #                           "Options: l2, cos [default: l2]"))
    # #
    # # # proto configuration
    # # parser.add_argument("--proto_hidden", nargs="+", type=int,
    # #                     default=[300, 300],
    # #                     help=("hidden dimension of the proto-net"))
    # #
    # # # maml configuration
    # # parser.add_argument("--maml", action="store_true", default=False,
    # #                     help=("Use maml or not. "
    # #                     "Note: maml has to be used with classifier=mlp"))
    # # parser.add_argument("--mlp_hidden", nargs="+", type=int, default=[300, 5],
    # #                     help=("hidden dimension of the proto-net"))
    # # parser.add_argument("--maml_innersteps", type=int, default=10)
    # # parser.add_argument("--maml_batchsize", type=int, default=10)
    # # parser.add_argument("--maml_stepsize", type=float, default=1e-1)
    # # parser.add_argument("--maml_firstorder", action="store_true", default=False,
    # #                     help="truncate higher order gradient")
    # #
    # # # lrd2 configuration
    # # parser.add_argument("--lrd2_num_iters", type=int, default=5,
    # #                     help=("num of Newton steps for LRD2"))
    # #
    # # # induction networks configuration
    # # parser.add_argument("--induct_rnn_dim", type=int, default=128,
    # #                     help=("Uni LSTM dim of induction network's encoder"))
    # # parser.add_argument("--induct_hidden_dim", type=int, default=100,
    # #                     help=("tensor layer dim of induction network's relation"))
    # # parser.add_argument("--induct_iter", type=int, default=3,
    # #                     help=("num of routings"))
    # # parser.add_argument("--induct_att_dim", type=int, default=64,
    # #                     help=("attention projection dim of induction network"))
    # #
    # # # aux ebd configuration (for fewrel)
    # # parser.add_argument("--pos_ebd_dim", type=int, default=5,
    # #                     help="Size of position embedding")
    # # parser.add_argument("--pos_max_len", type=int, default=40,
    # #                     help="Maximum sentence length for position embedding")
    # #
    # # # base word embedding
    # # parser.add_argument("--wv_path", type=str,
    # #                     default='../pretrain_wordvec',
    # #                     help="path to word vector cache")
    # # parser.add_argument("--word_vector", type=str, default='../pretrain_wordvec/wiki.en.vec',
    # #                     help=("Name of pretrained word embeddings."))
    # # parser.add_argument("--finetune_ebd", action="store_true", default=False,
    # #                     help=("Finetune embedding during meta-training"))
    # #
    # # # options for the distributional signatures
    # # parser.add_argument("--meta_idf", action="store_true", default=False,
    # #                     help="use idf")
    # # parser.add_argument("--meta_iwf", action="store_true", default=False,
    # #                     help="use iwf")
    # # parser.add_argument("--meta_w_target", action="store_true", default=False,
    # #                     help="use target importance score")
    # # parser.add_argument("--meta_w_target_lam", type=float, default=1,
    # #                     help="lambda for computing w_target")
    # # parser.add_argument("--meta_target_entropy", action="store_true", default=False,
    # #                     help="use inverse entropy to model task-specific importance")
    # # parser.add_argument("--meta_ebd", action="store_true", default=False,
    # #                     help="use word embedding into the meta model "
    # #                     "(showing that revealing word identity harm performance)")
    #
    # # training options
    # parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    # parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    # parser.add_argument("--patience", type=int, default=20, help="patience")
    # parser.add_argument("--clip_grad", type=float, default=None,
    #                     help="gradient clipping")
    # parser.add_argument("--cuda", type=int, default=-1,
    #                     help="cuda device, -1 for cpu")
    # parser.add_argument("--mode", type=str, default="test",
    #                     help=("Running mode."
    #                           "Options: [train, test, finetune]"
    #                           "[Default: test]"))
    # parser.add_argument("--save", action="store_true", default=False,
    #                     help="train the model")
    # parser.add_argument("--notqdm", action="store_true", default=False,
    #                     help="disable tqdm")
    # parser.add_argument("--result_path", type=str, default="")
    # parser.add_argument("--snapshot", type=str, default="",
    #                     help="path to the pretraiend weights")

    # parser.add_argument("--pretrain", type=str, default=None, help="path to the pretraiend weights for MLAD")

    return parser.parse_args()


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    print("""
          _________ __        ___.   .__                  ____________________ ___________________________   
         /   _____//  |______ \\_ |__ |  |   ____          \\______   \\______   \\_____  \\__    ___/\\_____  \\  
         \\_____  \\   __\\__  \\ | __ \\|  | _/ __ \\   ______ |     ___/|       _/ /   |   \\|    |    /   |   \\ 
         /        \\|  |  / __\\| \\_\\ \\  |_\\  ___/  /_____/ |    |    |    |   \\/    |    \\    |   /    |    \\
        /_______  /|__|  (___ /___  /____/\\___  >         |____|    |____|_  /\\_______  /____|   \\_______  /
                \\/          \\/    \\/          \\/                           \\/         \\/                 \\/                                                            
                                                                        
    """)


def neg_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_ones(class_name2, support2, support_label, query2, model, val_test_task_step, task_lr, N, K):
    for k in range(val_test_task_step):
        support3 = model.mlp2(support2, p0)  # [N*K, 256]
        class_name3 = model.mlp2(class_name2, p0)  # [N, 256]
        logits = neg_dist(support3, class_name3)  # [N*K, N]
        # logits = -logits / torch.mean(logits, dim=0)
        # _, pred = torch.max(logits, 1)
        loss = model.loss(logits, support_label)
        if torch.isnan(loss):
            print(loss)

        zero_grad(p0)
        grads = autograd.grad(loss, p0)
        p0 = p0 - task_lr * grads[0]

    query3 = model.mlp2(query2, p0)
    class_name3 = model.mlp2(class_name2, p0)
    logits_q = neg_dist(query3, class_name3)
    # logits_q = -logits_q / torch.mean(logits_q, dim=0)
    return logits_q


def train_one_batch(args, class_name, support0, support_label, query0, query_label, model, train_task_step, task_lr, it, task_optimizer):

    N = args.N
    K = args.K
    class_name1, support1, query1 = model.coder(class_name), model.coder(support0), model.coder(query0)  # [N/N*K/N*L, max_length, 768]
    class_name2, support2, query2 = model.BiLSTM(class_name1), model.BiLSTM(support1), model.BiLSTM(query1)  # # [N/N*K/N*L, max_length, 256]

    for it in range(train_task_step):
        class_name3, support3 = model.FC(class_name2), model.FC(support2)  # [N/N*K/N*L, 256]
        # logits_q = train_ones(class_name2, support2, support_label, query2, model, train_task_step, task_lr, N, K)
        logits = neg_dist(support3, class_name3)  # [N*K, N]
        # logits = -logits / torch.mean(logits, dim=0)
        # _, pred = torch.max(logits, 1)
        loss = model.loss(logits, support_label)
        task_optimizer.zero_grad()
        loss.backward()
        task_optimizer.step()

    class_name3, query3 = model.FC(class_name2), model.FC(query2)
    logits_q = neg_dist(query3, class_name3)
    loss_q = model.loss(logits_q, query_label)
    if torch.isnan(loss_q):
        print(loss_q)
    _, pred = torch.max(logits_q, 1)
    right_q = model.accuracy(pred, query_label)

    return loss_q, right_q


def train_model(model, args):

    meta_lr = args.meta_lr
    task_lr = args.task_lr
    train_iter = args.train_iter
    val_iter = args.val_iter
    test_iter = args.test_iter
    val_test_task_step = args.val_test_task_step
    train_task_step = args.train_task_step

    val_step = 500
    test_step = 2000

    N = args.N
    K = args.K
    L = args.L

    # load data
    train_data, val_data, test_data, label_dict, vocab = loader.load_dataset(args)

    n_way_k_shot = str(N) + '-way-' + str(K) + '-shot'
    n_way_k_shot = 'stable-PROTO-' + n_way_k_shot
    print('Start training ' + n_way_k_shot)

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()

    data_loader = {}
    data_loader['train'] = loader.get_dataloader(args, train_data, label_dict, N, K, L)
    # class_name, support, support_label, query, query_label = next(data_loader['train'])
    data_loader['val'] = loader.get_dataloader(args, val_data, label_dict, N, K, L)
    data_loader['test'] = loader.get_dataloader(args, test_data, label_dict, N, K, L)

    optim_params = [{'params': model.coder.parameters(), 'lr': 5e-5}]
    optim_params.append({'params': model.bilstm.parameters(), 'lr': meta_lr})
    optim_params_fc = [{'params': model.linear.parameters(), 'lr': task_lr}]

    meta_optimizer = AdamW(optim_params, lr=1)
    task_optimizer = AdamW(optim_params_fc, lr=1)

    best_acc, best_step, best_test_acc, best_test_step, best_changed = 0.0, 0, 0.0, 0, False
    iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0

    for it in range(train_iter):
        meta_loss, meta_right = 0.0, 0.0
        model.train()
        class_name, support, support_label, query, query_label = next(data_loader['train'])
        if cuda:
            support_label, query_label = support_label.cuda(), query_label.cuda()

        loss_q, right_q = train_one_batch(args, class_name, support, support_label, query, query_label, model, train_task_step, task_lr, it, task_optimizer)
        meta_loss = meta_loss + loss_q
        meta_right = meta_right + right_q

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        iter_loss = iter_loss + meta_loss
        iter_right = iter_right + meta_right
        iter_sample += 1

        if it % val_step == 0:
            iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0
        if (it + 1) % 100 == 0:
            print(
            '[TRAIN] step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                       100 * iter_right / iter_sample))
        if (it+1) % val_step == 0:
            acc = test_model(args, cuda, data_loader['val'], model, val_iter, val_test_task_step, task_lr)
            print('[EVAL] | accuracy: {0:2.2f}%'.format(acc * 100))
            if acc > best_acc:
                print('Best checkpoint!')
                best_model = copy.deepcopy(model)
                best_acc, best_step, best_changed = acc, (it + 1), True

        if (it+1) % test_step == 0 and best_changed:
            best_changed = False
            test_acc = test_model(args, cuda, data_loader['test'], best_model, test_iter, val_test_task_step, task_lr)
            print('[TEST] | accuracy: {0:2.2f}%'.format(test_acc*100))
            if test_acc > best_test_acc:
                #torch.save(best_model.state_dict(),n_way_k_shot+'.ckpt')
                best_test_acc, best_test_step = test_acc, best_step
            best_acc = 0.0

    print("\n####################\n")
    print('Finish training model! Best acc: ' + str(best_test_acc) + ' at step ' + str(best_test_step))


def test_model(args, cuda, data_loader, model, val_iter, val_test_task_step, task_lr):
    accs = 0.0
    model.eval()
    for it in range(val_iter):
        net = copy.deepcopy(model)
        class_name, support, support_label, query, query_label = next(data_loader)
        if cuda:
            support_label, query_label = support_label.cuda(), query_label.cuda()
        loss, right = train_one_batch(args, class_name, support, support_label, query, query_label, net, val_test_task_step,
                                      task_lr, it)
        accs += right
        if (it + 1) % 500 == 0:
            print('[EVAL/TEST] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it + 1)))

    return accs/val_iter


def main():

    args = parse_args()
    print_args(args)
    setup_seed(args.seed)

    model = Stable_PROTO(args)
    train_model(model, args)


if __name__ == '__main__':
    main()

