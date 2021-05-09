import os
import json
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from my_transformers.transformers import BertConfig,BertModel,BertTokenizer


class Stable_PROTO(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.N, self.K, self.L = args.N, args.K, args.L
        self.max_length = args.max_length
        self.coder = BERT(args.N, args.max_length)
        self.hidden_size = 128
        self.bilstm = nn.LSTM(input_size=768, hidden_size=self.hidden_size, num_layers=1,
                              bidirectional=True, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(args.max_length, 1)
        self.mlp = MLP(args.N, self.hidden_size*2)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, inputs):  # inputs: [N*K, max_length, 768]

        ebd, (hn, cn) = self.bilstm(inputs)  # -> [N*K, max_length, 128]
        outputs = self.linear(ebd.transpose(1, 2).contiguous()).squeeze(-1)  # -> [N*K, 128]

        return outputs

    def mlp2(self, inputs, params):  # inputs: [5, 256], params: [256, 256]

        out = F.linear(inputs, params, bias=None)

        return out

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


class MLP(nn.Module):
    def __init__(self, N, hs):
        super(MLP, self).__init__()
        self.n_way = N
        self.embedding_dim = 50

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(N, hs)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):  # [N,hidden_size]

        params = self.dropout(self.fc1(inputs.transpose(0, 1).contiguous()))  # [hidden_size, N] -> [hidden_size, hidden_size]
        params = F.normalize(params, dim=-1)  # [hidden_size, hidden_size]

        return params