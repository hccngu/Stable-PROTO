import collections
import json
from collections import defaultdict

import numpy as np
import torch
# from torchtext.vocab import Vocab, Vectors

from dataset.utils import tprint
import torch.utils.data as data
import random
from torch.autograd import Variable


def _get_20newsgroup_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
            'talk politics mideast': 0,
            'sci space': 1,
            'misc forsale': 2,
            'talk politics misc': 3,
            'comp graphics': 4,
            'sci crypt': 5,
            'comp windows x': 6,
            'comp os ms-windows misc': 7,
            'talk politics guns': 8,
            'talk religion misc': 9,
            'rec autos': 10,
            'sci med': 11,
            'comp sys mac hardware': 12,
            'sci electronics': 13,
            'rec sport hockey': 14,
            'alt atheism': 15,
            'rec motorcycles': 16,
            'comp sys ibm pc hardware': 17,
            'rec sport baseball': 18,
            'soc religion christian': 19,
        }

    val_classes = list(range(5))
    train_classes = list(range(5, 13))
    test_classes = list(range(13, 20))

    return train_classes, val_classes, test_classes, label_dict


def _get_amazon_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'Amazon Instant Video': 0,
        'Apps for Android': 1,
        'Automotive': 2,
        'Baby': 3,
        'Beauty': 4,
        'Books': 5,
        'CDs and Vinyl': 6,
        'Cell Phones and Accessories': 7,
        'Clothing Shoes and Jewelry': 8,
        'Digital Music': 9,
        'Electronics': 10,
        'Grocery and Gourmet Food': 11,
        'Health and Personal Care': 12,
        'Home and Kitchen': 13,
        'Kindle Store': 14,
        'Movies and TV': 15,
        'Musical Instruments': 16,
        'Office Products': 17,
        'Patio Lawn and Garden': 18,
        'Pet Supplies': 19,
        'Sports and Outdoors': 20,
        'Tools and Home Improvement': 21,
        'Toys and Games': 22,
        'Video Games': 23
    }

    val_classes = list(range(5))
    test_classes = list(range(5, 14))
    train_classes = list(range(14, 24))

    return train_classes, val_classes, test_classes, label_dict


def _get_rcv1_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = [1, 2, 12, 15, 18, 20, 22, 25, 27, 32, 33, 34, 38, 39,
                     40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                     54, 55, 56, 57, 58, 59, 60, 61, 66]
    val_classes = [5, 24, 26, 28, 29, 31, 35, 23, 67, 36]
    test_classes = [0, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 21, 30, 37,
                    62, 63, 64, 65, 68, 69, 70]

    return train_classes, val_classes, test_classes


def _get_fewrel_classes(args):
    '''
        @return list of classes associated with each split
    '''
    # head=WORK_OF_ART validation/test split
    train_classes = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                     22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                     39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                     59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                     76, 77, 78]

    val_classes = [7, 9, 17, 18, 20]
    test_classes = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]

    return train_classes, val_classes, test_classes


def _get_huffpost_classes(args):
    '''
        @return list of classes associated with each split
    '''

    val_classes = list(range(5))
    train_classes = list(range(5, 25))
    test_classes = list(range(25, 41))

    return train_classes, val_classes, test_classes


def _get_reuters_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(15))
    val_classes = list(range(15, 20))
    test_classes = list(range(20, 31))

    return train_classes, val_classes, test_classes


def _load_json(path):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': int(row['label']),
                'text': row['text'][:500]  # truncate the text to 500 tokens
            }

            text_len.append(len(row['text']))

            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]

            data.append(item)

        tprint('Class balance:')

        print(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data


def _read_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    return words


def _meta_split(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['label'] in train_classes:
            train_data.append(example)
        if example['label'] in val_classes:
            val_data.append(example)
        if example['label'] in test_classes:
            test_data.append(example)

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)


    # compute the max text length
    text_len = np.array([len(e['text']) for e in data])
    max_text_len = max(text_len)

    # initialize the big numpy array by <pad>
    text = vocab.stoi['<pad>'] * np.ones([len(data), max_text_len],
                                     dtype=np.int64)

    del_idx = []
    # convert each token to its corresponding id
    for i in range(len(data)):
        text[i, :len(data[i]['text'])] = [
                vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
                for x in data[i]['text']]

        # filter out document with only unk and pad
        if np.max(text[i]) < 2:
            del_idx.append(i)

    vocab_size = vocab.vectors.size()[0]

    text_len, text, doc_label, raw = _del_by_idx(
            [text_len, text, doc_label, raw], del_idx, 0)

    new_data = {
        'text': text,
        'text_len': text_len,
        'label': doc_label,
        'raw': raw,
        'vocab_size': vocab_size,
    }

    return new_data


def _split_dataset(data, finetune_split):
    """
        split the data into train and val (maintain the balance between classes)
        @return data_train, data_val
    """

    # separate train and val data
    # used for fine tune
    data_train, data_val = defaultdict(list), defaultdict(list)

    # sort each matrix by ascending label order for each searching
    idx = np.argsort(data['label'], kind="stable")

    non_idx_keys = ['vocab_size', 'classes2id', 'is_train', 'n_t', 'n_d', 'avg_ebd']
    for k, v in data.items():
        if k not in non_idx_keys:
            data[k] = v[idx]

    # loop through classes in ascending order
    classes, counts = np.unique(data['label'], return_counts=True)
    start = 0
    for label, n in zip(classes, counts):
        mid = start + int(finetune_split * n)  # split between train/val
        end = start + n  # split between this/next class

        for k, v in data.items():
            if k not in non_idx_keys:
                data_train[k].append(v[start:mid])
                data_val[k].append(v[mid:end])

        start = end  # advance to next class

    # convert back to np arrays
    for k, v in data.items():
        if k not in non_idx_keys:
            data_train[k] = np.concatenate(data_train[k], axis=0)
            data_val[k] = np.concatenate(data_val[k], axis=0)

    return data_train, data_val


def load_dataset(args):
    if args.dataset == '20newsgroup':
        train_classes, val_classes, test_classes, label_dict = _get_20newsgroup_classes(args)
    elif args.dataset == 'amazon':
        train_classes, val_classes, test_classes, label_dict = _get_amazon_classes(args)
    elif args.dataset == 'fewrel':
        train_classes, val_classes, test_classes = _get_fewrel_classes(args)
    elif args.dataset == 'huffpost':
        train_classes, val_classes, test_classes = _get_huffpost_classes(args)
    elif args.dataset == 'reuters':
        train_classes, val_classes, test_classes = _get_reuters_classes(args)
    elif args.dataset == 'rcv1':
        train_classes, val_classes, test_classes = _get_rcv1_classes(args)
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[20newsgroup, amazon, fewrel, huffpost, reuters, rcv1]')

    assert(len(train_classes) == args.n_train_class)
    assert(len(val_classes) == args.n_val_class)
    assert(len(test_classes) == args.n_test_class)

    tprint('Loading data')
    all_data = _load_json(args.data_path)

    tprint('Loading word vectors')

    if args.bert is not True:

        vectors = Vectors(args.word_vector, cache=args.wv_path)
        vocab = Vocab(collections.Counter(_read_words(all_data)), vectors=vectors,
                      specials=['<pad>', '<unk>'], min_freq=5)

        # print word embedding statistics
        wv_size = vocab.vectors.size()
        tprint('Total num. of words: {}, word vector dimension: {}'.format(
            wv_size[0],
            wv_size[1]))

        num_oov = wv_size[0] - torch.nonzero(
                torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
        tprint(('Num. of out-of-vocabulary words'
               '(they are initialized to zeros): {}').format( num_oov))

        # Split into meta-train, meta-val, meta-test data
        train_data, val_data, test_data = _meta_split(
                all_data, train_classes, val_classes, test_classes)
        tprint('#train {}, #val {}, #test {}'.format(
            len(train_data), len(val_data), len(test_data)))

        # Convert everything into np array for fast data loading
        train_data = _data_to_nparray(train_data, vocab, args)
        val_data = _data_to_nparray(val_data, vocab, args)
        test_data = _data_to_nparray(test_data, vocab, args)

        train_data['is_train'] = True
        # this tag is used for distinguishing train/val/test when creating source pool

        return train_data, val_data, test_data, label_dict, vocab

    else:
        train_data, val_data, test_data = _meta_split(
            all_data, train_classes, val_classes, test_classes)
        tprint('#train {}, #val {}, #test {}'.format(
            len(train_data), len(val_data), len(test_data)))
        vocab = None

        train_data, val_data, test_data = to_dict(train_data, val_data, test_data)

        return train_data, val_data, test_data, label_dict, vocab


def to_dict(train_data, val_data, test_data):

    train_dict, val_dict, test_dict = {}, {}, {}

    classes = []
    for train_d in train_data:
        classes.append(train_d['label'])
    classes = set(classes)
    for key in classes:
        train_dict[key] = []
    for train_d in train_data:
        train_dict[train_d['label']].append(train_d['text'])

    classes = []
    for val_d in val_data:
        classes.append(val_d['label'])
    classes = set(classes)
    for key in classes:
        val_dict[key] = []
    for val_d in val_data:
        val_dict[val_d['label']].append(val_d['text'])

    classes = []
    for test_d in test_data:
        classes.append(test_d['label'])
    classes = set(classes)
    for key in classes:
        test_dict[key] = []
    for test_d in test_data:
        test_dict[test_d['label']].append(test_d['text'])

    return train_dict, val_dict, test_dict




def get_dataloader(args, train_data, label_dict, N, K, L):
    data_loader = data.DataLoader(
        dataset=Data_Process(args, train_data, label_dict, N, K, L),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    return iter(data_loader)


class Data_Process(data.Dataset):
    def __init__(self, args, train_data, label_dict, N, K, L):
        super(Data_Process, self).__init__()

        self.data = train_data
        self.label_dict = label_dict
        self.N, self.K, self.L = N, K, L
        self.classes = list(self.data.keys())

    def __len__(self):
        return 1000000000

    def __getitem__(self, index):
        label_dict = self.label_dict
        N, K, L = self.N, self.K, self.L
        class_name = random.sample(self.classes, N)
        support, support_label, query, query_label = [], [], [], []
        for i, name in enumerate(class_name):
            rel = self.data[name]
            samples = random.sample(rel, K+L)
            for j in range(K):
                support.append([samples[j], i])
            for j in range(K, K+L):
                query.append([samples[j], i])

        # support=random.sample(support,N*K)
        query = random.sample(query, N*L)  # 这里相当于一个shuffle
        for i in range(N*K):
            support_label.append(support[i][1])
            support[i] = support[i][0]

        for i in range(N*L):
            query_label.append(query[i][1])
            query[i] = query[i][0]
        support_label = Variable(torch.from_numpy(np.stack(support_label, 0).astype(np.int64)).long())
        query_label = Variable(torch.from_numpy(np.stack(query_label, 0).astype(np.int64)).long())
        #if torch.cuda.is_available():support_label,query_label=support_label.cuda(),query_label.cuda()

        class_name_temp = []
        for cn in class_name:
            for ld in label_dict:
                t = label_dict[ld]
                if t == cn:
                    class_name_temp.append(ld.split())

        return class_name_temp, support, support_label, query, query_label