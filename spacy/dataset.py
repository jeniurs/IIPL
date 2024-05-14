import torch
import json
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torchtext.datasets import Multi30k
from torchdata.datapipes.iter import IterableWrapper
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader


import torch
from torch.nn.utils.rnn import pad_sequence

import random
from functools import partial

configs = {
    "source_max_seq_len":256,
    "target_max_seq_len":256,
    "batch_size":50,
    "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "embedding_dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "dropout": 0.25,
    "lr":0.0001,
    "n_epochs":500,
    "print_freq": 5,
    "model_path":"/content/model_transformer_translate_en_vi.pt",
    "early_stopping":5
}

# visualize log
def plot_loss(log_path, log_dir):
    log = json.load(open(log_path, "r"))

    plt.figure()
    plt.plot(log["train_loss"], label="train loss")
    plt.plot(log["valid_loss"], label="valid loss")
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_epoch.png"))

    # plot batch loss
    plt.figure()
    lst = log["train_batch_loss"]
    n = int(len(log["train_batch_loss"]) / len(log["valid_batch_loss"]))
    train_batch_loss = [lst[i:i + n][0] for i in range(0, len(lst), n)]
    plt.plot(train_batch_loss, label="train loss")
    plt.plot(log["valid_batch_loss"], label="valid loss")
    plt.title("Loss per batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_batch.png"))


multi_train, multi_valid, multi_test = Multi30k(language_pair=('en','de'))

en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
de_tokenizer = get_tokenizer(tokenizer='spacy', language='de_core_news_sm')

en_vocab = build_vocab_from_iterator(map(en_tokenizer, [english for english, _ in multi_train]), min_freq=2, specials=["<unk>", "<sos>", "<eos>", "<pad>"])
de_vocab = build_vocab_from_iterator(map(de_tokenizer, [de for _ , de in multi_train]), min_freq=2, specials=["<unk>", "<sos>", "<eos>", "<pad>"])

multi_train, multi_valid, multi_test = Multi30k(language_pair=('en','de'))
en_token2id = en_vocab.get_stoi()
de_token2id = de_vocab.get_stoi()
en_id2token = en_vocab.get_itos()
de_id2token = de_vocab.get_itos()

class Language:
    unk_token_id = 0
    sos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3

    def __init__(self, src_tokenizer, tgt_tokenizer, src_token2id, tgt_token2id, src_id2token, tgt_id2token):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_token2id = src_token2id
        self.tgt_token2id = tgt_token2id

        self.src_id2token = src_id2token
        self.tgt_id2token = tgt_id2token

    def src_encode(self, src_text):
        source_sentence = [ self.src_token2id.get(token, self.src_token2id['<unk>']) for token in self.src_tokenizer(src_text) ]
        return source_sentence

    def tgt_encode(self, tgt_text):
        target_sentence = [self.tgt_token2id['<sos>']] \
        + [ self.tgt_token2id.get(token, self.tgt_token2id['<unk>']) for token in self.tgt_tokenizer(tgt_text) ] \
        + [self.tgt_token2id['<eos>']]
        return target_sentence

    def src_decode(self, ids):
     sentences = []
     for batch in ids:
        sentence = list(map(lambda x: self.src_id2token[x], batch))
        sentences.append(" ".join(sentence))
     return sentences

    def tgt_decode(self, ids):
      sentences = []
      for batch in ids:
        sentence = list(map(lambda x: self.tgt_id2token[x], batch))
        sentences.append(" ".join(sentence))
      return sentences

pre_process = Language(en_tokenizer, de_tokenizer, en_token2id, de_token2id, en_id2token, de_id2token)
en_test, de_test = next(iter(multi_train))
en_encoded = pre_process.src_encode(en_test)
de_encoded = pre_process.tgt_encode(de_test)

class MultiDataset(Dataset):

    features=[]
    def __init__(self, data, language):
        self.data = data
        self.language = language
        self.sentences = self.preprocess()
        self.features= self.pad_features(pad_id=pre_process.pad_token_id,seq_length=256)


    def preprocess(self):
        # dataset 안에 길이가 0인 문장이 존재한다.
        sentences = [ (self.language.src_encode(eng), self.language.tgt_encode(de)) for eng, de in self.data if len(eng) > 0 and len(de) > 0]
        return sentences


    def pad_features(self, pad_id, seq_length):
     for i in range(len(self.sentences)):
      src_padding = np.full(seq_length-len(self.sentences[i][0]), pad_id, dtype=int)
      src_features= self.sentences[i][0].extend(src_padding)
      tgt_padding = np.full(seq_length-len(self.sentences[i][1]), pad_id, dtype=int)
      tgt_features=self.sentences[i][1].extend(tgt_padding)


    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)
language = Language(en_tokenizer, de_tokenizer, en_token2id, de_token2id, en_id2token, de_id2token)
multi_train_dataset = MultiDataset(multi_train, language)
multi_val_dataset = MultiDataset(multi_valid, language)

def collate_fn(batch_samples):
    pad_token_id = Language.pad_token_id

    src_sentences = pad_sequence([torch.tensor(src) for src, _ in batch_samples], batch_first=True, padding_value=pad_token_id)
    tgt_sentences = pad_sequence([torch.tensor(tgt) for _, tgt in batch_samples], batch_first=True, padding_value=pad_token_id)

    return src_sentences, tgt_sentences

def batch_sampling(sequence_lengths, batch_size):
    '''
    sequence_length: (source 길이, target 길이)가 담긴 리스트이다.
    batch_size: batch 크기
    '''

    seq_lens = [(i, seq_len, tgt_len) for i,(seq_len, tgt_len) in enumerate(sequence_lengths)]
    seq_lens = sorted(seq_lens, key=lambda x: x[1])
    seq_lens = [sample[0] for sample in seq_lens]
    sample_indices = [ seq_lens[i:i+batch_size] for i in range(0,len(seq_lens), batch_size)]

    random.shuffle(sample_indices) # 모델이 길이에 편향되지 않도록 섞는다.

    return sample_indices

batch_size=50

sequence_lengths = list(map(lambda x: (len(x[0]), len(x[1])), multi_train_dataset))
sequence_val_lengths=list(map(lambda x: (len(x[0]), len(x[1])), multi_val_dataset))
batch_train_sampler = batch_sampling(sequence_lengths, batch_size)
batch_valid_sampler=batch_sampling(sequence_val_lengths,batch_size)

train_loaders = DataLoader(multi_train_dataset, collate_fn=collate_fn, batch_sampler=batch_train_sampler)
valid_loaders = DataLoader(multi_val_dataset, collate_fn=collate_fn, batch_sampler=batch_valid_sampler)
