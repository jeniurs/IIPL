
import torch
import json
import os
from torch.utils.data import Dataset
import re
import numpy as np
import matplotlib.pyplot as plt

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from functools import partial

from configs import configs
def read_data(source_file, target_file):
        source_data = open(source_file, encoding='iso-8859-1').read().strip().split("\n")
        target_data = open(target_file, encoding='iso-8859-1').read().strip().split("\n")
        return source_data, target_data

train_src_data, train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])


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

en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
de_tokenizer = get_tokenizer(tokenizer='spacy', language='de_core_news_sm')

def preprocess_seq(seq):
        seq = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(seq))
        seq = re.sub(r"[ ]+", " ", seq)
        seq = re.sub(r"\!+", "!", seq)
        seq = re.sub(r"\,+", ",", seq)
        seq = re.sub(r"\?+", "?", seq)
        seq = seq.lower()
        return seq

preprocessed_srclist=[]
for english_sentences in train_src_data:
 preprocessed_src=preprocess_seq(english_sentences)
 preprocessed_srclist.append(preprocessed_src)

preprocessed_tgtlist=[]
for german_sentences in train_trg_data:
 preprocessed_tgt=preprocess_seq(german_sentences)
 preprocessed_tgtlist.append(preprocessed_tgt)

val_preprocessed_srclist=[]
for english_sentences in valid_src_data:
 preprocessed_src=preprocess_seq(english_sentences)
 val_preprocessed_srclist.append(preprocessed_src)

val_preprocessed_tgtlist=[]
for german_sentences in valid_trg_data:
 preprocessed_tgt=preprocess_seq(german_sentences)
 val_preprocessed_tgtlist.append(preprocessed_tgt)

en_vocab = build_vocab_from_iterator(map(en_tokenizer, [english for english in preprocessed_srclist]), min_freq=2, specials=["<unk>", "<sos>", "<eos>", "<pad>"])
de_vocab = build_vocab_from_iterator(map(de_tokenizer, [de for de in preprocessed_tgtlist]), min_freq=2, specials=["<unk>", "<sos>", "<eos>", "<pad>"])

en_token2id = en_vocab.get_stoi()
de_token2id = de_vocab.get_stoi()
en_id2token = en_vocab.get_itos()
de_id2token = de_vocab.get_itos()

unk_token_id = 0
sos_token_id = 1
eos_token_id = 2
pad_token_id = 3

def src_encode(src_text):
        source_sentence = [ en_token2id.get(token, en_token2id['<unk>']) for token in en_tokenizer(src_text) ]
        return source_sentence

def tgt_encode(tgt_text):
        target_sentence = [de_token2id['<sos>']] \
        + [ de_token2id.get(token, de_token2id['<unk>']) for token in de_tokenizer(tgt_text) ] \
        + [ de_token2id['<eos>']]
        return target_sentence


def src_decode(ids):
     sentences = []
     sentence = list(map(lambda x: en_id2token[x], ids))
     sentences.append(" ".join(sentence))
     return sentences

train_src_sentences = [ src_encode(eng) for eng in preprocessed_srclist if len(eng) > 0]
train_tgt_sentences = [ tgt_encode(ger) for ger in preprocessed_tgtlist if len(ger) > 0]
val_src_sentences = [ src_encode(eng) for eng in val_preprocessed_srclist if len(eng) > 0]
val_tgt_sentences = [ tgt_encode(ger) for ger in val_preprocessed_tgtlist if len(ger) > 0]
train_sentences = list(zip(train_src_sentences, train_tgt_sentences))
val_sentences = list(zip(val_src_sentences, val_tgt_sentences))
def pad_features(pad_id, s):
     for i in range(len(s)):
      src_padding = np.full(256-len(s[i][0]), pad_id, dtype=int)
      src_features= s[i][0].extend(src_padding)
      tgt_padding = np.full(256-len(s[i][1]), pad_id, dtype=int)
      tgt_features=s[i][1].extend(tgt_padding)
     return s

train_features=pad_features(pad_token_id,train_sentences)
val_features=pad_features(pad_token_id,val_sentences)