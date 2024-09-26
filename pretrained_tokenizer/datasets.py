import torch
import json
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re

configs = {
    "train_source_data":"/content/train.en",
    "train_target_data":"/content/train.de",
    "valid_source_data":"/content/val.en",
    "valid_target_data":"/content/val.de",
    "test_source_data": "/content/test_2016_flickr.en",
    "test_target_data": "/content/test_2016_flickr.de",
    "source_tokenizer":"bert-base-uncased",
    "target_tokenizer":"dbmdz/bert-base-german-cased",
    "source_max_seq_len":256,
    "target_max_seq_len":256,
    "batch_size":20,
    "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "embedding_dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "dropout": 0.1,
    "lr":0.0001,
    "n_epochs":12,
    "print_freq": 5,
    "beam_size":3,
    "model_path":"./model_transformer_translate_en_de.pt",
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


class TranslateDataset(Dataset):
    def __init__(self, source_tokenizer, target_tokenizer, source_data=None, target_data=None, source_max_seq_len=256, target_max_seq_len=256, phase="train"):
        self.source_data = source_data
        self.target_data = target_data
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.phase = phase

    def preprocess_seq(self, seq):
        seq = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(seq))
        seq = re.sub(r"[ ]+", " ", seq)
        seq = re.sub(r"\!+", "!", seq)
        seq = re.sub(r"\,+", ",", seq)
        seq = re.sub(r"\?+", "?", seq)
        seq = seq.lower()
        return seq

    def convert_line_uncased(self, tokenizer, text, max_seq_len):
        tokens = tokenizer.tokenize(text)[:max_seq_len-2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        tokens += [tokenizer.pad_token]*(max_seq_len - len(tokens))
        token_idx = tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_idx

    def __len__(self):
        return len(self.source_data)

    # create decoder input mask
    def create_decoder_mask(self, seq_len):
        mask = torch.ones(seq_len, seq_len).tril()
        return mask

    def __getitem__(self, index):
        # source_seq, source_idx = self.convert_line_uncased(
        #     tokenizer=self.source_tokenizer,
        #     text=self.preprocess_seq(self.source_data[index]),
        #     max_seq_len=self.source_max_seq_len
        # )
        # target_seq, target_idx = self.convert_line_uncased(
        #     tokenizer=self.target_tokenizer,
        #     text=self.preprocess_seq(self.target_data[index]),
        #     max_seq_len=self.target_max_seq_len
        # )

        source = self.source_tokenizer(
                text=self.preprocess_seq(self.source_data[index]),
                padding="max_length",
                max_length=self.source_max_seq_len,
                truncation=True,
                return_tensors="pt"
            )

        if self.phase == "train":
            target = self.target_tokenizer(
                text=self.preprocess_seq(self.target_data[index]),
                padding="max_length",
                max_length=self.target_max_seq_len,
                truncation=True,
                return_tensors="pt"
            )

            return {
                "source_seq": self.source_data[index],
                "source_ids": source["input_ids"][0],
                "target_seq": self.target_data[index],
                "target_ids": target["input_ids"][0],
                }
        else:
            return {
                "source_seq": self.source_data[index],
                "source_ids": source["input_ids"][0],
            }


def main():

    def read_data(source_file, target_file):
        source_data = open(source_file).read().strip().split("\n")
        target_data = open(target_file).read().strip().split("\n")
        return source_data, target_data

    train_src_data, train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])

    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])
    train_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_data=train_src_data,
        target_data=train_trg_data,
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True
    )
    for batch in train_loader:
        print(batch["source_seq"])
        print(batch["target_seq"])
        print(batch["source_ids"])
        print(batch["target_ids"])
        break


if __name__ == "__main__":
    main()

def main():


    def read_data(source_file, target_file):
        source_data = open(source_file).read().strip().split("\n")
        target_data = open(target_file).read().strip().split("\n")
        return source_data, target_data

    train_src_data, train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])

    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])
    train_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_data=train_src_data,
        target_data=train_trg_data,
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True
    )
    for batch in train_loader:
        print(batch["source_seq"])
        print(batch["target_seq"])
        print(batch["source_ids"])
        print(batch["target_ids"])
        break



if __name__ == "__main__":
    main()
