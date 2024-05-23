import torch
from torch.utils.data import Dataset

from utils import train_features, en_tokenizer, de_tokenizer

class TrainTranslateDataset(Dataset):

    def __init__(self, source_tokenizer, target_tokenizer, source_max_seq_len=256, target_max_seq_len=256):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.sentences=train_features



    def __len__(self):
        return len(self.sentences)

    # create decoder input mask
    def create_decoder_mask(self, seq_len):
        mask = torch.ones(seq_len, seq_len).tril()
        return mask

    def __getitem__(self, index):

            tensor_source=torch.tensor(self.sentences[index][0])
            tensor_target=torch.tensor(self.sentences[index][1])
            return {
                "source_ids": tensor_source,
                "target_ids": tensor_target
            }


a=TrainTranslateDataset(source_tokenizer=en_tokenizer, target_tokenizer=de_tokenizer, source_max_seq_len=256, target_max_seq_len=256)
print(a[0])
print(len(a[0]["source_ids"]))

class TrainTranslateDataset(Dataset):

    def __init__(self, source_tokenizer, target_tokenizer, source_max_seq_len=256, target_max_seq_len=256):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.sentences=train_features



    def __len__(self):
        return len(self.sentences)

    # create decoder input mask
    def create_decoder_mask(self, seq_len):
        mask = torch.ones(seq_len, seq_len).tril()
        return mask

    def __getitem__(self, index):

            tensor_source=torch.tensor(self.sentences[index][0])
            tensor_target=torch.tensor(self.sentences[index][1])
            return {
                "source_ids": tensor_source,
                "target_ids": tensor_target
            }


a=TrainTranslateDataset(source_tokenizer=en_tokenizer, target_tokenizer=de_tokenizer, source_max_seq_len=256, target_max_seq_len=256)
print(a[0])
print(len(a[0]["source_ids"]))