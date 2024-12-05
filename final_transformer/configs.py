import torch
import json
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import math

configs = {
    "train_source_data":"train.en",
    "train_target_data":"train.de",
    "valid_source_data":"val.en",
    "valid_target_data":"val.de",
    "test_source_data": "trimmed_test.en",
    "test_target_data": "trimmed_test.de",
    "source_max_seq_len":256,
    "target_max_seq_len":256,
    "batch_size":50,
    "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "embedding_dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "dropout": 0.15,
    "lr":math.exp(-8),
    "n_epochs":100,
    "print_freq": 5,
    "beam_size":3,
    "model_path":"/content/model_transformer_translate_ende.pt",
    "early_stopping":5
}
