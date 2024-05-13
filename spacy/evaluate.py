import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from torch.nn.utils.rnn import pad_sequence
from model import Transformer
from tqdm import tqdm
from torchdata.datapipes.iter import IterableWrapper
import numpy as np
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from dataset import pre_process, configs, en_tokenizer, de_tokenizer,multi_train
smoothie = SmoothingFunction()
src_test_path='/content/train.en'
trg_testfile_path='/content/train.de'
with open(src_test_path, "r") as file3:
 source_test_array=[]
 for i in range(5):
     source_testline=file3.readline()
     source_test_array.append(source_testline)

with open(trg_testfile_path, "r") as file4:
    target_testsentence_array=[]
    for e in range(5):
     target_test_line=file4.readline()
     target_testsentence_array.append(target_test_line)

pred_sents=[]
en_vocab = build_vocab_from_iterator(map(en_tokenizer, [english for english, _ in multi_train]), min_freq=2, specials=["<unk>", "<sos>", "<eos>", "<pad>"])
de_vocab = build_vocab_from_iterator(map(de_tokenizer, [de for _ , de in multi_train]), min_freq=2, specials=["<unk>", "<sos>", "<eos>", "<pad>"])
en_token2id = en_vocab.get_stoi()
de_token2id = de_vocab.get_stoi()
en_id2token = en_vocab.get_itos()
de_id2token = de_vocab.get_itos()

def src_encode(src_text):
  source_sentence = [ en_token2id.get(token, en_token2id['<unk>']) for token in en_tokenizer(src_text) ]
  return source_sentence

def tgt_encode(tgt_text):
        target_sentence = [de_token2id['<sos>']] \
        + [ de_token2id.get(token, de_token2id['<unk>']) for token in de_tokenizer(tgt_text) ] \
        + [de_token2id['<eos>']]
        return target_sentence


def tgt_decode(ids):
        sentence = list(map(lambda x: de_id2token[x], ids))
        return " ".join(sentence)

def translate_sentence(src_sentence,trg_sentence):

    device=torch.device("cpu")

    model = Transformer(
            source_vocab_size=6191,
            target_vocab_size=8014,
            embedding_dim=configs["embedding_dim"],
            source_max_seq_len=configs["source_max_seq_len"],
            target_max_seq_len=configs["target_max_seq_len"],
            num_layers=configs["n_layers"],
            num_heads=configs["n_heads"],
            dropout=configs["dropout"]
              )
    # Convert to Tensor

    sentence=en_tokenizer(src_sentence)[:256]
    s=[]
    for word in sentence:
     sentence_tokens=src_encode(word)

     for item in sentence_tokens:
      s.append(item)

    def pad_features(source, pad_id, seq_length):
      src_padding = np.full(seq_length-len(source), pad_id, dtype=int)
      src_features= source.extend(src_padding)
      return source
    padded=pad_features(s,pre_process.pad_token_id,256)

    decoder_input= torch.Tensor([pre_process.sos_token_id]).long().unsqueeze(0).to(device)

    src_sentence_tensor = torch.LongTensor(padded).unsqueeze(0).to(device)
    source_mask= model.make_source_mask(src_sentence_tensor, pre_process.pad_token_id).to(device)
    encoder_output = model.encoder.forward(src_sentence_tensor, source_mask)
    target_mask = model.make_target_mask(decoder_input).to(device)
    prediction=[]


    #one prediction of one sentence
    for _ in range(256):
     # Decoder forward pass
     pred = model.decoder.forward(decoder_input, encoder_output, source_mask, target_mask)
     pred = F.softmax(model.final_linear(pred), dim=-1)
     pred = pred[:, -1, :].view(-1)
     token_index = torch.argmax(pred, keepdim=True)
     prediction.append(token_index.item())
     token_index=token_index.unsqueeze(0)
     decoder_input=torch.cat([decoder_input,token_index],dim=1)
    print(prediction)
    if token_index.item()==pre_process.eos_token_id:
      print("EOS")
    sentence=tgt_decode(prediction)
    print(sentence)
    return sentence
for source_sentence, target_sentence in tqdm(zip(source_test_array,target_testsentence_array)):
 output = translate_sentence(source_sentence, target_sentence)
 pred_sents.append(output)
print(len(pred_sents))

def calculate_bleu_score(references,hypotheses):


    bleu_score = corpus_bleu(references, hypotheses)
    #weights = [(0.5, 0.5),(0.333, 0.333, 0.334),(0.25, 0.25, 0.25, 0.25)]
    #bleu_2 = corpus_bleu(references, hypotheses, weights=weights[0])
    #bleu_3 = corpus_bleu(references, hypotheses, weights=weights[1])
    #bleu_4 = corpus_bleu(references, hypotheses, weights=weights[2])
    #print(f"BLEU-2: {bleu_2} | BLEU-3: {bleu_3} | BLEU-4: {bleu_4}")
    #return {"bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}
    return bleu_score

for a, b in tqdm(zip(pred_sents,target_testsentence_array)):
 bleu_output = calculate_bleu_score(a, b)
