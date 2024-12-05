import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import re
import json
from tqdm import tqdm
from torchtext.data.metrics import bleu_score


from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction()

def tgt_decode(ids):
        sentence = list(map(lambda x: de_id2token[x], ids))[1:-1]
        return " ".join(sentence)


def load_model_tokenizer(configs):
    """
    This function will load model and tokenizer from pretrained model and tokenizer
    """
    device = torch.device(configs["device"])
    source_tokenizer = en_tokenizer
    target_tokenizer = de_tokenizer

    # Load model Transformer
    model = Transformer(
        source_vocab_size=10000,
        target_vocab_size=10000,
        embedding_dim=configs["embedding_dim"],
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        num_layers=configs["n_layers"],
        num_heads=configs["n_heads"],
        dropout=configs["dropout"]
    )
    model.load_state_dict(torch.load(configs["model_path"],map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
    print(f"Done load model on the {device} device")
    return model, source_tokenizer, target_tokenizer

def test_pad_features(pad_id, s):
     for i in range(len(s)):
      src_padding = np.full(256-len(s), pad_id, dtype=int)
      src_features= s.extend(src_padding)
     return s

test_source_data, test_target_data = read_data(configs["test_source_data"], configs["test_target_data"])


def translate(model, sentence, source_tokenizer, target_tokenizer, source_max_seq_len=256,
    target_max_seq_len=256, beam_size=3, device=torch.device("cpu"), print_process=False):
    """
    This funciton will translate give a source sentence and return target sentence using beam search
    """
    source_tensor = torch.tensor(sentence).unsqueeze(0).to(device)
    # Create source sentence mask
    source_mask = model.make_source_mask(source_tensor, pad_token_id).to(device)
    # Feed forward Encoder
    encoder_output = model.encoder.forward(source_tensor, source_mask)
    # Initialize beam list
    beams = [([sos_token_id], 0)]
    completed = []

    # Start decoding
    for _ in range(target_max_seq_len):
        new_beams = []
        for beam in beams:
            # Get input token
            input_token = torch.tensor([beam[0]]).to(device)
            # Create mask
            target_mask = model.make_target_mask(input_token).to(device)
            # Decoder forward pass
            pred = model.decoder.forward(input_token, encoder_output, source_mask, target_mask)
            # Forward to linear classify token in vocab and Softmax
            pred = F.softmax(model.final_linear(pred), dim=-1)
            # Get tail predict token
            pred = pred[:, -1, :].view(-1)
            # Get top k tokens
            top_k_scores, top_k_tokens = pred.topk(beam_size)
            # Update beams
            for i in range(beam_size):
                new_beams.append((beam[0] + [top_k_tokens[i].item()], beam[1] + top_k_scores[i].item()))

        import copy
        beams = copy.deepcopy(new_beams)
        # Sort beams by score
        beams = sorted(beams, key=lambda x: x[1], reverse=True)[:beam_size]
        # Add completed beams to completed list and reduce beam size
        for beam in beams:
            if beam[0][-1] == eos_token_id:
                completed.append(beam)
                beams.remove(beam)
                beam_size -= 1

        if beam_size == 0:
            break


    # Sort the completed beams
    completed.sort(key=lambda x: x[1], reverse=True)
    # Get target sentence tokens
    target_tokens = completed[0][0]
    # Convert target sentence from tokens to string
    target_sentence = tgt_decode(target_tokens)

    return target_sentence


def preprocess_seq(seq):
        seq = re.sub(
        r"[\*\"“”\n\…\+\-\/\=‘•:
\|’\!;]", " ", str(seq))
        seq = re.sub(r"[ ]+", " ", seq)
        seq = re.sub(r"\!+", "!", seq)
        seq = re.sub(r"\,+", ",", seq)
        seq = re.sub(r"\?+", "?", seq)
        seq = seq.lower()
        return seq

def calculate_bleu_score(model, source_tokenizer, target_tokenizer, configs):
    device = torch.device(configs["device"])
    valid_src_data, valid_trg_data = read_data(configs["test_source_data"], configs["test_target_data"])

    pred_sents = []

    hypotheses=[]
    for sentence in tqdm(valid_src_data):
      preprocessed=preprocess_seq(sentence)
      pred_sents.append(preprocessed)
      for en in pred_sents:
        source_before_padding=src_encode(en)
      source_after_padding=test_pad_features(pad_token_id,source_before_padding)

      pred_trg = translate(model, source_after_padding, source_tokenizer, target_tokenizer, configs["source_max_seq_len"], configs["target_max_seq_len"], configs["beam_size"], device)
      hypotheses.append(pred_trg)

    references=[[preprocess_seq(sent).split()] for sent in valid_trg_data]
    hypothesis=[sent.split() for sent in hypotheses]
    print(references)
    print(hypothesis)
    bleu_score=corpus_bleu(references, hypothesis, weights=(0.25,0.25,0.25,0.25))
    print(bleu_score)



def main():
    model, source_tokenizer, target_tokenizer = load_model_tokenizer(configs)
    bleus = calculate_bleu_score(model, source_tokenizer, target_tokenizer, configs)

if __name__ == "__main__":
    main()
