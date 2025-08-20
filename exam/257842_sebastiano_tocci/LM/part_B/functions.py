# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import math
import torch
import torch.nn as nn
import os, re, glob
import importlib

# Local device to avoid circular imports and star re-exports
from main import DEVICE as _DEVICE


# This class computes and stores our vocab
# Word to ids and ids to word
class Lang:
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        output = model(sample["source"])
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample["source"])
            loss = eval_criterion(output, sample["target"])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul : (idx + 1) * mul]
                        )
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


# Helper functions to save/load models
def get_last_run_number(model_dir='bin'):
    os.makedirs(model_dir, exist_ok=True)
    files = glob.glob(f'{model_dir}/training_run_*.pt')
    runs = [
        int(m.group(1))
        for f in files
        if (m := re.search(r'training_run_(\d+)\.pt$', os.path.basename(f)))
    ]
    return max(runs, default=0)



# Load model function
def load_model(model_index = 0, model_dir='bin'):
    """Load the model at the given index. If a index is not provided it loads the model with the greatest index"""

    highest_run_index = get_last_run_number(model_dir)
    if highest_run_index == 0:
        return None

    index = model_index
    if model_index == 0:
        index = highest_run_index

    model_path = f'{model_dir}/training_run_{index}.pt'
    meta_path = f'{model_dir}/training_run_{index}.txt'

    if not os.path.exists(meta_path):
        print(f"The provided index {meta_path} does not correspond to an existing model. Please check the available models in ./bin and try again.")
        return None

    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                try:
                    meta[k] = int(v)
                except ValueError:
                    meta[k] = v  # keep non-int metadata (e.g., model_class)

    emb_size = meta.get('emb_size')
    hid_size = meta.get('hid_size')
    vocab_len = meta.get('vocab_len')
    pad_index = meta.get('pad_index', 0)
    if None in (emb_size, hid_size, vocab_len):
        raise RuntimeError('Invalid metadata: emb_size, hid_size, and vocab_len are required.')

    class_name = meta.get('model_class', 'LM_RNN')

    ModelClass = getattr(importlib.import_module('model'), class_name)

    model = ModelClass(emb_size, hid_size, vocab_len, pad_index=pad_index).to(_DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=_DEVICE))
    return model

def save_model(model, emb_size, hid_size, pad_index, vocab_len, model_dir='bin'):
    os.makedirs(model_dir, exist_ok=True)
    run = get_last_run_number(model_dir) + 1
    model_path = f'{model_dir}/training_run_{run}.pt'
    meta_path = f'{model_dir}/training_run_{run}.txt'

    print(f"Saving model {model.__class__.__name__} (emb_size={emb_size}, hid_size={hid_size}, vocab_len={vocab_len}) to {model_path} and metadata to {meta_path}...")

    torch.save(model.state_dict(), model_path)
    with open(meta_path, 'w') as f:
        f.write(f'emb_size={emb_size}\n')
        f.write(f'hid_size={hid_size}\n')
        f.write(f'vocab_len={vocab_len}\n')
        f.write(f'pad_index={pad_index}\n')
        f.write(f'model_class={model.__class__.__name__}\n')


def test_saved_model(index):
    from utils import read_file, PennTreeBank, partial, collate_fn
    from torch.utils.data import DataLoader

    if not index:
        print("Warning: a model index is required to run model evaluation. Exiting...")
        return
    model = load_model(model_index=index)

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    # Wrtite the code to load the datasets and to run your functions

    test_dataset = PennTreeBank(test_raw, lang)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
    )
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )
    final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
    print("Test ppl: ", final_ppl)

    exit(0)
