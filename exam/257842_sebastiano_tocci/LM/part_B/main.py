# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import everything from functions.py file
from functions import *
from utils import *
from model import LM_LSTM_DROPOUT_WEIGHT_TYING, LM_LSTM_VAR_DROPOUT




if __name__ == "__main__":

    # test_saved_model(2)

    download_file(
        "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt"
    )
    download_file(
        "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt"
    )
    download_file(
        "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt"
    )

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    # Wrtite the code to load the datasets and to run your functions
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
    )

    # Experiment also with a smaller or bigger model by changing hid and emb sizes
    # A large model tends to overfit
    hid_size = 200
    emb_size = 300


    lr = 0.0005
    weight_decay = 0.15
    betas=(0.9, 0.999)
    clip = 5  # Clip the gradient
    out_dropout = 0.4
    emb_dropout = 0.4
    n_layers = 2

    train_with_NTAvSDG = True


    # Don't forget to experiment with a lower training batch size
    # Increasing the back propagation steps can be seen as a regularization step


    vocab_len = len(lang.word2id)

    model = LM_LSTM_VAR_DROPOUT(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], out_dropout=out_dropout, emb_dropout=emb_dropout, n_layers=n_layers).to(
        DEVICE
    )
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, n_epochs))

    if train_with_NTAvSDG:
        lr = 1
        # in the original paper L is set as the iterations per epoch, but since when we run the "train_loop" we are in fact performing all the iterations in an epoch,
        # we have to set it to 1, such that k % L == 0 is true after each train_loop call. We could also remove the check entirely, but for the sake of clarity it is preserved.
        L = 1
        print("Setting L to: ", L)
        n = 5
        ## End of Hyperparams
        k = 0
        t = 0
        T = 0
        logs = []
        models = []
        opt = optim.SGD(params=model.parameters(), lr=lr)
        for _ in pbar:
            train_loop(train_loader, opt, criterion_train, model, clip)
            # Keep independent snapshots; state_dict returns references otherwise
            models.append(copy.deepcopy(model.state_dict()))
            if k % L == 0 and T == 0:
                # losses_train.append(np.asarray(loss).mean())
                v, _ = eval_loop(dev_loader, criterion_eval, model)
                pbar.set_description("PPL: %f" % v)
                # When we have more than n evals, compare with the best before the last n
                if t > n and (v > min(logs[: t - n])):
                    T = k

                logs.append(v)
                t+=1
            k+=1
        averaged_model = dict()
        # Just take the keys of any of the models
        # Average parameters from trigger step T (or last step if T is too late)
        start = min(T, max(0, k - 1))
        for key in models[0]:
            stacked = torch.stack([models[i][key] for i in range(start, k)], dim=0)
            averaged_model[key] = stacked.mean(dim=0)
        # Update the weights with the new average!
        model.load_state_dict(averaged_model)
        best_model = model
        best_model.to(DEVICE)
    else:
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if ppl_dev < best_ppl:  # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to("cpu")
                    patience = 3
                else:
                    patience -= 1

                if patience <= 0:  # Early stopping with patience
                    break  # Not nice but it keeps the code clean

        if best_model:
            best_model.to(DEVICE)

    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print("Test ppl: ", final_ppl)

    save_model(best_model, emb_size, hid_size, lang.word2id["<pad>"], vocab_len)
