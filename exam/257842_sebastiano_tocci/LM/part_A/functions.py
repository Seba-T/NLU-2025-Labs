# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import math
import torch
import torch.nn as nn


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
