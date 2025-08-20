import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class LM_RNN(nn.Module):
    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        out_dropout=0.1,
        emb_dropout=0.1,
        n_layers=1,
    ):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(
            emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True
        )
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _ = self.rnn(emb)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output


class LM_LSTM(nn.Module):
    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        out_dropout=0.1,
        emb_dropout=0.1,
        n_layers=1,
    ):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's LSTM layer: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True
        )
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _ = self.lstm(emb)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output

class LM_LSTM_DROPOUT(nn.Module):
    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        out_dropout=0.1,
        emb_dropout=0.1,
        n_layers=1,
    ):
        super(LM_LSTM_DROPOUT, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True
        )
        self.pad_token = pad_index

        self.out_dropout = nn.Dropout(p=out_dropout)

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop_out_emb = self.emb_dropout(emb)
        rnn_out, _ = self.lstm(drop_out_emb)
        rnn_out_drop_out = self.out_dropout(rnn_out)
        output = self.output(rnn_out_drop_out).permute(0, 2, 1)
        return output
