import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch


class LM_LSTM_DROPOUT_WEIGHT_TYING(nn.Module):
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
        super(LM_LSTM_DROPOUT_WEIGHT_TYING, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True
        )
        self.pad_token = pad_index

        self.out_dropout = nn.Dropout(p=out_dropout)

         # If the hidden size is different from the embedding size, then we need a linear layer to "bridge" between the two
        if hidden_size != emb_size:
            self.proj = nn.Linear(hidden_size, emb_size, bias=False)
            out_in_features = emb_size
        else:
            self.proj = None
            out_in_features = hidden_size
        self.output = nn.Linear(out_in_features, output_size, bias=True)
        self.output.weight = self.embedding.weight  # weight tying

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop_out_emb = self.emb_dropout(emb)
        rnn_out, _ = self.lstm(drop_out_emb)
        rnn_out_drop_out = self.out_dropout(rnn_out)
        h = self.proj(rnn_out_drop_out) if self.proj is not None else rnn_out_drop_out
        output = self.output(h).permute(0, 2, 1)
        return output

class LM_LSTM_VAR_DROPOUT(nn.Module):
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
        super(LM_LSTM_VAR_DROPOUT, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout_p = emb_dropout

        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True
        )
        self.pad_token = pad_index
        self.out_dropout_p = out_dropout

        if hidden_size != emb_size:
            self.proj = nn.Linear(hidden_size, emb_size, bias=False)
            out_in_features = emb_size
        else:
            self.proj = None
            out_in_features = hidden_size
        self.output = nn.Linear(out_in_features, output_size, bias=True)
        self.output.weight = self.embedding.weight  # weight tying

    def locked_dropout(self, x, dropout=0.5, training=True):
        if not training or dropout == 0.0:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = m.div_(1 - dropout).expand_as(x)
        return mask * x

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop_out_emb = self.locked_dropout(emb, self.emb_dropout_p, self.training)
        rnn_out, _ = self.lstm(drop_out_emb)
        rnn_out_drop_out = self.locked_dropout(rnn_out, self.out_dropout_p, self.training)
        h = self.proj(rnn_out_drop_out) if self.proj is not None else rnn_out_drop_out
        output = self.output(h).permute(0, 2, 1)
        return output
