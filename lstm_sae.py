import random

import torch
import torch.nn as nn

class Seq2CEncoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, bsz=64, n_layers=1):
        super(Seq2CEncoder, self).__init__()
        h0 = torch.zeros(n_layers, bsz, hid_dim)
        c0 = torch.zeros(n_layers, bsz, hid_dim)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.LSTM = nn.LSTM(input_size=inp_dim,
                            hidden_size=hid_dim,
                            num_layers=n_layers)

    def forward(self, x):
        outputs, (hidden, cell) = self.LSTM(x)
        return hidden, cell

class C2SeqDecoder(nn.Module):
    def __init__(self, out_dim, hid_dim, bsz=64, n_layers=1):
        super(C2SeqDecoder, self).__init__()
        h0 = torch.zeros(n_layers, bsz, hid_dim)
        c0 = torch.zeros(n_layers, bsz, hid_dim)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.LSTM = nn.LSTM(input_size=out_dim,
                            hidden_size=hid_dim,
                            num_layers=n_layers)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.LSTM(x)
        return output, (hidden, cell)

class LSTM_AE(nn.Module):
    def __init__(self,
                 inp_dim,
                 out_dim,
                 hid_dim,
                 bsz=64,
                 n_layers=1,
                 use_cuda=False):
        super(LSTM_AE, self).__init__()
        self.bsz = bsz
        self.use_cuda = use_cuda
        self.output_dim = out_dim
        self.encoder = Seq2CEncoder(inp_dim, hid_dim, bsz, n_layers)
        self.decoder = C2SeqDecoder(out_dim, hid_dim, bsz, n_layers)
        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, src):
        # 자기 표현 학습
        bsz = self.bsz
        seq_len = src.shape[0] # seq_len
        out_dim = self.output_dim

        outputs = torch.zeros(seq_len, bsz, out_dim)
        if self.use_cuda:
            outputs = outputs.cuda()

        hidden, cell = self.encoder(src)

        # start first input
        input_ = src[0,:]
        for t in range(1, seq_len): # 1 to seq_len-1, except first input
            output, (hidden, cell) = self.decoder(input_, hidden, cell)
            outputs[t] = output
            input_ = src[t, :] # next input

        return outputs

class LSTM_SAE(nn.Module):
    """Pre-training model"""
    def __init__(self,
                 inp_dim,
                 out_dim,
                 hid_dim,
                 bsz=64,
                 n_layers=1,
                 use_cuda=False):
        super(LSTM_SAE, self).__init__()
        self.block1 = LSTM_AE(inp_dim, out_dim, hid_dim, bsz, n_layers, use_cuda)
        self.block2 = LSTM_AE(inp_dim, out_dim, hid_dim, bsz, n_layers, use_cuda)
        self.block3 = LSTM_AE(inp_dim, out_dim, hid_dim, bsz, n_layers, use_cuda)

class DLSTM(nn.Module):
    """Fine-tuning model"""
    def __init__(self):
        pass
