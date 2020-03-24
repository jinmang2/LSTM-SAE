import torch


def fine_tune_generator(src, trg, seq_len, bsz=32, use_cuda=False):
    assert src.dim() == 3, "source shape is (seq_len, total_length, num_dim)"
    assert trg.dim() == 2, "target shape is (seq_len, total_length)"
    total_length = trg.shape[1]
    num_dim = src.shape[-1]
    src_, trg_ = [], []
    for i in range(total_length):
        src_.append(src[:,i,:])
        trg_.append(trg[:,i])
        if len(src_) == bsz and len(trg_) == bsz:
            outsrc = torch.cat(src_, dim=0).view(seq_len, bsz, num_dim)
            outtrg = torch.cat(trg_, dim=0).view(seq_len, bsz)
            if use_cuda:
                outsrc = outsrc.cuda()
                outtrg = outtrg.cuda()
            yield outsrc, outtrg
            src_, trg_ = [], []
    if src_ is not [] and trg_ is not []:
        num_need_pad_zero = bsz - len(src_)
        for i in range(num_need_pad_zero):
            src_.append(torch.zeros(seq_len, num_dim))
            trg_.append(torch.zeros(seq_len))
        outsrc = torch.cat(src_, dim=0).view(seq_len, bsz, num_dim)
        outtrg = torch.cat(trg_, dim=0).view(seq_len, bsz)
        if use_cuda:
            outsrc = outsrc.cuda()
            outtrg = outtrg.cuda()
        yield outsrc, outtrg

def pre_train_generator(src, seq_len, bsz=32, use_cuda=False):
    assert src.dim() == 3, "source shape is (seq_len, total_length, num_dim)"
    total_length = src.shape[1]
    num_dim = src.shape[-1]
    src_ = []
    for i in range(total_length):
        src_.append(src[:,i,:])
        if len(src_) == bsz:
            outsrc = torch.cat(src_, dim=0).view(seq_len, bsz, num_dim)
            if use_cuda:
                outsrc = outsrc.cuda()
            yield outsrc
            src_ = []
    if src_ is not []:
        num_need_pad_zero = bsz - len(src_)
        for i in range(num_need_pad_zero):
            src_.append(torch.zeros(seq_len, num_dim))
        outsrc = torch.cat(src_, dim=0).view(seq_len, bsz, num_dim)
        if use_cuda:
            outsrc = outsrc.cuda()
        yield outsrc
