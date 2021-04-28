import torch
import numpy as np


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=768, num_layers=2, bidirectional=True)


if __name__ == '__main__':
    lstm = LSTM()
    params = sum([np.prod(list(p.size())) for p in lstm.parameters()])
    print(params)
    for p in lstm.parameters():
        print(p.shape)