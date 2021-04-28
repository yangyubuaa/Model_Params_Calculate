import torch

if __name__ == '__main__':
    ln = torch.nn.LayerNorm(768)
    for p in ln.parameters():
        print(p.shape)