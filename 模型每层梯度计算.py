import torch
from torch.nn.functional import cross_entropy


class ResLinear(torch.nn.Module):
    def __init__(self):
        super(ResLinear, self).__init__()
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid()
        )
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid()
        )
        self.linear3 = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid()
        )
        self.linear4 = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid()
        )
        self.linear5 = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid()
        )
        self.linear6 = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        linear1_out = input + self.linear1(input)
        linear2_out = linear1_out + self.linear2(linear1_out)
        linear3_out = linear2_out + self.linear3(linear2_out)
        linear4_out = linear3_out + self.linear4(linear3_out)
        linear5_out = linear4_out + self.linear5(linear4_out)
        linear6_out = linear5_out + self.linear6(linear5_out)
        return linear6_out


class SigmoidLinear(torch.nn.Module):
    def __init__(self):
        super(SigmoidLinear, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid(),
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid(),
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid(),
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid(),
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid(),
            torch.nn.Linear(300, 300),
            torch.nn.Sigmoid(),
        )

    def forward(self, input):
        return self.linear(input)


class ReLULinear(torch.nn.Module):
    def __init__(self):
        super(ReLULinear, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
        )

    def forward(self, input):
        return self.linear(input)


if __name__ == '__main__':
    l = ResLinear()
    input = torch.ones((1, 300))
    target = torch.tensor([1])
    print(target.shape)
    loss = cross_entropy(l(input), target)
    loss.backward()
    print(len(list(l.parameters())))
    for p in l.parameters():
        print(p.shape)
        print(p.grad)