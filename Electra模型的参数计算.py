from transformers import ElectraModel
import numpy as np

if __name__ == '__main__':
    e = ElectraModel.from_pretrained("/Volumes/PortableSSD/bert模型/chinese-electra-180g-small-discriminator")
    params = sum([np.prod(list(p.size())) for p in e.parameters()])
    print(params*4 / 1000000)
    for p in e.parameters():
        print(p.shape)