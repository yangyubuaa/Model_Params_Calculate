from transformers import BertModel
import numpy as np

if __name__ == '__main__':
    e = BertModel.from_pretrained("/Volumes/PortableSSD/bert模型/bert-base-chinese")
    params = sum([np.prod(list(p.size())) for p in e.parameters()])
    print(params*4 / 1000000)
    for p in e.parameters():
        print(p.shape)