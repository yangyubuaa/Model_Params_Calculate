from transformers import BertModel
from torchstat import stat


if __name__ == '__main__':
    bert = BertModel.from_pretrained("/Volumes/PortableSSD/bert模型/bert-base-chinese")
    print(stat(bert, (64, 500, )))
