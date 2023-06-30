import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
from model import LlamaForCausalLM, LlamaConfig
from tqdm import tqdm

# 528917317

device = "cuda" if torch.cuda.is_available() else "cpu"
word_index = {}
max_lens = 128
batch_size = 24
config = LlamaConfig.from_json_file("config.json")
lr = 1e-4
epochs = 25


# 数据读取
def read_data():
    global word_index

    with open("xiaohuangji.txt", "r", encoding="utf-8") as f:
        datas = f.readlines()

    if os.path.exists("word_index.json"):
        with open("word_index.json", "r", encoding="utf-8") as f:
            word_index = json.load(f)
    else:

        word_dict = {}

        for i in datas:
            for n in i:
                if n in word_dict:
                    word_dict[n] += 1
                else:
                    word_dict[n] = 1

        word_list = [[k, v] for k, v in word_dict.items() if k not in [" ", "\n", "\t"]]
        word_list.sort(key=lambda x:x[-1], reverse=True)

        word_dict = {
            "pad": 0,
            "sta": 1,
            "end": 2,
            "unk": 3
        }

        for word, _ in word_list:
            word_dict[word] = len(word_dict)

        json_data = json.dumps(word_dict, indent=4, ensure_ascii=False)

        with open("word_index.json", "w", encoding="utf-8") as f:
            f.write(json_data)

    return datas


# 自定义数据集
class XHJData(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def to_word_id(self, text):
        value = []
        for word in text:
            if word in word_index:
                value.append(word_index[word])
            else:
                continue

        return value

    def __getitem__(self, item):

        sentence = self.datas[item]

        texts = sentence.split("\t")

        q = texts[0]
        a = texts[-1]

        q = self.to_word_id(q)
        a = self.to_word_id(a)

        # 输入什么就输出什么

        values = q + [word_index["unk"]] + a
        values = values[:(max_lens-2)]
        values = [word_index["sta"]] + values + [word_index["end"]]
        for _ in range(max_lens-len(values)):
            values.append(word_index["pad"])

        values = np.int32(values)

        return values


def train():

    datas = read_data()

    train_data = XHJData(datas)
    train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    model = LlamaForCausalLM(config=config)
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        pbar = tqdm(train_data)

        for step, values in enumerate(pbar):
            values = values.to(device)
            input_ids = values[:, :-1]
            labels = values[:, 1:].long()

            attention_mask = (input_ids != 0).int()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            bz, seq_len, hidden_size = logits.shape

            loss = loss_fc(logits.reshape((bz*seq_len, hidden_size)), labels.reshape((bz*seq_len,)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=-1)
            index = torch.where(attention_mask != 0)

            pred = pred[index]
            labels = labels[index]

            acc = torch.mean((pred == labels).float())

            s = "epoch:{} - step:{} - loss:{:.3f} - acc:{:.3f}".format(epoch, step, loss, acc)

            pbar.set_description(s)

        torch.save(model.state_dict(), "model.pkl")


if __name__ == '__main__':
    train()

