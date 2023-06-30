

with open("xiaohuangji50w_nofenci.conv", "r", encoding="utf-8") as f:
    datas = f.read()

datas = datas.split("E\n")


texts = []
for i in datas[1:]:

    i = i[1:-1]
    i = i.split("\nM")

    sentence = i[0] + "\t" + i[-1]
    texts.append(sentence+"\n")

with open("xiaohuangji.txt", "w", encoding="utf-8") as f:
    f.writelines(texts)


