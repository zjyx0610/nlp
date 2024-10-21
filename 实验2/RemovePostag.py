# -*- encoding: utf-8 -*-

str1 = ''

with open('../lab3/data/train.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip().split()
        line_clean = []
        for w in line:
            parts = w.split('/', 1)  # 最多分割一次，从右边开始
            w = parts[0]
            line_clean.append(w)
        str1 = str1 + ' '.join(line_clean) + '\n'

with open('data/train_Rp.txt', 'w', encoding='utf8') as f:
    f.write(str1)
