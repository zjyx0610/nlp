import shutil
from sys import platform


# 计算行数，就是单词数
def getFileLineNums(filename):
    f = open(filename, 'r', encoding="utf8")
    count = 0
    for line in f:
        count += 1
    return count


# Linux或者Windows下打开词向量文件，在开始增加一行
def prepend_line(infile, outfile, line):
    with open(infile, 'r', encoding="utf8") as old:
        with open(outfile, 'w', encoding="utf8") as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r', encoding="utf8") as fin:
        with open(outfile, 'w', encoding="utf8") as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load(filename):
    num_lines = getFileLineNums(filename)
    gensim_file = './data/glove.model.6B.300d.txt'
    gensim_first_line = "{} {}".format(num_lines, 300)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)



load('./data/glove.6B.300d.txt')

