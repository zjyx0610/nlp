import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from gensim.models import KeyedVectors
import jieba


class TextCNN(nn.Module):
    def __init__(self, vec_dim, filter_num, sentence_max_size, label_size, kernel_list):
        super(TextCNN, self).__init__()
        chanel_num = 1
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chanel_num, filter_num, (kernel, vec_dim)),
            nn.ReLU(),
            # 经过卷积之后，得到一个维度为sentence_max_size - kernel + 1的一维向量
            nn.MaxPool2d((sentence_max_size - kernel + 1, 1))
        )
            for kernel in kernel_list])
        # 全连接层，因为有2个标签
        self.fc = nn.Linear(filter_num * len(kernel_list), label_size)
        # dropout操作，防止过拟合
        self.dropout = nn.Dropout(0.5)
        # 分类
        self.sm = nn.Softmax(0)

    def forward(self, x):
        # Conv2d的输入是个四维的tensor，每一位分别代表batch_size、channel、length、width
        in_size = x.size(0)  # x.size(0)，表示的是输入x的batch_size
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(in_size, -1)  # 设经过max pooling之后，有output_num个数，将out变成(batch_size,output_num)，-1表示自适应
        out = F.dropout(out)
        out = self.fc(out)  # nn.Linear接收的参数类型是二维的tensor(batch_size,output_num),一批有多少数据，就有多少行
        return out


class MyDataset(Dataset):

    def __init__(self, file_list, label_list, sentence_max_size, embedding, word2id):
        self.x = file_list
        self.y = label_list
        self.sentence_max_size = sentence_max_size
        self.embedding = embedding
        self.word2id = word2id

    def __getitem__(self, index):
        # 读取文章内容
        words = []
        with open(self.x[index], "r", encoding="utf8") as file:
            for line in file.readlines():
                words.extend(segment(line.strip()))
        # 生成文章的词向量矩阵
        tensor = generate_tensor(words, self.sentence_max_size, self.embedding, self.word2id)
        return tensor, self.y[index]

    def __len__(self):
        return len(self.x)



def segment(content):
    res = []
    for word in jieba.cut(content):
        if word.strip() != "":
            res.append(word)
    return res


def get_file_list(source_dir):
    file_list = []
    # os.walk()遍历给定目录下的所有子目录，每个walk是三元组(root,dirs,files)
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    for root, dirs, files in os.walk(source_dir):
        file = [os.path.join(root, filename) for filename in files]
        file_list.extend(file)
    return file_list


def get_label_list(file_list):
    label_list = []
    for file in file_list:
        if "neg" in file:
            label_list.append(0)
        else:
            label_list.append(1)
    return label_list


def generate_tensor(sentence, sentence_max_size, embedding, word2id):
    tensor = torch.zeros([sentence_max_size, embedding.embedding_dim])
    for index in range(0, sentence_max_size):
        if index >= len(sentence):
            break
        else:
            word = sentence[index]
            if word in word2id:
                vector = embedding.weight[word2id[word]]
                tensor[index] = vector
            elif word.lower() in word2id:
                vector = embedding.weight[word2id[word.lower()]]
                tensor[index] = vector
    return tensor.unsqueeze(0)  # tensor是二维的，必须扩充为三维，否则会报错


def train_textcnn_model(net, train_loader, epoch, lr):
    print("begin training")
    net.train()  # 将模型设置为训练模式
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):  # 多批次循环
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # 清除所有优化的梯度
            output = net(data)  # 传入数据并前向传播获取输出
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            print("train epoch=" + str(i) + ",batch_id=" + str(batch_idx) + ",loss=" + str(loss.item() / 64))
    print('Finished Training')


def textcnn_model_test(net, test_loader):
    net.eval()  #将模型设置为测试模式
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            print("test batch_id=" + str(i))
            outputs = net(data)
            # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
            _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print('Accuracy of the network on test set: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    print("textCNN模型构建文本分类器")
    train_dir = "./data/aclImdb/train"  # 训练集路径
    test_dir = "./data/aclImdb/test"  # 测试集路径
    word2vec_dir = "./data/glove.model.6B.300d.txt"  # 训练好的词向量文件,写成相对路径好像会报错
    net_dir = "net.pkl"
    sentence_max_size = 300  # 每篇文章的最大词数量
    batch_size = 64
    filter_num = 100  # 每种卷积核的个数
    epoch = 8  # 迭代次数
    kernel_list = [3, 4, 5]  # 卷积核的大小
    label_size = 2
    lr = 0.001
    # 加载词向量模型
    print("加载词向量模型")
    # 加载词向量模型
    wv = KeyedVectors.load_word2vec_format(word2vec_dir, binary=False)
    word2id = {}  # word2id是一个字典，存储{word:id}的映射
    for i, word in enumerate(wv.index_to_key):
        word2id[word] = i
    # 根据已经训练好的词向量模型，生成Embedding对象
    print("生成Embedding对象")
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))
    # 获取训练数据
    print("获取训练数据")
    train_set = get_file_list(train_dir)
    train_label = get_label_list(train_set)
    train_dataset = MyDataset(train_set, train_label, sentence_max_size, embedding, word2id)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 获取测试数据
    print("获取测试数据")
    test_set = get_file_list(test_dir)
    test_label = get_label_list(test_set)
    test_dataset = MyDataset(test_set, test_label, sentence_max_size, embedding, word2id)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # 定义模型
    net = TextCNN(vec_dim=embedding.embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                  label_size=label_size,kernel_list=kernel_list)
    # 训练
    print("开始训练模型")
    train_textcnn_model(net, train_dataloader, epoch, lr)
    # 保存模型
    torch.save(net, net_dir)
    print("开始测试模型")
    textcnn_model_test(net, test_dataloader)

