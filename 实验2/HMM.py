import json
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def data_split(datafile):
    with open(datafile, 'r', encoding='utf8') as f:
        dataset = f.readlines()
    train_set, test_set = train_test_split(dataset, random_state=22, test_size=0.2)

    with open('./data/train.txt', 'w', encoding='utf8') as f1:
        for line in train_set:
            f1.write(line)
    with open('./data/test.txt', 'w', encoding='utf8') as f2:
        for line in test_set:
            f2.write(line)
    f1.close()
    f2.close()

class HMM:
    def __init__(self):
        self.trans_p = {'S': {}, 'B': {}, 'M': {}, 'E': {}}  # 转移
        self.emit_p = {'S': {}, 'B': {}, 'M': {}, 'E': {}}  # 发射
        self.start_p = {'S': 0, 'B': 0, 'M': 0, 'E': 0}  # 初始
        self.state_num = {'S': 0, 'B': 0, 'M': 0, 'E': 0}
        self.state_list = ['S', 'B', 'M', 'E']
        self.line_num = 0
        self.smooth = 1e-6

    @staticmethod
    def get_state(word):
        if len(word) == 1:
            state = ['S']
        else:
            state = list('B' + 'M' * (len(word) - 2) + 'E')
        return state

    def train(self, filepath, save_model=False):
        print("正在训练模型……")
        start_time = time.thread_time()
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f.readlines():
                self.line_num += 1
                line = line.strip().split()
                # 获取观测（字符）序列
                char_seq = list(''.join(line))
                # 获取状态（BMES）序列
                state_seq = []
                for word in line:
                    state_seq.extend(self.get_state(word))
                # 判断是否等长
                assert len(char_seq) == len(state_seq)
                # 统计参数
                for i, s in enumerate(state_seq):
                    self.state_num[s] = self.state_num.get(s, 0) + 1.0
                    self.emit_p[s][char_seq[i]] = self.emit_p[s].get(char_seq[i], 0) + 1.0
                    if i == 0:
                        self.start_p[s] += 1.0
                    else:
                        last_s = state_seq[i - 1]
                        self.trans_p[last_s][s] = self.trans_p[last_s].get(s, 0) + 1.0
        # 归一化：
        self.start_p = {
            k: (v + 1.0) / (self.line_num + 4)
            for k, v in self.start_p.items()
        }
        self.emit_p = {
            k: {w: num / self.state_num[k]
                for w, num in dic.items()}
            for k, dic in self.emit_p.items()
        }
        self.trans_p = {
            k1: {k2: num / self.state_num[k1]
                 for k2, num in dic.items()}
            for k1, dic in self.trans_p.items()
        }
        end_time = time.thread_time()
        print("训练完成，耗时 {:.3f}s".format(end_time - start_time))
        # 保存参数
        if save_model:
            parameters = {
                'start_p': self.start_p,
                'trans_p': self.trans_p,
                'emit_p': self.emit_p
            }
            jsonstr = json.dumps(parameters, ensure_ascii=False, indent=4)
            param_filepath = "./data/HmmParam_Token.json"
            with open(param_filepath, 'w', encoding='utf8') as jsonfile:
                jsonfile.write(jsonstr)

    def viterbi(self, text):
        text = list(text)
        num_states = len(self.state_list)
        num_chars = len(text)

        # 初始化dp矩阵，使用numpy的二维数组
        dp = np.zeros((num_chars, num_states), dtype=object)
        back_pointers = np.empty((num_chars, num_states), dtype=object)

        # 初始化dp和back_pointers的第一行
        for i, ch in enumerate(self.state_list):
            dp[0, i] = (self.start_p[ch] * self.emit_p[ch].get(text[0], self.smooth), '_start_')
            back_pointers[0, i] = None

        # 动态规划地更新dp矩阵
        for i in range(1, num_chars):
            for j, ch in enumerate(self.state_list):
                max_prob = float('-inf')
                last_state = None
                for k, ls in enumerate(self.state_list):
                    prob = dp[i - 1, k][0] * self.trans_p[ls].get(ch, self.smooth) * self.emit_p[ch].get(text[i],self.smooth)
                    if prob > max_prob:
                        max_prob = prob
                        last_state = k
                dp[i, j] = (max_prob, last_state)
                back_pointers[i, j] = last_state

        # 回溯最优路径
        path = []
        back_point = np.argmax(dp[num_chars - 1, :], axis=0)
        path.append(self.state_list[back_point])
        for i in range(num_chars - 2, -1, -1):
            back_point = back_pointers[i + 1, back_point]
            path.append(self.state_list[back_point])
        path.reverse()

        return path

    def cut(self, text):
        state = self.viterbi(text)
        cut_res = []
        begin = 0
        for i, ch in enumerate(text):
            if state[i] == 'B':
                begin = i
            elif state[i] == 'E':
                cut_res.append(text[begin:i + 1])
            elif state[i] == 'S':
                cut_res.append(text[i])
        return cut_res

def eval(predict, groundtruth):
    assert len(predict) == len(groundtruth)
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(predict)):
        right = len([j for j in predict[i] if j in groundtruth[i]])
        tp += right
        fn += len(groundtruth[i]) - right
        fp += len(predict[i]) - right
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def evalutate(dataset, token_res):
    precision, recall, f1 = eval(token_res, dataset)
    print("精确率:\t{:.3%}".format(precision))
    print("召回率:\t{:.3%}".format(recall))
    print("f1:\t{:.3%}".format(f1))

def word_segmentation_eval(trainfile,testfile):
    # 利用 HMM 模型进行分词
    # 训练
    hmm_model = HMM()
    hmm_model.train(trainfile, save_model=True)

    with open(testfile, 'r', encoding='utf8') as f:
        dataset = [line.strip().split(' ') for line in f.readlines()]
    # dataset = dataset[0:6000]
    input_data = [''.join(line) for line in dataset]

    token_res = []
    stime = time.thread_time()
    for line in tqdm(input_data):
        token_res.append(hmm_model.cut(line))  # 预测分词
    etime = time.thread_time()
    evalutate(dataset, token_res)
    print("训练完成，耗时 {:.3f}s".format(etime - stime))

    # 保存分词结果
    with open('./data/PeopleDaily_Token_mp_result.txt', 'w', encoding='utf8') as f:
        for i in token_res:
            f.write(' '.join(i) + '\n')
    return token_res



if __name__ == "__main__":

    # 数据预处理
    # data_split("./data/train_Rp.txt")

    # 评估分词模型
    word_segmentation_eval('./data/train.txt','./data/test.txt')

