from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# 正态分布
def normal_dist(x, mean, std):
    prob_density = (1 / std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    return prob_density

# 高斯朴素贝叶斯分类
def gnb_classify(x_train, x_test, y_train, y_test):
    # 提取训练集的标签种类
    labels = np.unique(y_train, return_counts=False)

    model = {}
    for i in labels:
        # 取所有y=i的行
        group_i = x_train[y_train == i]
        # 对每一列求平均 axis＝0 列
        mean_i = np.mean(group_i, axis=0)
        # 对每一列求标准差
        std_i = np.std(group_i, axis=0)
        model[i] = {'mean': mean_i, 'std': std_i}

    test_results = []
    for i in range(x_test.shape[0]):
        result = []
        for j in labels:
            # 当前类别概率
            probs_j = normal_dist(x_test[i], model[j]['mean'], model[j]['std'])
            # 初始化总概率
            prob_j = 1
            # 求该类的概率
            for k in range(x_test.shape[1]):
                prob_j *= probs_j[k]
            result.append(prob_j)
        # 最大值的标签
        label_j = np.argmax(result)
        test_results.append(label_j)

    print("正确值：", list(y_test))
    print("预测值：", test_results)
    acc = (y_test == test_results).astype(int)
    print(f'模型预测准确率为{(np.sum(acc) / acc.shape[0]):.2%}')


# 二分类-乳腺癌分类数据集
breast_cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = \
    train_test_split(breast_cancer.data, breast_cancer.target, random_state=22,test_size=0.2)
print("二分类：")
gnb_classify(x_train, x_test, y_train, y_test)

# 三分类-鸢尾花数据集
iris = load_iris()
x_train, x_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, random_state=22, test_size=0.2)
print("三分类：")
gnb_classify(x_train, x_test, y_train, y_test)
