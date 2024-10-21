import torch
from torchcrf import CRF
from tqdm import tqdm
import time
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd

pad_token = '<pad>'
pad_id = 0
unk_token = '<unk>'
unk_id = 1
tag_to_id = {'<pad>': 0, 'O': 1, 'B-LOC': 2, 'I-LOC': 3, 'B-PER': 4, 'I-PER': 5, 'B-ORG': 6, 'I-ORG': 7}
id_to_tag = {id: tag for tag, id in tag_to_id.items()}
word_to_id = {'<pad>': 0, '<unk>': 1}
tags_num = len(tag_to_id)

def read_data(filepath):
    sentences = []
    tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        tmp_sentence = []
        tmp_tags = []
        for line in f:
            if line == '\n' and len(tmp_sentence) != 0:
                assert len(tmp_sentence) == len(tmp_tags)
                sentences.append(tmp_sentence)
                tags.append(tmp_tags)
                tmp_sentence = []
                tmp_tags = []
            else:
                line = line.strip().split(' ')
                tmp_sentence.append(line[0])
                tmp_tags.append(line[3])
        if len(tmp_sentence) != 0:
            assert len(tmp_sentence) == len(tmp_tags)
            sentences.append(tmp_sentence)
            tags.append(tmp_tags)
    return sentences, tags

def build_vocab(sentences):
    global word_to_id
    for sentence in sentences:  # 建立word到索引的映射
        for word in sentence:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
    return word_to_id

def convert_to_ids_and_padding(seqs, to_ids):
    ids = []
    for seq in seqs:
        if len(seq)>=maxlen: # 截断
            ids.append([to_ids[w] if w in to_ids else unk_id for w in seq[:maxlen]])
        else: # padding
            ids.append([to_ids[w] if w in to_ids else unk_id for w in seq] + [0]*(maxlen-len(seq)))

    return torch.tensor(ids, dtype=torch.long)

def load_data(filepath, word_to_id, shuffle=False):
    sentences, tags = read_data(filepath)

    inps = convert_to_ids_and_padding(sentences, word_to_id)
    trgs = convert_to_ids_and_padding(tags, tag_to_id)

    inp_dset = torch.utils.data.TensorDataset(inps, trgs)
    inp_dloader = torch.utils.data.DataLoader(inp_dset,batch_size=batch_size,shuffle=shuffle,num_workers=4)
    return inp_dloader

class BiLSTM_CRF(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BiLSTM_CRF, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
        self.bi_lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size // 2, batch_first=True,
                                     bidirectional=True)  # , dropout=0.2)
        self.hidden2tag = torch.nn.Linear(hidden_size, tags_num)

        self.crf = CRF(num_tags=tags_num, batch_first=True)

    def init_hidden(self, batch_size):
        # device = 'cpu'
        _batch_size = batch_size
        return (torch.randn(2, _batch_size, self.hidden_size // 2, device=device),
                torch.randn(2, _batch_size, self.hidden_size // 2, device=device))

    def forward(self, inp):  # inp [b, seq_len=60]
        self.bi_lstm.flatten_parameters()

        embeds = self.embedding(inp)  # [b,seq_len]=>[b, seq_len, embedding_dim]
        lstm_out, _ = self.bi_lstm(embeds, None)

        logits = self.hidden2tag(lstm_out)  # [b, seq_len, hidden_size]=>[b, seq_len, tags_num]
        return logits # [b, seq_len=60, tags_num=10]

    # 计算CRF 条件对数似然，并返回其负值作为loss
    def crf_neg_log_likelihood(self, inp, tags, mask=None, inp_logits=False):  # [b, seq_len, tags_num], [b, seq_len]
        if inp_logits:
            logits = inp
        else:
            logits = self.forward(inp)

        if mask is None:
            mask = torch.logical_not(torch.eq(tags, torch.tensor(0)))
            mask = mask.type(torch.uint8)

        crf_llh = self.crf(logits, tags, mask, reduction='mean')
        return -crf_llh

    def crf_decode(self, inp, mask=None, inp_logits=False):
        if inp_logits:
            logits = inp
        else:
            logits = self.forward(inp)

        if mask is None and inp_logits is False:
            mask = torch.logical_not(torch.eq(inp, torch.tensor(0)))
            mask = mask.type(torch.uint8)

        return self.crf.decode(emissions=logits, mask=mask)

def train_step(model, inps, tags, optimizer):
    inps = inps.to(device)
    tags = tags.to(device)
    mask = torch.logical_not(torch.eq(inps, torch.tensor(0)))  # =>[b, seq_len]

    model.train()  # 设置train mode
    optimizer.zero_grad()  # 梯度清零

    # forward
    logits = model(inps)
    loss = model.crf_neg_log_likelihood(logits, tags, mask=mask, inp_logits=True)

    # backward
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    preds = model.crf_decode(logits, mask=mask, inp_logits=True) # List[List]
    pred_without_pad = []
    for pred in preds:
        pred_without_pad.extend(pred)
    tags_without_pad = torch.masked_select(tags, mask).cpu().numpy() # 返回是1维张量
    metric = accuracy_score(pred_without_pad, tags_without_pad)

    return loss.item(), metric

@torch.no_grad()
def validate_step(model, inps, tags):
    inps = inps.to(device)
    tags = tags.to(device)
    mask = torch.logical_not(torch.eq(inps, torch.tensor(0)))
    model.eval()  # 设置eval mode

    # forward
    logits = model(inps)
    loss = model.crf_neg_log_likelihood(logits, tags, mask=mask, inp_logits=True)

    preds = model.crf_decode(logits, mask=mask, inp_logits=True)  # List[List]
    pred_without_pad = []
    for pred in preds:
        pred_without_pad.extend(pred)
    tags_without_pad = torch.masked_select(tags, mask).cpu().numpy()  # 返回是1维张量
    metric = accuracy_score(pred_without_pad, tags_without_pad)

    return loss.item(), metric


def train_model(model, train_dloader, val_dloader, optimizer, num_epochs=10, print_every=150):
    starttime = time.time()
    print('start training...')

    best_metric = 0.
    for epoch in range(1, num_epochs+1):
        # 训练
        loss_sum, metric_sum = 0., 0.
        for step, (inps, tags) in enumerate(train_dloader, start=1):
            loss, metric = train_step(model, inps, tags, optimizer)
            loss_sum += loss
            metric_sum += metric

            # 打印batch级别日志
            if step % print_every == 0:
                print(f'[step = {step}] loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}')

        # 验证 一个epoch的train结束，做一次验证
        val_loss_sum, val_metric_sum = 0., 0.
        for val_step, (inps, tags) in enumerate(val_dloader, start=1):
            val_loss, val_metric = validate_step(model, inps, tags)
            val_loss_sum += val_loss
            val_metric_sum += val_metric


        # 记录和收集 1个epoch的训练和验证信息
        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record

        # 打印epoch级别日志
        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
               record[0], record[1], metric_name, record[2], record[3], metric_name, record[4]))

        # 保存最佳模型参数
        current_metric_avg = val_metric_sum/val_step
        if current_metric_avg > best_metric:
            best_metric = current_metric_avg
            checkpoint = f'epoch{epoch:03d}_valacc{current_metric_avg:.3f}_ckpt.tar'
            model_sd = copy.deepcopy(model.state_dict())
            # 保存
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
            }, checkpoint)


    endtime = time.time()
    time_elapsed = endtime - starttime
    print('training finished...')
    print('and it costs {} h {} min {:.2f} s'.format(int(time_elapsed // 3600),
                                                               int((time_elapsed % 3600) // 60),
                                                               (time_elapsed % 3600) % 60))

    print('Best val Acc: {:4f}'.format(best_metric))
    return df_history

@torch.no_grad()
def eval_step(model, inps, tags):
    inps = inps.to(device)
    tags = tags.to(device)
    mask = torch.logical_not(torch.eq(inps, torch.tensor(0)))

    # forward
    logits = model(inps)
    preds = model.crf_decode(logits, mask=mask, inp_logits=True)  # List[List]
    pred_without_pad = []
    for pred in preds:
        pred_without_pad.extend(pred)
    tags_without_pad = torch.masked_select(tags, mask).cpu()  # 返回是1维张量

    return torch.tensor(pred_without_pad), tags_without_pad

def evaluate(model, test_dloader):
    model.eval()  # 设置eval mode
    starttime = time.time()
    print('start evaluating...')
    preds, labels = [], []
    for step, (inps, tags) in enumerate(tqdm(test_dloader), start=1):
        pred, tags = eval_step(model, inps, tags)
        preds.append(pred)
        labels.append(tags)

    y_true = torch.cat(labels, dim=0)
    y_pred = torch.cat(preds, dim=0)
    endtime = time.time()
    print('evaluating costs: {:.2f}s'.format(endtime - starttime))
    return y_true.cpu(), y_pred.cpu()

def get_metrics(y_true, y_pred):
    average = 'weighted'
    print(average+'_precision_score:{:.3f}'.format(precision_score(y_true, y_pred, average=average)))
    print(average+'_recall_score:{:.3}'.format(recall_score(y_true, y_pred, average=average)))
    print(average+'_f1_score:{:.3f}'.format(f1_score(y_true, y_pred, average=average)))

    print('accuracy:{:.3f}'.format(accuracy_score(y_true, y_pred)))
    print('confusion_matrix:\n', confusion_matrix(y_true, y_pred))
    print('classification_report:\n', classification_report(y_true, y_pred))

if __name__ == '__main__':

    LR = 1e-3
    EPOCHS = 30
    maxlen = 60
    embedding_dim = 100
    hidden_size = 128
    batch_size = 512
    device = 'cpu'

    metric_name = 'acc'
    df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])

    train_dloader = load_data('./data/train.txt', word_to_id)
    val_dloader = load_data('./data/dev.txt', word_to_id)

    model = BiLSTM_CRF(len(word_to_id), hidden_size)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_model(model, train_dloader, val_dloader, optimizer, num_epochs=EPOCHS, print_every=50)

    checkpoint = 'epoch001_valacc0.853_ckpt.tar'

    # 加载测试数据
    test_dloader = load_data('./data/test.txt', word_to_id)

    # 加载模型
    reloaded_model = BiLSTM_CRF(len(word_to_id), hidden_size)
    reloaded_model = reloaded_model.to(device)

    print('Loading model weights...')
    ckpt = torch.load(checkpoint, map_location=device)
    model_sd = ckpt['net']

    reloaded_model.load_state_dict(model_sd)
    print('Model loaded success!')

    y_true, y_pred = evaluate(reloaded_model, test_dloader)
    get_metrics(y_true, y_pred)