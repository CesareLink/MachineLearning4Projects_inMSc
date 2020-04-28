# -*- coding: utf-8 -*-
#依次搭建Word Averaging, RNN/LSTM和CNN    -3
#_____________________________________________________________________________________
#数据准备
#TorchText提供了Field这个概念决定了数据的处理方式，例如下方使用spacy的方式来tokenize英文句子
#如果不声明tokenize，则默认的分词方法是使用空格

#导入模块
import torch
from torchtext import data
#为了可复现设置随机量为固定随机量
SEED =1234
#设置torch的随机量
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#设置文档和标签的切割
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

#TorchText内置很多常见的自然语言处理数据集,从IMDB下载数据集
from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

#观察数据集状态
print(f'Number of training samples: {len(train_data)}')
print(f'Number of testing samples: {len(test_data)}')

#建立validation set
import random
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
#默认7：3分开

#创建vocabulary：将每个单词映射到向量中
TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

#利用iteration来进行batch的sample的填充，实现不断替换batch
BATCH_SIZE = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE)

#_____________________________________________________________________________________
#搭建CNN模型
import torch.nn as nn
import torch.nn.functional as F
#搭建模型
class CNN(nn.Module):
    #初始化模型的结构
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 output_dim, dropout, pad_idx):
        super().__init__()
        #设置内部embed参数为输入的参数
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        #设置RNN层
        self.convs = nn.ModuleLis([nn.Conv2d(in_channels = 1,out_channels=n_filters,
                            kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        #设置全连接层
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        #设置dropout的方法
        self.dropout = nn.Dropout(dropout)
        
    #设置前向通路
    def forward(self, text):
        text = text.permute(1, 0) # [batch size, sent len] #permute是一个维度换位函数
        #设置编码层，从原始数据到embedded
        embedded = self.embedding(text) 
        #向量维度为[batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1) #batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #torch.cat用于拼接tensor，dim=0为竖着拼，dim=1为横着拼
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)
    
#设置模型超参数
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

#建立模型，模型实例化
model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS,
            FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

#计算模型所拥有的参数总量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#读取预训练模型
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

#设置UNK的不常用词向量和pad词向量
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

#_____________________________________________________________________________________
#训练模型
#采用调用函数的模式进行模型训练和评估

#导入优化器
import torch.optim as optim
optimizer = optim.Adam(model.parameters())
#使用BCE和逻辑回归作为损失
criterion = nn.BCEWithLogitsLoss()
#如果需要导入到GPU则应加入：
#model = model.to(device)
#criterion = criterion.to(device)

#计算准确率
def binary_accuracy(preds,y):
    #读取输入的预测preds和真实结果y，.round返回四舍五入的结果（以匹配y），sigmoid用来进行预测二值分类
    rounded_preds = torch.round(torch.sigmoid(preds))
    #将预测与真实标签y对比，将正确的导出
    correct = (rounded_preds==y).float()
    #用正确的总和除以总的长度为准确率
    acc = correct.sum()/len(correct)      
    return acc

#用于训练模型的调用函数
def train(model, iterator, optimizer, criterion):
    #设置每个epoch的初始化参数，以作调用
    epoch_loss = 0
    epoch_acc = 0
    #声明该内容为模型的训练过程，需要进行权重的更替    
    model.train()
    #执行循环    
    for batch in iterator:
        #置零优化器梯度避免出现障碍
        optimizer.zero_grad()
        #提供预测结果并压缩至一维
        predictions = model(batch.text).squeeze(1)
        #从结果中导出loss（由上面初始化设定的nn.BCEWithLogitsLoss）和acc由上述函数计算
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        #进行反向传播
        loss.backward()
        #执行优化器优化
        optimizer.step()
        #向epoch中充填每次运行的loss，作为batch的loss，同理acc
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    #返回batch的loss和准确率
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#评估模型函数
def evaluate(model, iterator, criterion):
    #初始化
    epoch_loss = 0
    epoch_acc = 0
    #表明该函数为测试函数，权重不进行更新
    model.eval()
    #进入循环
    #with torch.no_grad 用使得autograd不计算计算图，也就是不更新权重
    with torch.no_grad():
        #进行测试机batch的循环
        for batch in iterator:
            #类似训练集的评估模式
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    #返回结果loss和acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#构建时间轴以记录整个过程的时间窗
import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    #转换成分钟
    elapsed_mins = int(elapsed_time /60)
    #转换成秒
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#_____________________________________________________________________________________
#正式运行主程序
#设置迭代次数
N_EPOCHS = 5
#设置最佳验证集loss结果
best_valid_loss = float('inf')
#循环训练
for epoch in range(N_EPOCHS):
    #设置时间轴以提取时间
    start_time = time.time()
    #训练模型
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    #验证模型
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    #结束时间轴
    end_time = time.time()
    #运算运行时间
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #比较验证集loss并存储，进入下一个循环
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')
    #输出当前的loss和结果
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

   


