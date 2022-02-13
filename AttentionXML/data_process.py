import jieba
import numpy as np
from gensim.models import Word2Vec
import re
import emoji
import pandas as pd
pd.options.mode.use_inf_as_na = True

org_data = pd.read_csv('./data/only_textinformation_feed_content_label.csv')
org_data = org_data.drop(columns=['Unnamed: 0'])
# 将content为空的数据去除
org_data = org_data[org_data['content'].notnull()]

# 获取所有动态的文本内容
df = org_data[['content', 'label_ids']]

# 读取停用词表
stop_word_file = './data/stop_words.npy'
stop_words = np.load(stop_word_file).tolist()

# 读取领域词汇
jieba.load_userdict('./data/new_dict.txt')

# 读入量词表
quantifier_dict = list()
with open('./data/quantifier_dict.txt', 'r') as f:
    my_data = f.readlines()
    for line in my_data:
        quantifier_dict.append(line.strip())


# 去除数字和其后面的量词
# 只有数字后面跟的是量词的时候才会把量词和数字都去掉，否则不去掉数字
def remove_quantifier(s : str) -> str:
    digit = re.findall(r'\d+', s)
    order = [i.start() for i in re.finditer(r'\d+', s)] 
#     for index, number in enumerate(digit):
#         print(index, number,'\n')
    res = ''
    last_pos = 0
    arr_size = len(digit)
    for i in range(arr_size):
        number, pos = digit[i], order[i]
        next_pos = pos+len(number)
        res = res + s[last_pos : pos]
        if next_pos < len(s):
            next_word = s[next_pos]
            if next_word in quantifier_dict:
                # 对数字+量词进行删除
                last_pos = next_pos + 1
            else:
                last_pos = pos
    res = res + s[last_pos : ]
    return res


# 分词
# 分词 -> 保证出来的是list
def content_cut(x):
    x = remove_quantifier(x) # 移除量词
    x = re.sub(emoji.get_emoji_regexp(), '', x)   # 删除表情
    x = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x)   # 删除URL
    x = re.sub('\[.*?\]', '', x)   # 删除所有[...]
    x = re.sub('<br>', '。', x)   # 将<br>替换为‘,’
    x = re.sub('<.*?>', '', x)   # 删除所有<...>
    x = re.sub('&nbsp', ' ', x)   # 将&nbsp替换为空格
    x = re.sub('\\u200b', '', x)   # 删除所有\u200b
#     x = re.sub('[^A-Za-z\u4e00-\u9fa5]+', ' ', x)   # 将中文字符和字母以外的字符换成空格
    x = re.sub('[^\u4e00-\u9fa5]+',' ',x)   # 将中文字符以外的字符换成空格
    
    x = jieba.lcut(x, cut_all=False, HMM=True)
    x = " ".join(x)
    return x.split()


# conovert a str to a list
# '[1,3]' -> ['1','3']
def str2list(x : str) -> list :
    x = x[1:-1].split(',')
    x = [str(i) for i in x]
    return x

# 分词：处理中文、处理停用词
id_org_content = dict()       # id号 索引 原文
id_cuted_content = dict()     # id号 索引 分词后的结果
content_cuted = list()        # 只有分词后的结果
labels = list()               # 标签. [['1','334'],['1'],...]
for index, val in df.iterrows():
    v, l = val[0], str2list(val[1])
    temp = list()
    
    if isinstance(v, float):
        # content内容为'空'的情况
        # temp.append('空')
        continue
    v = content_cut(v)
    for i in v:
        if i not in stop_words and i != '丨':
            temp.append(i)
    if len(temp) == 0:
        # 经过去停用词后，content变为空的情况去掉
        continue
    
    id_org_content[index] = v
    id_cuted_content[index] = temp
    content_cuted.append(temp)
    labels.append(l)

np.save('./data/id_org_content.npy', id_org_content)
np.save('./data/id_cuted_content.npy', id_cuted_content)
np.save('./data/content_cuted.npy', content_cuted)
np.save('./data/train_labels.npy', labels)


#  构建 Embedding Matrix
x = [len(v) for v in content_cuted]

# 长度 30 可以覆盖超过91%的动态
np.percentile(x, 91)

MAX_LEN = 30

# 将出现频率小的单词去除掉
# word_dict的结构是 {'单词': [单词新的标号, 出现次数]}
word_dict_raw = {'PADDING': [0, 999999]}
# word_dict_raw = dict()

for feed in content_cuted:
    for word in feed:
        if word in word_dict_raw:
            word_dict_raw[word][1] += 1
        else:
            word_dict_raw[word] = [len(word_dict_raw), 1]

word_dict = word_dict_raw
        

id_cuted_tokenizer = dict()
feed_words = []
# feed_words = [[0] * MAX_LEN]
for index, feed in id_cuted_content.items():
    word_id = []
    for word in feed:
        if word in word_dict:
            word_id.append(word_dict[word][0])
    word_id = word_id[:MAX_LEN]                                    # 截取。一个标题最多30个单词。
    feed_words.append(word_id + [0] * (MAX_LEN - len(word_id)))    # 填充。不足30个单词的标题，补0。
    id_cuted_tokenizer[index] = np.array(word_id + [0] * (MAX_LEN - len(word_id)), dtype='int32')
feed_words = np.array(feed_words, dtype='int32')

np.save('./data/train_texts_tokenizer.npy', feed_words)

vocab = list(word_dict.keys())
np.save('./data/vocab.npy', vocab)


# 读取中文的embedding vector模型
# 读取预训练好的中文的embedding vector模型
from gensim.models.keyedvectors import KeyedVectors

# sina word
word2vec_model = KeyedVectors.load_word2vec_format('../feed_text_classification/代码规整/utils/sgns.weibo.word', binary=False)

embedding_dict = {}

for k, v in word_dict.items():
    if k in word2vec_model:
        embedding_dict[k] = word2vec_model[k]
        
embedding_matrix = [0]*len(word_dict)    # embedding_matrix的用法是：输入单词的编号，得到单词的索引

for i in embedding_dict:
    embedding_matrix[word_dict[i][0]] = np.array(embedding_dict[i], dtype='float32')


words_list = list(word_dict.keys())
for i in range(len(embedding_matrix)):
    if type(embedding_matrix[i]) == int:                      # 袋外词处理:利用单词中每个字的平均处理
        embedding_matrix[i] = np.zeros(300, dtype='float32')
        word = words_list[i]
        for c in word:
            if c in word2vec_model:
                embedding_matrix[i] += word2vec_model[c]
        embedding_matrix[i] = embedding_matrix[i] / len(word)
        
embedding_matrix[0] = np.zeros(300, dtype='float32')
embedding_matrix = np.array(embedding_matrix, dtype='float32')

# 8662个词
np.save('./data/embedding_matrix.npy', embedding_matrix)
