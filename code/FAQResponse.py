#利用BERT模型做FAQ在喜得上的回應
from transformers import BertTokenizer
import torch
import pickle
import jieba
import sys
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW
jieba.load_userdict('Dataset/userDict.txt')

def toBertIds(q_input):
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q_input)))

def diff(listA,listB):
    retA = list(set(listA).intersection(set(listB)))
    return len(retA)>0

def predict(q_inputs):
    a = open('Dataset/anstype.txt', "r",encoding="utf-8")
    answers = a.readlines()
    a.close()

    pkl_file = open('Dataset/data_features_FAQ.pkl', 'rb')
    data_features = pickle.load(pkl_file)
    answer_dic = data_features['answer_dic']
    question_dic = data_features['question_dic']

    bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
    config = bert_config.from_pretrained('trained_model/config.json')
    model = bert_class.from_pretrained('trained_model/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'),config=config)
    model.eval()
    for q_input in q_inputs:
        bert_ids = toBertIds(q_input)

        assert len(bert_ids) <= 512
        input_ids = torch.LongTensor(bert_ids).unsqueeze(0)
        outputs = model(input_ids)

        predicts = outputs[:2]
        predicts = predicts[0]

        max_val = torch.max(predicts)
        label = (predicts == max_val).nonzero().numpy()[0][1]
        # input()
        # print(label)
        # print(answers[int(label)])
        return answers[int(label)]

if __name__ == "__main__":
    predict(q_inputs)
    