from transformers import BertTokenizer
import torch
import pickle
import jieba
import sys
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW
jieba.load_userdict('Dataset/userDict.txt')

def toBertIds(tokenizer,q_input):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q_input)))

def diff(listA,listB):
    retA = list(set(listA).intersection(set(listB)))
    return len(retA)>0

def main():
    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    pkl_file = open('Dataset/data_features_domain.pkl', 'rb')
    data_features = pickle.load(pkl_file)
    answer_dic = data_features['answer_dic']
    question_dic = data_features['question_dic']
   
    bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
    config = bert_config.from_pretrained('/share/nas165/Wendy/EZAIDialogue/DomainClassification_FAQ/trained_model/config.json')
    model = bert_class.from_pretrained('/share/nas165/Wendy/EZAIDialogue/DomainClassification_FAQ/trained_model/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'),config=config)
    model.eval()
    
    #讀取驗證資料問題集(user query)
    q = open('Dataset/Query_Test/question_test.txt', "r",encoding="utf-8")
    q_inputs = q.readlines()
    q.close()

    #讀取驗證資料回答集(ans of query)
    a = open('Dataset/Test_Label/DomainLabelForTest.txt', "r",encoding="utf-8")
    answer = a.readlines()
    a.close()
    
    predict_ans_dict={}
    predict_ans=[]
    q_inputs=['我要買1份香蔥蛋椒鹽燒肉土司加蛋和伯爵紅茶']
    for q_input in q_inputs:
        cutQ=jieba.lcut(q_input)
        #keywordls=["我想訂","號餐","我想要","內用","我要","我要訂","一份","一個"]
        #keywordls=["我想吃"]
        bert_ids = toBertIds(tokenizer,q_input)        
        assert len(bert_ids) <= 512
        input_ids = torch.LongTensor(bert_ids).unsqueeze(0)
        
        outputs = model(input_ids)

        predicts = outputs[:2]
        predicts = predicts[0]
     
        max_val = torch.max(predicts)
        label = (predicts == max_val).nonzero().numpy()[0][1]     
        # if(diff(keywordls,cutQ)):
        #     label=0
        predict_ans.append(label)
        predict_ans_dict[q_input.replace("\n", "")]=label
    return predict_ans_dict
    # count=0
    # fp = open("log2.txt", "a")
    # for q,a,p in zip(q_inputs,answer,predict_ans):
    #     if(int(a)==int(p)):
    #         count+=1
    #     else:
    #         fp.write(str(q)+" "+str(p))
    # fp.close()
    # acc=(count/len(answer))*100
    # print("accuracy:",acc)

if __name__ == "__main__":
    predict_ans=main()
    