from DomainClassifier import *
from OrderResponse import *
from FAQResponse import *
import sys

predict_ans_dict=main()
domaintypeOne=[]
domaintypeZero=[]

for text,domaintype in predict_ans_dict.items():
    if(int(domaintype)==1):
        #類別是1者為點餐
        domaintypeOne.append(text)
    else:
        #類別是0者為FAQ
        domaintypeZero.append(text)
Query=domaintypeOne[0]
Order=mainOrder(Query)
print(domaintypeOne[0])
print(Order)
# FAQans=predict(domaintypeZero)
# print(domaintypeZero[0])
# print(FAQans)
