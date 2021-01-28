# -*- coding: utf-8 -*-
import jieba
import sys
import re
import json
from sentence_transformers import SentenceTransformer
import scipy.spatial
jieba.load_userdict("../text/itemdict.txt")
embedder = SentenceTransformer('bert-base-nli-mean-tokens')
# 套餐
meal_setList = ['一號餐','二號餐','三號餐','四號餐','五號餐','六號餐','七號餐','八號餐','九號餐','十號餐',
'1號餐','2號餐','3號餐','4號餐','5號餐','6號餐','7號餐','8號餐','9號餐','10號餐',
'11號餐','12號餐','13號餐','14號餐','15號餐','16號餐','17號餐','18號餐','19號餐',
'1號','2號','3號','4號','5號','6號','7號','8號','9號','10號',
'11號','12號','13號','14號','15號','16號','17號','18號','19號']
# 單點 
hito_mealsList = ['香蔥蛋椒鹽燒肉土司','香濃起司營養三明治', '丹麥DiDi卡', '花生培根蛋土司','花生培根蛋吐司', '醬燒豬排三明治', '雞肉棒', '鮪魚玉米夾蛋丹麥吐司','鮪魚玉米夾蛋丹麥土司', '黃金炸豬起司刈包',
'火星小薯餅', '雙倍香濃起司蛋土司','雙倍香濃起司蛋吐司', '丹麥鮮蔬薯泥', '招牌三明治', '泰式辣鬆蛋餅', '豬排蛋吐司', '香酥雞肉堡', '特級培根義大利麵', '起士土司','起士吐司',
'薯條', '起司蛋吐司', '奶酥吐司', '燒肉雞蛋丹麥吐司', '玉米蛋餅', '奶油餐包', '燒肉蛋吐司', '招牌', '里肌豬排三明治', '波浪薯條', '慢烤起司三明治',
 '原味蛋餅', '雞塊', '菜脯雞肉三明治', '火腿薯餅蛋土司','火腿薯餅蛋吐司', '黃金炸豬起司堡', '脆皮雞腿堡', '花生醬起司牛肉堡', '丹麥吉事蛋', '里肌豬肉三明治',
'起司蛋餅', '唐揚炸雞', '黃瓜雞蛋三明治', '香蔥蛋椒盤燒肉土司','香蔥蛋椒盤燒肉吐司', '牛肉丼拌麵', '甜不辣', '薯泥沙拉蛋土司','薯泥沙拉蛋吐司', '黃金炸豬義大利麵', '香蔥椒鹽燒肉堡',
'餐包', '菜脯蔥蛋土司','菜脯蔥蛋吐司', '火腿蛋餅', '黃金薯餅蛋餅', '香濃起司玉米蛋土司','香濃起司玉米蛋吐司', '四方薯餅', '吐司','土司', '花生豬排培根蛋三明治', '培根雞肉義大利麵',
'花枝炸物土司','花枝炸物吐司', '豬肉丼拌麵', '玉米起司蛋吐司','玉米起司蛋土司', '白吐司','白土司', '脆皮雞腿義大利麵', '火腿雞蛋堡', '肉鬆蔥花起司三明治', '培根蛋餅', 
'肉鬆薯泥土司','肉鬆薯泥吐司', '蘿蔔糕', '起司薯泥鮪魚蛋三明治', '菜脯蔥蛋餅', '薯泥起司蛋餅', '肉鬆蛋吐司','肉鬆蛋土司', '牛肉丼刈包', '豬排蛋餅', '沙茶燒肉拌麵',
'培根牛肉起司堡', '花枝天婦羅義大利麵', '椒麻燒肉拌麵', '椒麻拌麵', '香酥雙雞三明治', '烏龍湯麵', '花生薯餅蛋三明治', '沙茶拌麵',
'薯泥火腿蛋土司','薯泥火腿蛋吐司', '黃金脆薯', '泰式辣味雞腿堡', '卡啦雞堡', '豬肉丼刈包', '香辣雞腿肉蛋堡', '熱狗', '醬燒肉片三明治', '煎蛋蘿蔔糕',
'培根三明治', '特級培根蛋餅', '經典雞塊', 'DiDi卡三明治', '薯餅雞蛋堡', '鮮蔬雞蛋堡', '地瓜黃金球', '特級培根三明治','起司吐司','起司土司','蛋','起士蛋土司','起士蛋吐司','鮪魚蛋加起司吐司','鮪魚蛋加起司土司']
# regex_hito_mealsList=re.compile('.土司') 
# 飲料
drink_List = ['手工冬瓜茶', '豆漿鮮奶', '伯爵鮮奶', '伯爵紅茶', '鮮奶', '綠茶鮮奶', '招牌咖啡', '拿鐵', '紅茶鮮奶', '豆漿',
 '巧克力鮮奶', '摩卡咖啡', '豆漿紅茶', '冬瓜鮮奶', '拿鐵咖啡', '古早味紅茶', '伯爵紅', '茶', '茉香綠茶', '柳橙汁','鮮奶茶']
# 湯品 
soupList = ['玉米濃湯','雞肉濃湯','濃湯']
# 飲料溫度 
drink_temperatureList = ['熱的','熱','溫','冰','去冰','微冰','少冰','正常冰','多冰','常溫']
# 飲料糖份 
drink_sugarList = ['無糖','微糖','半糖','少糖','正常甜']
# 飲料容量 
drink_sizeList = ['大杯','小杯','中杯','L','l','M','m','S','s']
# 數量
quantityList = ['1片','2片','3片','4片','5片','6片','7片','8片','9片','10片','一片','二片','三片','四片','五片','六片','七片','八片','九片','十片','兩片',
'1個','2個','3個','4個','5個','6個','7個','8個','9個','10個','一個','二個','三個','四個','五個','六個','七個','八個','九個','十個','兩個',
'1份','2份','3份','4份','5份','6份','7份','8份','9份','10份','一份','二份','三份','四份','五份','六份','七份','八份','九份','十份','兩份',"一顆"
]
drink_quantity = ['1杯','2杯','3杯','4杯','5杯','6杯','7杯','8杯','9杯','10杯','一杯','二杯','三杯','四杯','五杯','六杯','七杯','八杯','九杯','十杯','兩杯']
# 動作 
meal_takeList = ['外帶自取','內用','外送','自取','外帶']
# 取餐日期 
order_dateList = ['今天','明天']
# 取餐時間 
order_timetext = ['等一下','早上']
#湯品大小
soup_size=['大份','小份']
#加點
hito_meals_plus=['加蛋']

Responses={'meal_takeList_meal':'好的，請問您要「內用」、「外帶自取」還是「外送」?','meal_takeList_hito':'好的，請問您要「內用」、「外帶自取」還是「外送」?'
,'drink_sizeList':'好的，請問您的飲料要「大杯」、「中杯」還是「小杯」?','drink_temperatureList':'好的，請問您的飲料要「冰的」、「溫的」、「熱的」還是「去冰」?'
,'drink_sugarList':'請問您的飲料甜度要「全糖」、「半糖」、「微糖」或是「無糖」?','meal_takeList_drink':'好的，請問您要「內用」、「外帶自取」還是「外送」'
,'soupSize':'好的，請問您的湯品要「大份」還是「小份」','meal_takeList_soup':'好的，請問您要「內用」、「外帶自取」還是「外送」'}
#BufferFiles
Jsonfilename="Slot.json"
Historyfilename="History.txt"
# 1 true 0 false
def processDescription(slot_result):
    temp_order = {}
    temp_order=slot_result.copy()
    return temp_order

def appendToFinalList(slot_result,order_finalList):
    # global temp_order,order_finalList
    temp_order=processDescription(slot_result)
    if temp_order!={}:
        if temp_order not in order_finalList:
            order_finalList.append(temp_order)
    temp_order={}
    return order_finalList

def printResult(slot_result,order_finalList):
    #global order_finalList,oreder_ls
    # 輸出所有結果(不過我是在cmd上直接print)，order_finalList type=list
    meal_order={'meal_setList':' ','quantityList_mealset':'1','meal_takeList_meal':' '}
    hito_meals={'hito_mealsList':' ','quantityList_hitoset':'1','hito_meals_plus':' ','meal_takeList_hito':' '}
    drink={'drink_List':' ','drink_quantity':'1','drink_sizeList':' ','drink_temperatureList':' ','drink_sugarList':' ','meal_takeList_drink':' '}
    soup={'soupList':' ','soupSize':' ','meal_takeList_soup':' '}
    order_time={'order_dateList':' ','order_timetext':' ','order_timeList':' '}

    # Response={'meal_order':' ','hito_meals':' ','drink':' ','soup':' ','order_time':' ','other':' '}
    
    order_finalList=appendToFinalList(slot_result,order_finalList)
    Response=DM(order_finalList,meal_order,hito_meals,drink,soup,order_time)
    
    return Response

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def ReadSlotFromJson(Jsonfilename):
    file = open(Jsonfilename, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        dic = json.loads(line)
        data.append(dic)
    return data

def WriteSlotToJson(Jsonfilename,Slot,truncate=False):
    with open(Jsonfilename,'a') as outfile:
        if(truncate==True):
            outfile.seek(0)
            outfile.truncate() 
            for idx in range(len(Slot)):
                json.dump(Slot[idx],outfile,ensure_ascii=False)
                outfile.write('\n')
        else:
            json.dump(Slot,outfile,ensure_ascii=False)     
            outfile.write('\n')

def ReadHistory(filename):
    fp = open(filename, "r")
    lines = fp.readlines() 
    fp.close()
    return lines[0]

def WriteHistory(filename,History):
    fp = open(filename, "w")
    fp.write(History)
    fp.close()

def DM(order_finalList,meal_order,hito_meals,drink,soup,order_time):  
    ls_meal_order=[]
    ls_hito_meals=[]
    ls_drink=[]
    ls_soup=[]
    ls_order_time=[]
    for order in order_finalList:
        if('meal_setList' in order.keys()):
            if(order['meal_setList']!=' ' and 'quantityList' in order):
                order["quantityList_mealset"]=order.pop("quantityList")
        if('hito_mealsList' in order.keys()):
            if(order['hito_mealsList']!=' 'and 'quantityList' in order):
                order["quantityList_hitoset"]=order.pop("quantityList")
        #設定對話的Dictionary
        for idx,(key,value) in enumerate(order.items()):
            if(idx!=0):
                break
            else:
                if(key in meal_order):
                    new_meal_order=merge_two_dicts(meal_order,order)
                    WriteSlotToJson(Jsonfilename,new_meal_order)
                    ls_meal_order.append(new_meal_order)# example:[{'hito_mealsList': '香蔥蛋椒鹽燒肉土司', 'quantityList_hitoset': '1份', 'hito_meals_plus': ' ', 'meal_takeList_hito': ' '}]
                elif(key in hito_meals):
                    new_hito_meals=merge_two_dicts(hito_meals,order)
                    WriteSlotToJson(Jsonfilename,new_hito_meals)
                    ls_hito_meals.append(new_hito_meals)
                elif(key in drink):
                    new_drink=merge_two_dicts(drink,order)
                    WriteSlotToJson(Jsonfilename,new_drink)
                    ls_drink.append(new_drink)
                elif(key in soup):
                    new_soup=merge_two_dicts(soup,order)
                    WriteSlotToJson(Jsonfilename,new_soup)
                    ls_soup.append(new_soup)
                elif(key in order_time):
                    new_order_time=merge_two_dicts(order_time,order)
                    WriteSlotToJson(Jsonfilename,new_order_time)
                    ls_order_time.append(new_order_time)
  
    if(ls_meal_order==[] and ls_hito_meals==[] and ls_drink==[] and ls_soup==[]):
        return '不好意思查無此菜單'
    else:
        return ' '
                           
def readfile():
    # 載入query
    file_Query = open("../text/query.txt", "r",encoding="utf-8")
    rQueryLines = file_Query.readlines()
    file_Query.close()
    return rQueryLines

def diff(listA,listB):
    #求交集的两种方式
    retA = [i for i in listA if i in listB]
    return retA

def mainResponse(bufferslot,Responses):
    for dic in bufferslot:
        for k,v in dic.items():
            if(v==" "):
                WriteHistory(Historyfilename,Responses[k])
                return Responses[k]
class getoutofloop(Exception): pass
def mainOrder(qline):
    bufferslot=ReadSlotFromJson(Jsonfilename)
    if(bufferslot!=[]):
        # History=list(ReadHistory(Historyfilename))
        # corpus_embeddings = embedder.encode(History)
        # qline=list(qline)
        # query_embeddings = embedder.encode(qline)
        # distances = scipy.spatial.distance.cdist(query_embeddings, corpus_embeddings, "cosine")[0][0]
        History=ReadHistory(Historyfilename)
        History_list = [t for t in jieba.cut(History, cut_all=False, HMM=True)]
        input("History_list")
        print(History_list)
        qline_list = [t for t in jieba.cut(qline, cut_all=False, HMM=True)]
        input("qline_list")
        print(qline_list)
        Intersection=diff(History_list,qline_list)
        input("Intersection")
        print(Intersection)
        try:
            if(("外帶" in qline_list) or ("自取"in qline_list)):
                for dic in bufferslot:
                        for k,v in dic.items():
                            if(v==" "):
                                if(("外帶" in qline_list)==True):
                                    dic[k]="外帶"
                                    raise getoutofloop()
                                elif(("自取"in qline_list)==True):
                                    dic[k]="自取"
                                    raise getoutofloop()
            else:
                if (len(Intersection)!=0):
                    for dic in bufferslot:
                        for k,v in dic.items():
                            if(v==" "):
                                dic[k]=Intersection[0]
                                raise getoutofloop()
        except getoutofloop:
            pass
        print(bufferslot)
        
        WriteSlotToJson(Jsonfilename,bufferslot,truncate=True)
        mainResponse(bufferslot,Responses)
    else:
        Response=RuleBased(qline)
        return Response
#####
def RuleBased(qline):
#def RuleBased(): 
    #rQueryLines=readfile()
    #先檢查是否有儲存slot資料     
    # for idx,qline in enumerate(rQueryLines):
    # 讀取每一行，去掉Q: 將逗號、換行、 白替換成底線，方便斷句用
    qline = qline.replace("Q：","")
    qline = qline.replace("，","_")
    qline = qline.replace(",","_")
    qline = qline.replace(" ","_")
    qline = qline.replace("\n","")
    qline = qline.replace("跟","_")
    qline = qline.replace("和","_")
    #qline = qline.replace("+","_")
    qline = qline.split("_")
    slot_result = {}
    order_finalList = []
    # 依照每一行的斷句分析slot，讀到重複的slot，先處理slot_result並存進order_finalList
    for part_of_qline in qline:
        if part_of_qline != '':
            noun_list = [t for t in jieba.cut(part_of_qline, cut_all=False, HMM=True)]#斷詞       
            for i,noun in enumerate(noun_list):
                # 讀到重複的slot，先處理slot_result並存進order_finalList
                # 再將新的資料填進slot
                if (noun in meal_setList):
                    if "meal_setList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["meal_setList"] = noun
                    
                elif (noun in hito_mealsList):
                    if "hito_mealsList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["hito_mealsList"] = noun

                elif (noun in drink_List):
                    if "drink_List" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["drink_List"] = noun
                    
                elif (noun in soupList):
                    if "soupList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["soupList"] = noun
                    
                elif (noun in drink_temperatureList):
                    if "drink_temperatureList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["drink_temperatureList"] = noun

                elif (noun in drink_sugarList):
                    if "drink_sugarList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["drink_sugarList"] = noun

                elif (noun in drink_sizeList):
                    if "drink_sizeList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["drink_sizeList"] = noun

                elif (noun in drink_quantity):
                    if "drink_quantity" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["drink_quantity"] = noun

                elif (noun in meal_takeList):
                    if "meal_takeList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                    slot_result["meal_takeList"] = noun
                    
                elif (noun in order_dateList):
                    if "order_dateList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["order_dateList"] = noun

                elif (noun in quantityList):
                    if "quantityList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["quantityList"] = noun

                elif (noun.isdigit()):
                    if "order_timeList" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        continue
                    #處理句子中數字但是判斷前後是否有:會超過範圍
                    elif(((i-1) not in range(len(noun_list))) or ((i+1) not in range(len(noun_list)))):
                        # print("處理句子中數字但是判斷前後是否有:會超過範圍")
                        if(((i-1) not in range(len(noun_list))) and noun_list[i+1]!=':'):
                            if "quantityList" in slot_result.keys():
                                appendToFinalList(slot_result,order_finalList)
                            slot_result["quantityList"] = noun
                        elif(((i+1) not in range(len(noun_list))) and noun_list[i-1]!=':'):
                            if "quantityList" in slot_result.keys():
                                appendToFinalList(slot_result,order_finalList)                       
                            slot_result["quantityList"] = noun
                        else:
                            slot_result["order_timeList"]= noun+noun_list[i+1]+noun_list[i+2]
                    #處理句子中數字但是並非時間中的數值
                    elif(noun_list[i+1]!=':' and noun_list[i-1]!=':' and ((i+1) in range(len(noun_list))) and ((i-1) in range(len(noun_list)))):
                        #print("處理句子中數字但是並非時間中的數值")
                        if "quantityList" in slot_result.keys():
                            appendToFinalList(slot_result,order_finalList)
                        slot_result["quantityList"] = noun
                    #處理句子中數字並且是時間中的數值透過:去判斷
                    else:  
                        #print("處理句子中數字並且是時間中的數值透過:去判斷")
                        slot_result["order_timeList"]= noun+noun_list[i+1]+noun_list[i+2]

                elif (noun in order_timetext):
                    if "order_timetext" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["order_timetext"] = noun

                elif (noun in soup_size):
                    if "soup_size" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["soup_size"] = noun
                elif (noun in hito_meals_plus):
                    if "hito_meals_plus" in slot_result.keys():
                        appendToFinalList(slot_result,order_finalList)
                        slot_result = {}
                    slot_result["hito_meals_plus"] = noun

            appendToFinalList(slot_result,order_finalList)
            slot_result = {}
    Response=printResult(slot_result,order_finalList)
    
    if(Response!=" "):
        return Response
    else:
        bufferslot=ReadSlotFromJson(Jsonfilename)
        mainResponse(bufferslot,Responses)
                    
# if __name__ == '__main__':
#     RuleBased()