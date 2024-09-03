import pandas as pd
import pymongo
from urllib import parse
from concurrent.futures import ThreadPoolExecutor
from pymongo.errors import AutoReconnect, OperationFailure
from bson import ObjectId
from pymongo import MongoClient, InsertOne
from bson.objectid import ObjectId
import random
import time


disputes = [
    '与企业有关的纠纷',
    '企业出资人权益确认纠纷',
    '侵害企业出资人权益纠纷',
    '企业公司制改造合同纠纷',
    '企业股份合作制改造合同纠纷',
    '企业债权转股权合同纠纷',
    '企业分立合同纠纷',
    '企业租赁经营合同纠纷',
    '企业出售合同纠纷',
    '挂靠经营合同纠纷',
    '企业兼并合同纠纷',
    '联营合同纠纷',
    '企业承包经营合同纠纷',
    '中外合资经营企业承包经营合同纠纷',
    '中外合作经营企业承包经营合同纠纷',
    '外商独资企业承包经营合同纠纷',
    '乡镇企业承包经营合同纠纷',
    '中外合资经营企业合同纠纷',
    '中外合作经营企业合同纠纷',
    '与公司有关的纠纷',
    '股东资格确认纠纷',
    '股东名册记载纠纷',
    '请求变更公司登记纠纷',
    '股东出资纠纷',
    '新增资本认购纠纷',
    '股东知情权纠纷',
    '请求公司收购股份纠纷',
    '股权转让纠纷',
    '公司决议纠纷',
    '公司决议效力确认纠纷',
    '公司决议撤销纠纷',
    '公司设立纠纷',
    '公司证照返还纠纷',
    '发起人责任纠纷',
    '公司盈余分配纠纷',
    '损害股东利益责任纠纷',
    '损害公司利益责任纠纷',
    '股东损害公司债权人利益责任纠纷',
    '公司关联交易损害责任纠纷',
    '公司合并纠纷',
    '公司分立纠纷',
    '公司减资纠纷',
    '公司增资纠纷',
    '公司解散纠纷',
    '申请公司清算',
    '清算责任纠纷',
    '上市公司收购纠纷',
    '合伙企业纠纷',
    '入伙纠纷',
    '退伙纠纷',
    '合伙企业财产份额转让纠纷',
    '与破产有关的纠纷',
    '申请破产清算',
    '申请破产重整',
    '申请破产和解',
    '请求撤销个别清偿行为纠纷',
    '请求确认债务人行为无效纠纷',
    '对外追收债权纠纷',
    '追收未缴出资纠纷',
    '追收抽逃出资纠纷',
    '追收非正常收入纠纷',
    '破产债权确认纠纷',
    '职工破产债权确认纠纷',
    '普通破产债权确认纠纷',
    '取回权纠纷',
    '一般取回权纠纷',
    '出卖人取回权纠纷',
    '破产抵销权纠纷',
    '别除权纠纷',
    '破产撤销权纠纷',
    '损害债务人利益赔偿纠纷',
    '管理人责任纠纷',
    '证券纠纷',
    '证券权利确认纠纷',
    '股票权利确认纠纷',
    '公司债券权利确认纠纷',
    '国债权利确认纠纷',
    '证券投资基金权利确认纠纷',
    '证券交易合同纠纷',
    '股票交易纠纷',
    '公司债券交易纠纷',
    '国债交易纠纷',
    '证券投资基金交易纠纷',
    '金融衍生品种交易纠纷',
    '证券承销合同纠纷',
    '证券代销合同纠纷',
    '证券包销合同纠纷',
    '证券投资咨询纠纷',
    '证券资信评级服务合同纠纷',
    '证券回购合同纠纷',
    '股票回购合同纠纷',
    '国债回购合同纠纷',
    '公司债券回购合同纠纷',
    '证券投资基金回购合同纠纷',
    '质押式证券回购纠纷',
    '证券上市合同纠纷',
    '证券交易代理合同纠纷',
    '证券上市保荐合同纠纷',
    '证券发行纠纷',
    '证券认购纠纷',
    '证券发行失败纠纷',
    '证券返还纠纷',
    '证券欺诈责任纠纷',
    '证券内幕交易责任纠纷',
    '操纵证券交易市场责任纠纷',
    '证券虚假陈述责任纠纷',
    '欺诈客户责任纠纷',
    '证券托管纠纷',
    '证券登记、存管、结算纠纷',
    '融资融券交易纠纷',
    '客户交易结算资金纠纷',
    '期货交易纠纷',
    '期货经纪合同纠纷',
    '期货透支交易纠纷',
    '期货强行平仓纠纷',
    '期货实物交割纠纷',
    '期货保证合约纠纷',
    '期货交易代理合同纠纷',
    '侵占期货交易保证金纠纷',
    '期货欺诈责任纠纷',
    '操纵期货交易市场责任纠纷',
    '期货内幕交易责任纠纷',
    '期货虚假信息责任纠纷',
    '信托纠纷',
    '民事信托纠纷',
    '营业信托纠纷',
    '公益信托纠纷',
    '保险纠纷',
    '财产保险合同纠纷',
    '财产损失保险合同纠纷',
    '责任保险合同纠纷',
    '信用保险合同纠纷',
    '保证保险合同纠纷',
    '保险人代位求偿权纠纷',
    '人身保险合同纠纷',
    '人寿保险合同纠纷',
    '意外伤害保险合同纠纷',
    '健康保险合同纠纷',
    '再保险合同纠纷',
    '保险经纪合同纠纷',
    '保险代理合同纠纷',
    '进出口信用保险合同纠纷',
    '保险费纠纷',
    '票据纠纷',
    '票据付款请求权纠纷',
    '票据追索权纠纷',
    '票据交付请求权纠纷',
    '票据返还请求权纠纷',
    '票据损害责任纠纷',
    '票据利益返还请求权纠纷',
    '汇票回单签发请求权纠纷',
    '票据保证纠纷',
    '确认票据无效纠纷',
    '票据代理纠纷',
    '票据回购纠纷',
    '信用证纠纷',
    '委托开立信用证纠纷',
    '信用证开证纠纷',
    '信用证议付纠纷',
    '信用证欺诈纠纷',
    '信用证融资纠纷',
    '信用证转让纠纷'
]


Source_ip = '192.168.11.248:27017/admin'
Source_user_name = parse.quote_plus("root")
Source_password = parse.quote_plus("123QAZwsx")

Source_Client = pymongo.MongoClient("mongodb://{0}:{1}@{2}".format(Source_user_name, Source_password, Source_ip))
Source_Client.server_info()
if Source_Client.server_info():
    print('源数据库连接完成')
else:
    print("无法连接到MongoDB服务器")

Source_DB = Source_Client['cpws']


Target_ip = 'mongo.fqyai.cn:27017/cpws'
Target_user_name = parse.quote_plus("mongo")
Target_password = parse.quote_plus("faqianyan2024")

Target_Client = pymongo.MongoClient("mongodb://{0}:{1}@{2}".format(Target_user_name, Target_password, Target_ip))
Target_Client.server_info()
if Target_Client.server_info():
    print('目标数据库连接完成')
else:
    print("无法连接到MongoDB服务器")

Target_DB = Target_Client['cpws']


last_processed_position = 0
last_db = '2014'
Dispute = '与企业有关的纠纷'

Base = 'ws_'
size = 10000

with open('Check.txt', 'r') as file:
    [last_db, last_processed_position] = file.read().split(',')
    last_processed_position = ObjectId(last_processed_position)


END = 99999999999999
Flag = 0

while last_db <= '2023':
    
    Source_Base_Name = Base + last_db
    Source_collection = Source_DB[Source_Base_Name]
    
    try:
        start_time = time.time()
        while True:
            
            # 获取数据
            if last_processed_position == 0:
                Cursor = Source_collection.find({}).sort("_id", pymongo.ASCENDING).limit(size)
            else:
                Cursor = Source_collection.find({"_id": {"$gt": ObjectId(last_processed_position)}}).sort("_id", pymongo.ASCENDING).limit(size)

            # 判断DB读取完成
            document_list = list(Cursor)
            Flag += len(document_list)
            if len(document_list) == 0 or Flag > END:
                print(last_db, ' 完成')
                last_db = last_db[:-1] + str(int(last_db[-1]) + 1)
                last_processed_position = 0
                break
            
            #进度点保存
            document_dict = pd.DataFrame(document_list)
            last_processed_position = document_dict["_id"].iloc[-1]
            with open('Check.txt', 'w') as file:
                file.write(last_db + ',' + str(last_processed_position))

            document_dict = document_dict[pd.notnull(document_dict['s11'])]
            document_dict['s11'] = document_dict['s11'].apply(lambda x: None if len(x) == 0 else x)

            document_dict = document_dict[pd.notnull(document_dict['s11'])]

            document_dict['s11'] = document_dict['s11'].apply(lambda x: x if any(item in disputes for item in x) else None)
            document_dict = document_dict[pd.notnull(document_dict['s11'])]
            
            document_list = document_dict.to_dict('records')

            # 插入DB
            Target_Base_Name = 'company_ securities_ insurance_ bill_' + last_db
            Target_collection = Target_DB[Target_Base_Name]
            operations = [InsertOne(doc) for doc in document_list]
            if operations:
                Target_collection.bulk_write(operations)
        
        end = time.time()
        print(f"插入时间：{end - start_time} 秒")

    # 报错时切换DB
    except Exception as e:
        print(last_db,' 发生错误 ！！')
        last_db = last_db[:-1] + str(int(last_db[-1]) + 1)
        last_processed_position = 0
        with open('Error_Check.txt', 'w') as file:
            file.write(last_db)
    
    if Flag > END:
        break

    # 更换DB时保存进度点
    with open('Check.txt', 'w') as file:
        file.write(last_db + ',' + str(last_processed_position))

        



