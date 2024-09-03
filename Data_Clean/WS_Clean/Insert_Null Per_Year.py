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
import socket
from pymongo.errors import DuplicateKeyError


disputes = [
    '合同、无因管理、不当得利纠纷',
    '合同纠纷',
    '缔约过失合同纠纷',
    '确认合同效力纠纷',
    '确认合同有效纠纷',
    '确认合同无效纠纷',
    '债权人代位权纠纷',
    '债权人撤销权纠纷',
    '债权转让合同纠纷',
    '债务转移合同纠纷',
    '债权债务概括转移合同纠纷',
    '悬赏广告纠纷',
    '建设用地使用权合同纠纷',
    '买卖合同纠纷',
    '分期付款买卖合同纠纷',
    '凭样品买卖合同纠纷',
    '试用买卖合同纠纷',
    '互易纠纷',
    '国际货物买卖合同纠纷',
    '网络购物合同纠纷',
    '电视购物合同纠纷',
    '招标投标买卖合同纠纷',
    '拍卖合同纠纷',
    '建设用地使用权出让合同纠纷',
    '建设用地使用权转让合同纠纷',
    '临时用地合同纠纷',
    '探矿权转让合同纠纷',
    '采矿权转让合同纠纷',
    '房地产开发经营合同纠纷',
    '委托代建合同纠纷',
    '合资、合作开发房地产合同纠纷',
    '项目转让合同纠纷',
    '房屋买卖合同纠纷',
    '商品房预约合同纠纷',
    '商品房预售合同纠纷',
    '商品房销售合同纠纷',
    '商品房委托代理销售合同纠纷',
    '经济适用房转让合同纠纷',
    '农村房屋买卖合同纠纷',
    '房屋拆迁安置补偿合纠纷',
    '供用电合同纠纷',
    '供用水合同纠纷',
    '供用气合同纠纷',
    '供用热力合同纠纷',
    '赠与合同纠纷',
    '公益事业捐赠合同纠纷',
    '附义务赠与合同纠纷',
    '借款合同纠纷',
    '金融借款合同纠纷',
    '同业拆借纠纷',
    '企业借贷纠纷',
    '民间借贷纠纷',
    '小额借款合同纠纷',
    '金融不良债权转让合同纠纷',
    '金融不良债权追偿纠纷',
    '保证合同纠纷',
    '抵押合同纠纷',
    '质押合同纠纷',
    '定金合同纠纷',
    '进出口押汇纠纷',
    '储蓄存款合同纠纷',
    '银行卡纠纷',
    '借记卡纠纷',
    '信用卡纠纷',
    '租赁合同纠纷',
    '土地租赁合同纠纷',
    '房屋租赁合同纠纷',
    '车辆租赁合同纠纷',
    '建筑设备租赁合同纠纷',
    '融资租赁合同纠纷',
    '承揽合同纠纷',
    '加工合同纠纷',
    '定作合同纠纷',
    '修理合同纠纷',
    '复制合同纠纷',
    '测试合同纠纷',
    '检验合同纠纷',
    '铁路机车、车辆建造合同纠纷',
    '建设工程合同纠纷',
    '建设工程勘察合同纠纷',
    '建设工程设计合同纠纷',
    '建设工程施工合同纠纷',
    '建设工程价款优先受偿权纠纷',
    '建设工程分包合同纠纷',
    '建设工程监理合同纠纷',
    '装饰装修合同纠纷',
    '铁路修建合同纠纷',
    '农村建房施工合同纠纷',
    '运输合同纠纷',
    '公路旅客运输合同纠纷',
    '公路货物运输合同纠纷',
    '水路旅客运输合同纠纷',
    '水路货物运输合同纠纷',
    '航空旅客运输合同纠纷',
    '航空货物运输合同纠纷',
    '出租汽车运输合同纠纷',
    '管道运输合同纠纷',
    '城市公交运输合同纠纷',
    '联合运输合同纠纷',
    '多式联运合同纠纷',
    '铁路货物运输合同纠纷',
    '铁路旅客运输合同纠纷',
    '铁路行李运输合同纠纷',
    '铁路包裹运输合同纠纷',
    '国际铁路联运合同纠纷',
    '保管合同纠纷',
    '仓储合同纠纷',
    '委托合同纠纷',
    '进出口代理合同纠纷',
    '货运代理合同纠纷',
    '民用航空运输销售代理合同纠纷',
    '诉讼、仲裁、人民调解代理合同纠纷',
    '委托理财合同纠纷',
    '金融委托理财合同纠纷',
    '民间委托理财合同纠纷',
    '行纪合同纠纷',
    '居间合同纠纷',
    '补偿贸易纠纷',
    '借用合同纠纷',
    '典当纠纷',
    '合伙协议纠纷',
    '种植、养殖回收合同纠纷',
    '彩票、奖券纠纷',
    '中外合作勘探开发自然资源合同纠纷',
    '农业承包合同纠纷',
    '林业承包合同纠纷',
    '渔业承包合同纠纷',
    '牧业承包合同纠纷',
    '农村土地承包合同纠纷',
    '土地承包经营权转包合同纠纷',
    '土地承包经营权转让合同纠纷',
    '土地承包经营权互换合同纠纷',
    '土地承包经营权入股合同纠纷',
    '土地承包经营权抵押合同纠纷',
    '土地承包经营权出租合同纠纷',
    '服务合同纠纷',
    '电信服务合同纠纷',
    '邮寄服务合同纠纷',
    '医疗服务合同纠纷',
    '法律服务合同纠纷',
    '旅游合同纠纷',
    '房地产咨询合同纠纷',
    '房地产价格评估合同纠纷',
    '酒店服务合同纠纷',
    '财会服务合同纠纷',
    '餐饮服务合同纠纷',
    '娱乐服务合同纠纷',
    '有线电视服务合同纠纷',
    '网络服务合同纠纷',
    '教育培训合同纠纷',
    '物业服务合同纠纷',
    '家政服务合同纠纷',
    '庆典服务合同纠纷',
    '殡葬服务合同纠纷',
    '农业技术服务合同纠纷',
    '农机作业服务合同纠纷',
    '保安服务合同纠纷',
    '银行结算合同纠纷',
    '演出合同纠纷',
    '劳务合同纠纷',
    '离退休人员返聘合同纠纷',
    '广告合同纠纷',
    '展览合同纠纷',
    '追偿权纠纷',
    '不当得利纠纷',
    '无因管理纠纷'
]

Source_ip = 'oldmongo.fqyai.cn:27017/admin'
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
Base = 'ws_'


# 只在这里更改
################
Year = '2020'
size = 3000
END = 30000
################

# 同时打开关闭
#####################

# File_Name = 'Check_' + Year + '.txt'
# with open(File_Name, 'r') as file:
#     last_processed_position = file.read().split(',')
#     last_processed_position = ObjectId(last_processed_position)

#####################

def increment_year(year_str):
    year_int = int(year_str)  
    incremented_year = year_int + 1  
    return str(incremented_year)  

def contains_dispute(s11_items):
    # 检查s11_items是否是列表，如果是，则进行迭代比较
    if isinstance(s11_items, list):
        return any(item in disputes for item in s11_items)
    return False

Flag = 0
while True:
    Source_Base_Name = Base + Year
    Source_collection = Source_DB[Source_Base_Name]

    try:
        start_time = time.time()
        while True:
            start_time_1 = time.time()
            # 获取数据
            if last_processed_position == 0:
                Cursor = Source_collection.find({}).sort("_id", pymongo.ASCENDING).limit(size)
            else:
                Cursor = Source_collection.find({"_id": {"$gt": ObjectId(last_processed_position)}}).sort("_id",
                                                                                                          pymongo.ASCENDING).limit(
                    size)
            end_1 = time.time()
            print(f"这批数据获取时间：{end_1 - start_time_1} 秒")
            # 直接将游标转换为DataFrame
            document_df = pd.DataFrame(list(Cursor))
            end_2 = time.time()
            print(f"这批数据转换DataFrame时间：{end_2 - start_time_1} 秒")

            Flag += document_df.shape[0]
            if document_df.empty or Flag > END:
                print(1/0)
                break

            # 进度点保存
            last_processed_position = document_df["_id"].iloc[-1]


            # 数据清洗
            document_df = document_df[document_df['s11'].apply(contains_dispute)]
            end_3 = time.time()
            print(f"这批数据清洗时间：{end_3 - end_2} 秒")

            # 插入DB
            if not document_df.empty:
                Target_Base_Name = 'contract_unjust_enrichment_gratuitous_management_' + Year
                Target_collection = Target_DB[Target_Base_Name]
                for doc in document_df.to_dict('records'):
                    try:
                        Target_collection.insert_one(doc)
                    except DuplicateKeyError:
                        print("忽略主键冲突，继续插入下一条记录")
            end_4 = time.time()
            print(f"这批数据插入数据库时间：{end_4 - end_3} 秒")

            # 进度点保存
            fileName = 'Check_' + Year + '.txt'
            with open(fileName, 'w') as file:
                file.write(str(last_processed_position))

        end = time.time()
        print(f"插入总时间：{end - start_time} 秒")
    
    # 跑完了一张表
    except ZeroDivisionError as e:
        print(Year,' 年数据处理完成')
        break

    # 服务器异常
    except socket.timeout as e:
        print("捕获到 socket.timeout 异常:", e)
        # Client 重连
        Source_Client = pymongo.MongoClient("mongodb://{0}:{1}@{2}".format(Source_user_name, Source_password, Source_ip))
        Source_DB = Source_Client['cpws']
        Target_Client = pymongo.MongoClient("mongodb://{0}:{1}@{2}".format(Target_user_name, Target_password, Target_ip))
        Target_DB = Target_Client['cpws']

    except Exception as e:
        print(Year, ' 发生错误 ！！')
        print(e,'\n')
        fileName = 'Error_Check' + Year + '.txt'
        with open(fileName, 'a') as file:
            file.write(str(last_processed_position) + '\n')
