'''
仅执行一次的函数
'''

import pymongo
from bson import ObjectId
import os
# 登录
def Login( IP, USN, PSW):
    Client = pymongo.MongoClient("mongodb://{0}:{1}@{2}".format(USN, PSW, IP))
    Client.server_info()
    if Client.server_info():
        print('数据库连接完成')
        return Client
    else:
        print("无法连接到MongoDB服务器")

# 加载进度
def load_progress(year):
    try:
        Pro_directory = "Progress"
        filename = os.path.join(Pro_directory, 'progress_' + year[-2:] + '.txt')
        # print(year)
        with open(filename, 'r') as file:
            content = file.read().strip().split(',')
            collection_name = content[0]
            current_position = content[1]
            current_Progress = int(content[2])
        return collection_name, current_position, current_Progress
    except FileNotFoundError:
        return 'ws_'+ year, 0, 0
    
# 初始化游标
def Initialize_Cursor(Client, last_processed_collection, last_processed_position, size, fields):
    db = Client['cpws']
    if last_processed_position == 0:
        Cursor = db[last_processed_collection].find({},fields).sort("_id", pymongo.ASCENDING).limit(size)
    else:
        Cursor = db[last_processed_collection].find({"_id": {"$gt": ObjectId(last_processed_position)}},fields).sort("_id", pymongo.ASCENDING).limit(size)
    return Cursor


