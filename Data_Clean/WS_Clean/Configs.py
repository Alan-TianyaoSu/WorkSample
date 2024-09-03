import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
import random
import math
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import os
import pymongo
from urllib import parse
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
from pymongo.errors import AutoReconnect, OperationFailure
import json
import logging
from bson import ObjectId
import ast
import regex
import difflib
import traceback
import time
import logging
from Word_Mapping import *
from City_Province import *
from Config_Functions import *


# 增加逻辑需要在表头添加相应字段

# 表头配置
Writ_Col = ['_id','标题','判决日期',
                '审理法院','检察机关', '相关人员', '申请金额', '实际判决金额','原告人', '被告人',
                '审理经过','诉称','法院查明', '地区', '原告诉称', '被告辩称',
                '本院认为','判决结果','落款', '诉称与法院查明','s23','法官姓名', '是否公开',
                
                # 已有
                '案号', '法官名字', '案件名', '裁判机构', '裁判日期', '案件类型', '文书类型', '法院判决主文',
                
                # 一期
                '起诉/提出执行申请日期','受理日期','审理阶段','关联案例案号','当事人','被告类型',
                '提出回购请求所依据的文件','提出回购请求所依据的条款','判决支持情况','裁判地区','案由', 

                # 二期
                '诉讼请求金额','法院实际判决金额','申请执行的金额','执行成功的金额',
                '被告类型','一审判决支持情况','二审判决支持情况','被告数量','诉讼请求',

                # 三期
                '判决支持总金额比率','执行成功金额比率','诉讼请求与实际执行到位金额比率','处理案件所需时间','诉讼请求金额','法院实际判决金额',
                '申请执行的金额','执行成功的金额','判决支持回购款及利息金额比率',

                # 四期
                '裁判地区','处理案件所需时间','判决支持回购款及利息金额比率','判决支持总金额比率','执行成功金额比率','诉讼请求与实际执行到位金额比率'


                ]

# 结果列
Result_Col = ['_id','标题','判决日期', 
                '审理法院','检察机关', '相关人员', '原告人', '被告人', '申请金额', '实际判决金额',
                '审理经过','原告诉称', '被告辩称', '法院查明','地区',
                '本院认为','判决结果','落款','法官姓名', '是否公开',
                
                # 已有
                '案号', '法官名字', '案件名', '裁判机构', '裁判日期', '案件类型', '文书类型', '法院判决主文',
                
                # 一期
                '起诉/提出执行申请日期','受理日期','审理阶段','关联案例案号','当事人','被告类型',
                '提出回购请求所依据的文件','提出回购请求所依据的条款','判决支持情况','裁判地区','案由', 

                # 二期
                '诉讼请求金额','法院实际判决金额','申请执行的金额','执行成功的金额',
                '被告类型','一审判决支持情况','二审判决支持情况','被告数量','诉讼请求',

                # 三期
                '判决支持总金额比率','执行成功金额比率','诉讼请求与实际执行到位金额比率','处理案件所需时间','诉讼请求金额','法院实际判决金额',
                '申请执行的金额','执行成功的金额','判决支持回购款及利息金额比率',

                # 四期
                '裁判地区','处理案件所需时间','判决支持回购款及利息金额比率','判决支持总金额比率','执行成功金额比率','诉讼请求与实际执行到位金额比率'
                
                ]



# Result_Col = ['案号', '申请金额', '实际判决金额','原告诉称', '诉称']


# 两种匹配方式,两次筛选
pattern_1 = re.compile('<divid=\'2\'[^>]*>(.*?)<\/div>\s*<divid=')
pattern_2 = re.compile('(?:^|>)(?:(?!代理|辩护)[^<\s])+(?:<|$)')
pattern_3 = re.compile('right.*?</div>(.*?)right', flags=re.DOTALL)


pattern_4 = re.compile('right.*?>.*?<.*?>(.*)<.*?right', flags=re.DOTALL)
pattern_5 = re.compile('right.*?>.*?<.*?>(.*)', flags=re.DOTALL)

# 备用的文本清洗规则
replace_rules = {'Ｘ':'x', 'ｘ':'x', '×':'&times;', '·':'&middot;', 
                    " ":'', '\'':'', '[':'',']':'', '、':','}

fields = {'_id': 1,'s1': 1,'s7': 1,'s8': 1,'s9': 1,'s11': 1,'s31': 1,'s22': 1,'s17': 1,
        's23': 1,'s25': 1,'s26': 1,'s27': 1,'s28': 1,'qwContent': 1}


# 服务器配置
sever_ip = 'oldmongo.fqyai.cn:27017/admin'
user_name = parse.quote_plus("root")
password = parse.quote_plus("123QAZwsx")


Client = Login(sever_ip, user_name, password)
last_processed_collection, last_processed_position, Progress = None,None,None
Cursor = None
IfChange = 0; IfEnd = 0

# 文档配置，默认值
size = 10
year = '2014'
total_progress = 100
# 基础配置
Original_Data, Length = None, None
Datatable, cleaned_html = None, None
executor = ThreadPoolExecutor(max_workers=1)
last_processed_collection, last_processed_position, Progress = None, None, None
Cursor = None
logger = None

