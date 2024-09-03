from Configs import *
from Algorithm import *

def Initial(year, size):
    global last_processed_collection, last_processed_position, Progress, Cursor
    last_processed_collection, last_processed_position, Progress = load_progress(year)
    Cursor = Initialize_Cursor(Client, last_processed_collection, last_processed_position, size, fields)
    return Progress

# 定位游标
def Locate_Cursor(size):
    global Cursor, fields
    Cursor = Client['cpws'][last_processed_collection]\
        .find({"_id": {"$gt": last_processed_position}},fields).sort("_id", pymongo.ASCENDING).limit(size)

# 检查空列表
def check_empty_or_nan(x):
    if isinstance(x, list):
        return len(x) == 0
    else:
        return pd.isnull(x)

    
# 服务器文件读取方式    
def New_Data(size):
    global last_processed_position, IfEnd, Progress
    processed_count = 0
    data_list = list(Cursor)
    all_data = pd.DataFrame(data_list)
    all_data['s11'] = all_data['s11'].apply(lambda x: ['未知'] if check_empty_or_nan(x) else x)
    Progress += len(data_list); processed_count = len(data_list)
    last_processed_position = data_list[-1]["_id"]
    all_data["_id"] = all_data["_id"].apply(lambda x: str(x))
    if processed_count < size:
        IfEnd = 1
    Locate_Cursor(size)
    return all_data, len(data_list), Progress

# 获取新数据
def Get_Data(size):
    global Original_Data, Datatable
    Original_Data, Length, p = New_Data(size)
    Datatable = New_Table(Length)
    return p

# 创建清洗结果表的结构
def New_Table(Len):
    global cleaned_html
    cleaned_html = pd.DataFrame(index=range(Len), columns = ['text'])
    return pd.DataFrame(index=range(Len), columns = Writ_Col)

# 重置下一轮
def Refresh():
    global Datatable, Original_Data
    Datatable = None; Original_Data = None

def remove_newlines_without_period(text):
    return re.sub(r'(?<!。)\n', '', text)

# 表格克隆
def Clone_table():
    global Datatable, Original_Data
    Datatable['_id'] = np.where(pd.isnull(Original_Data['_id']), '无', Original_Data['_id'])
    Datatable['标题'] = np.where(pd.isnull(Original_Data['s1']), '无', Original_Data['s1'])
    Datatable['案号'] = np.where(pd.isnull(Original_Data['s7']), '无', Original_Data['s7'])
    Datatable['判决日期'] = np.where(pd.isnull(Original_Data['s31']), '无', Original_Data['s31'])
    vectorized_func = np.vectorize(remove_newlines_without_period)
    Datatable['本院认为'] = vectorized_func(np.where(pd.isnull(Original_Data['s26']), '无', Original_Data['s26']))
    Datatable['判决结果'] = vectorized_func(np.where(pd.isnull(Original_Data['s27']), '无', Original_Data['s27']))
    Datatable['落款'] = np.where(pd.isnull(Original_Data['s28']), '无', Original_Data['s28'])
    Datatable['案由'] = Original_Data['s11'].apply(lambda x: [Classification_mapping.get(i, i) for i in x])
    # Datatable['案由'] = np.where(pd.isnull(Datatable['案由']), '无', Datatable['案由'])
    Datatable['是否公开'] = np.where(pd.isnull(Original_Data['qwContent']), '信息公开', '文书公开')
    Datatable['s23'] = np.where(pd.isnull(Original_Data['s23']), '无', Original_Data['s23'])
    Datatable['案件类型'] = np.where(pd.isnull(Original_Data['s8']), '无', Original_Data['s8'])
    Datatable['审理阶段'] = np.where(pd.isnull(Original_Data['s9']), '无', Original_Data['s9'])


# 头部数据清洗
def Head_Data_Cleaning():
    # Set保存处理好的头部数据：审理法院、文书类型、检察机关
    global Datatable, Original_Data
    s22_data = Original_Data['s22'].replace(r'\s+', '').str.split('\n', expand=True)

    def clean_row(row):
        Set = ['其他', '其他', '无'] 

        if pd.isnull(row).all():
            return Set[0], Set[1], Set[2]
        for element in row:
            if not pd.notnull(element):
                break
            if len(element) > 20:
                element = element[-20:]
            if '检察' in str(element):
                Set[2] = element
            Is_Court = '法院' in str(element)
            matched_words = [word for word in word_bag['文书类型'] if word in str(element)]

            if Is_Court and not matched_words:
                Set[0] = element
            elif Is_Court and matched_words:
                court_index = str(element).index('法院')
                keyword_index = min(
                    str(element).index('书') if '书' in str(element) else len(str(element)),
                    str(element).index('令') if '令' in str(element) else len(str(element))
                )
                Set[0] = str(element)[:court_index]
                Set[1] = word_mapping.get(matched_words[-1], matched_words[-1])
            elif not Is_Court and matched_words:
                Set[1] = word_mapping.get(matched_words[-1], matched_words[-1])
        return Set[0], Set[1], Set[2]

    results = s22_data.apply(clean_row, axis=1)
    results_df = pd.DataFrame(results.tolist(), columns=['审理法院', '文书类型', '检察机关'])
    Datatable[['审理法院', '文书类型', '检察机关']] = results_df

# 主文本清洗
def Process_Text():
    
    global Datatable, Original_Data

    def replace_tag_1(match):
        tag = match.group(0)
        id_match = re.search(r"id", tag)
        if id_match:
            return f'({id_match.group(0)})'
        else:
            return ''

    def replace_tag_2(match):
        preceding_char = match.string[match.start()-1] if match.start() != 0 else ''
        if preceding_char == '。':
            return '(id)'
        else:
            return ''
    def full_to_half(match):
        char = match.group(0)
        return chr(ord(char) - 0xFEE0)
    
    def fullwidth_to_halfwidth(text):
    # 匹配所有全角字符
        return re.sub(r'[\uFF01-\uFF5E]', full_to_half, text).replace('\u3000', ' ').lower()
    
    def Is_Formulated(text):
        if text == "":
            return False
        lines = text.split('\n')
        for line in lines[:-1]:
            if not line.endswith('。'):
                return False
        return True

    def remove_unicode_control_characters(s):
        return s.replace('\u200F', '')

    Index = 0
    for row in Datatable.itertuples():
        
        if Datatable['是否公开'].iloc[Index] == '信息公开':
            Datatable['相关人员'].iloc[Index] = '不公开'
            Datatable['审理经过'].iloc[Index] = '不公开'
            Datatable['诉称与法院查明'].iloc[Index] = '不公开'
            Index += 1
            continue

        HTML = Original_Data['qwContent'].iloc[Index]
        HTML = remove_unicode_control_characters(HTML)
        if not '<p' in HTML:
            cleaned_html = re.sub(r'\s|&nbsp;|', '', HTML, flags=re.MULTILINE | re.DOTALL).lower()
            cleaned_html = re.sub(f'，，', '，', cleaned_html, flags=re.MULTILINE | re.DOTALL)
            match = re.search(pattern_3, cleaned_html)
            if match:
                cleaned_html = match.group(1)
            cleaned_html = re.sub(r'<.*?right>.*?<.*?>', '', cleaned_html)
            cleaned_html = re.sub(r'<.*?>', replace_tag_1, cleaned_html)
            cleaned_html = re.sub(r"\(id\)", replace_tag_2, cleaned_html)
            cleaned_html = re.sub(r'<.*', '', cleaned_html)
            Number = fullwidth_to_halfwidth(Datatable['案号'].iloc[Index])
        
            index = cleaned_html.find(Number)
            if index != -1:
                cleaned_html = cleaned_html[index + len(Number):]
            if Datatable['本院认为'].iloc[Index] == '':
                Datatable['本院认为'].iloc[Index] = '无'
            if Datatable['本院认为'].iloc[Index] != '无': 
                Sentence_to_Del = Datatable['本院认为'].iloc[Index].split('。')[0]
                index_to_del = cleaned_html.find(Sentence_to_Del)
                if index_to_del != -1:
                    cleaned_html = cleaned_html[:index_to_del]
            index = cleaned_html.find('审理终结。')
            if index != -1 and not cleaned_html[index:].startswith('审理终结。(id)'):
                cleaned_html = cleaned_html[:index] + '审理终结。(id)' + cleaned_html[index+len('审理终结。'):]
            paragraphs = re.split(r"\(id\)", cleaned_html)

            
            # 检查
            if not pd.isnull(Original_Data['s23'].iloc[Index]) and Is_Formulated(Original_Data['s23'].iloc[Index]):
                text = Original_Data['s23'].iloc[Index]
                sentences = text.split('。')
                first_sentence = sentences[0] if len(sentences) > 0 else ''
                last_sentence = sentences[-1] if len(sentences) > 1 else first_sentence

                if_Find = paragraphs[0].find(first_sentence)

                cleaned_html = re.sub(r"\(id\)", '', cleaned_html)
                first_index = cleaned_html.find(first_sentence + '。')
                last_index = cleaned_html.find(last_sentence + '。')

                if if_Find:
                    if first_index != -1:
                        paragraphs[0] = cleaned_html[:first_index]
                    if last_index != -1:
                        if len(paragraphs) >= 2:
                            paragraphs[1] = cleaned_html[first_index:last_index + len(last_sentence + '。')] 
                        if len(paragraphs) >= 3:
                            paragraphs[2] = cleaned_html[last_index + len(last_sentence + '。'):]

            
            for i,para in enumerate(paragraphs):
                if para:
                    if para[-1] == '，':
                        paragraphs[i] = para[:-1] + '。'

            List = ['相关人员','审理经过','诉称与法院查明']
            for i in range(3):
                if i >= len(paragraphs):
                    Datatable[List[i]].iloc[Index] = '无'
                else:
                    Datatable[List[i]].iloc[Index] = paragraphs[i]
            
        
        else:
            Datatable['审理经过'].iloc[Index] = re.sub(r'\s|&nbsp;| |\n|', '', Datatable['s23'].iloc[Index], flags=re.MULTILINE | re.DOTALL).lower()
            Datatable['审理经过'].iloc[Index] = re.sub(f'，，', '，', Datatable['审理经过'].iloc[Index], flags=re.MULTILINE | re.DOTALL)
            Datatable['审理经过'].iloc[Index] = fullwidth_to_halfwidth(Datatable['审理经过'].iloc[Index])
            cleaned_html = re.sub(r'\s|&nbsp;| |\n|', '', HTML, flags=re.MULTILINE | re.DOTALL).lower()
            cleaned_html = re.sub(f'，，', '，', cleaned_html, flags=re.MULTILINE | re.DOTALL)
            cleaned_html = fullwidth_to_halfwidth(cleaned_html)
            cleaned_html = re.sub(r'<.*?>', '', cleaned_html)
            escaped_text_to_remove = re.escape(Datatable['审理经过'].iloc[Index].replace('\\|\n|','').split(',')[0])
            
            Names = re.split(escaped_text_to_remove, cleaned_html)[0]
            Number = fullwidth_to_halfwidth(Datatable['案号'].iloc[Index])
            index = Names.find(Number)
            if index != -1:
                Names = Names[index + len(Number):]
            if not len(Names)==0:
                Datatable['相关人员'].iloc[Index] = Names
            else:
                Datatable['相关人员'].iloc[Index] = '无'
            


            Datatable['诉称与法院查明'].iloc[Index] = Original_Data['s25'].iloc[Index] if not pd.isnull(Original_Data['s25'].iloc[Index]) else '无'
            Datatable['诉称与法院查明'].iloc[Index] = re.sub(r'\s|&nbsp;| |\n|', '', Datatable['审理经过'].iloc[Index], flags=re.MULTILINE | re.DOTALL).lower()
            Datatable['诉称与法院查明'].iloc[Index] = re.sub(f'，，', '，', Datatable['审理经过'].iloc[Index], flags=re.MULTILINE | re.DOTALL)
            Datatable['诉称与法院查明'].iloc[Index] = fullwidth_to_halfwidth(Datatable['诉称与法院查明'].iloc[Index])
        
        try:
            if Datatable['相关人员'].iloc[Index][-1] == '：':
                Datatable['相关人员'].iloc[Index] = Datatable['相关人员'].iloc[Index].split('：')
                for i in Datatable['相关人员'].iloc[Index]:
                    i += '。'
            else:
                Datatable['相关人员'].iloc[Index] = Datatable['相关人员'].iloc[Index].split('。')
        except Exception as e:
            Datatable['相关人员'].iloc[Index] = ['无']
        
        Index += 1

    sc_list = []
    cm_list = []

    for i in Datatable['诉称与法院查明'].values:
        if i != '不公开':
            data = process_sc_cc(i)
            sc_list.append(data.iloc[0])
            cm_list.append(data.iloc[1])
        else:
            sc_list.append('不公开')
            cm_list.append('不公开')
    Datatable['诉称'] = sc_list
    Datatable['法院查明'] = cm_list
    Datatable = Datatable.drop('诉称与法院查明', axis=1)

def check_word_in_sen(sen, words, pure):
    index = None
    for word in words:
        if word in sen:
            index = pure.index(sen)
        else:
            continue
    return index

def process_sc_cc(text):
    if pd.isnull(text):
        Set = ['无', '无']
        return pd.Series({'诉称': Set[0], '法院查明': Set[1]})
    else:
        Set = ['', '']
        pure_data = text.replace('\n', '')
        s25_data = [i.replace('\n', '') for i in text.replace(r'\s+', '').split('。')]
        # 先找到法院的开始句子
        court_index = None
        futures = [executor.submit(check_word_in_sen, i, word_bag['法院'], s25_data) for i in s25_data]
        results = [i.result() for i in futures]
        for result in results:
            if result is not None:
                court_index = result
                break
        if court_index is not None:
            # 有的话直接分割，对应加入到诉称和法院里面
            tar_sen = s25_data[court_index]
            Set[0] = pure_data[:pure_data.find(tar_sen)]
            Set[1] = pure_data[pure_data.find(tar_sen):]
        else:
            # 没有的话先判断有没有原告，被告的词袋，没有的话就加入到法院，有的话加入到诉称
            talk_logo = 0
            futures = [executor.submit(check_word_in_sen, i, word_bag['诉称'], s25_data) for i in s25_data]
            results = [i.result() for i in futures]
            for i in results:
                if i is not None:
                    talk_logo = 1
                    break
            if talk_logo == 0:
                Set[0] = '无'
                Set[1] = pure_data
            else:
                Set[0] = pure_data
                Set[1] = '无'

        if Set[0] is None or Set[0] == '，' or Set[0] == '':
            Set[0] = '无'
        if Set[1] is None or Set[1] == '，' or Set[1] == '':
            Set[1] = '无'
        return pd.Series({'诉称': Set[0], '法院': Set[1]})


def Fuc_Name(Str):
    
    NameList = []
    global Judge_Tokenizer
    pattern = '|'.join(f"(?={re.escape(keyword)})" for keyword in Judge_Tokenizer)
    List = re.split(f"({pattern})", Str.replace('\n', ''))
    for Name in List:
        if '审判员' in Name:
            Name = Name.replace('审判员', '审判员：')
            NameList.append(Name)
        if '审判长' in Name:
            Name = Name.replace('审判长', '审判长：')
            NameList.append(Name)
    if len(NameList) == 0:
        NameList = ['无']
    return NameList

def Judge_Name():
    global Datatable
    Datatable['法官姓名'] = Datatable['落款'].apply(lambda x: Fuc_Name(x))


'''
下方增加逻辑
'''

# 供地区使用的apply方法
def Location_Process(row):
    if not row:
        return '无'
    keys = list(province_abbreviations.keys())
    for k in keys:
        if k in row:
            return province_abbreviations.get(k)

# 地区
def Location():
    global Datatable
    Datatable['地区'] = Datatable['审理法院'].apply(lambda x: Location_Process(x))
    Datatable['地区'] = Datatable['地区'].apply(lambda x: '北京' if not x else x)
    
# 供审理阶段使用的apply方法
def Process_Stage_Process(row):
    try:
        if not row:
            return '无'
        int(row)
        return '无'
    except Exception as e:
        return row

# 审理阶段
def Process_Stage():
    global Datatable
    Datatable['审理阶段'] = Datatable['审理阶段'].apply(lambda x: Process_Stage_Process(x))

def Plaintiff_and_Deffendant_Process(row):
    if not row:
        return ['无', '无']
    global Deffendant_Split
    Plaintiff = ''
    Deffendant = ''
    for keyword in Deffendant_Split:
        index = row.find(keyword)
        if index != -1:
            # 找到句号的位置
            period_index = row.rfind('。', 0, index+1)
            # 如果没有找到句号，就从行的开始处取文本
            if period_index == -1:
                period_index = 0
            else:
                # 如果找到了句号，从句号的下一个字符开始取文本
                period_index += 1
            Plaintiff = row[:period_index].strip()
            Deffendant = row[period_index:].strip()
            break
    if Deffendant:
        return [Plaintiff, Deffendant]
    else:
        return [row, '无']

def Plaintiff_and_Deffendant():
    global Datatable
    split_data = Datatable['诉称'].apply(Plaintiff_and_Deffendant_Process).tolist()
    
    # Create new DataFrame from the list of lists
    split_df = pd.DataFrame(split_data, columns=['原告诉称', '被告辩称'], index=Datatable.index)
    
    # Assign the new columns back to the original DataFrame
    Datatable['原告诉称'] = split_df['原告诉称']
    Datatable['被告辩称'] = split_df['被告辩称']
    

# 供涉案金额使用的apply方法
def Amount_Process(Text_to_process):
    Sum = 0
    amounts = []
    pattern = re.compile(r'（.*?）')
    Clean_Text = re.sub(pattern, '', Text_to_process)
    matches = re.findall(r"\d+(?:\.\d+)?万?(?:元|万元)", Clean_Text)
    if matches:
        for match in matches:
            amount = re.search(r"\d+(?:\.\d+)?", match).group()
            if "万元" in match:
                amount = float(amount) * 10000
            else:
                amount = float(amount)
            amounts.append(amount)
    else:
        amounts.append(0)

    Sum = Calculate_Amount(amounts)
    if Sum == 0:
        return '无'
    else:
        return str(Sum)

# 涉案金额
def Amount_Involved():
    global Datatable
    Apply = Datatable['原告诉称']
    Datatable['申请金额'] = Apply.apply(lambda x: Amount_Process(x))
    Apply = Datatable['判决结果']
    Datatable['实际判决金额'] = Apply.apply(lambda x: Amount_Process(x))


# def Split_Plaintiff_and_Deffendant_Process(row):
#     try:
#         sentences = row.split('。')
#     except Exception as e:
#         sentences = row
#     plaintiff = []
#     deffendant = []
#     for sentence in sentences:
#         index_p = 99999999999
#         index_d = 99999999999
#         for keyword in Plaintiff_Token:
#             Pos = sentence.find(keyword)
#             if Pos != -1:
#                 index_p = min(index_p, Pos)
        
#         for keyword in Deffendant_Token:
#             Pos = sentence.find(keyword)
#             if Pos != -1:
#                 index_d = min(index_d, Pos)
        
#         if index_p == 99999999999 and index_d == 99999999999:
#             pass
#         else:
#             if index_p < index_d:
#                 plaintiff.append(sentence)
#             else:
#                 deffendant.append(sentence)
#     if len(plaintiff) == 0:
#         plaintiff = ['无']
#     if len(deffendant) == 0:
#         deffendant = ['无']

#     return [plaintiff, deffendant] 


# def Split_Plaintiff_and_Deffendant():
#     global Datatable
#     split_data = Datatable['相关人员'].apply(Split_Plaintiff_and_Deffendant_Process).tolist()
    
#     # Create new DataFrame from the list of lists
#     split_df = pd.DataFrame(split_data, columns=['原告人', '被告人'], index=Datatable.index)
    
#     # Assign the new columns back to the original DataFrame
#     Datatable['原告人'] = split_df['原告人']
#     Datatable['被告人'] = split_df['被告人']





'案号', '法官名字', '案件名', '裁判机构','裁判地区','裁判日期','案由','案件类型','文书类型','法院判决主文',




'''
上方增加逻辑
'''

def Origin_Text(row):
    if pd.isnull(row):
        return None
    patterN = r'<.*?>'
    Txt = re.sub(patterN, '', row)
    return Txt


def save_into_DB(Year):
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
    from pymongo import MongoClient, InsertOne
    Target_Base_Name = 'Test'
    Target_collection = Target_DB[Target_Base_Name]


    insert_Col = ['案号', '法官名字', '案件名', '裁判机构', '裁判日期', '案件类型', '文书类型', '法院判决主文']
    
    # 创建一个新的 DataFrame，只包含 insert_Col 中指定的列
    Test_DF = Datatable[insert_Col].copy()
    Test_DF['法官名字'] = Datatable['法官姓名']
    Test_DF['案件名'] = Datatable['标题']
    Test_DF['裁判机构'] = Datatable['审理法院']
    Test_DF['裁判日期'] = Datatable['判决日期']
    Test_DF['案件类型'] = Datatable['案件类型']
    Test_DF['文书类型'] = Datatable['文书类型']
    Test_DF['法院判决主文'] = Datatable['判决结果']
    Test_DF['原文'] = Original_Data['qwContent'].apply(lambda x: Origin_Text(x))

    operations = [InsertOne(doc) for doc in Test_DF.to_dict('records')]
    Target_collection.bulk_write(operations)

# 入库
# def save_into_DB():
#     global Client, Datatable, Result_Col
#     global Datatable, Result_Col, last_processed_collection, Progress, last_processed_position
#     Datatable["_id"] = Datatable["_id"].apply(lambda x: ObjectId(x))
#     # print(Datatable)
#     # Datatable_1 包含案由中有'合同纠纷'或'不当得利纠纷'或'无因管理节分'的记录
#     conditions_1 = (Datatable['案由'].str.contains('合同纠纷') | 
#                     Datatable['案由'].str.contains('不当得利纠纷') | 
#                     Datatable['案由'].str.contains('无因管理节分'))
#     Datatable_1 = Datatable[conditions_1][Result_Col]


#     # Datatable_2 包含案由中有'与公司、证券、保险、票据等有关的民事纠纷'的记录
#     condition_2 = Datatable['案由'].str.contains('与公司、证券、保险、票据等有关的民事纠纷',na=False, regex=False)
#     Datatable_2 = Datatable[condition_2][Result_Col]
    
#     # Datatable_0 包含除了Datatable_1和Datatable_2之外的所有记录
#     Datatable_0 = Datatable[~(conditions_1 | condition_2)][Result_Col]
    
#     random_number = random.randint(1, 4)
#     db = Client['cpws']

#     Table_Name_1 = '合同、无因管理、不当得利纠纷_' + str(random_number)
#     Table_Name_2 = '与公司、证券、保险、票据等有关的民事纠纷_' + str(random_number)
#     Else_Name = 'Else_' + str(random_number)

#     if Table_Name_1 not in db.list_collection_names():
#         db.create_collection(Table_Name_1)
#     if Table_Name_2 not in db.list_collection_names():
#         db.create_collection(Table_Name_2)
#     if Else_Name not in db.list_collection_names():
#         db.create_collection(Else_Name)

#     Table_1 = db[Table_Name_1]
#     Table_2 = db[Table_Name_2]
#     ELSE = db[Else_Name]

#     Table_1_dict = Datatable_1.to_dict('records')
#     Table_2_dict = Datatable_2.to_dict('records')
#     ELSE_dict = Datatable_0.to_dict('records')

#     try:
#         Table_1.insert_many(Table_1_dict)
#     except Exception as e:
#         pass
#     try:
#         Table_2.insert_many(Table_2_dict)
#     except Exception as e:
#         pass
#     try:
#         ELSE.insert_many(ELSE_dict)
#     except Exception as e:
#         pass

#     year = last_processed_collection[-2:]
#     save_progress(last_processed_collection, last_processed_position, Progress, year)

# 进度保存
def save_progress( collection_name, current_position, current_progress, year):
    Pro_directory = "Progress"
    if not os.path.exists(Pro_directory):
        os.makedirs(Pro_directory)
    filename = os.path.join(Pro_directory, 'progress_' + year + '.txt')
    with open(filename, 'w') as file:
        file.write(f"{collection_name},{current_position},{current_progress}")

# 将Datatable保存为jsonl格式的文件
def save_to_jsonl():
    global Datatable, Result_Col, last_processed_collection, Progress, last_processed_position
    save_path = './Ws_Output/'
    filename = str(last_processed_collection) + str(int(Progress/(size*100))).zfill(5) + '.jsonl'
    save_path = save_path + str(last_processed_collection) + '/'
    Datatable = Datatable[Result_Col]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, filename), 'a', encoding='utf-8') as file:
        for _, row in Datatable.iterrows():
            json_data = row.to_dict()
            json.dump(json_data, file, ensure_ascii=False)
            file.write('\n')
    file.close()
    year = last_processed_collection[-2:]
    save_progress(last_processed_collection, last_processed_position, Progress, year)

def save_error_index():
    file_name = f'Error_{last_processed_collection}.txt'
    try:
        with open(file_name, 'a') as file:
            file.write(f"{last_processed_position}\n")  # 添加换行符，使每个错误索引占一行
    except Exception as e:
        print(f"Error while saving error index: {e}\n")
    finally:
        if file and not file.closed:
            file.close()

def ReLogin():
    Client = Login(sever_ip, user_name, password)
    Initialize_Cursor()
    Client.server_info()

class ReconnectionError(Exception):
    pass

def reconnect_with_retry(reconnect_function, max_retries, delay_seconds):

    for attempt in range(1, max_retries + 99999999999):
        try:
            reconnect_function()
            print(f"Reconnection successful on attempt {attempt}\n")
            logging.info(f"Reconnection successful on attempt {attempt}\n")
            return  # 如果成功则退出函数
        except Exception as e:
            print(f"Reconnection attempt {attempt} failed. Error: {e}\n")
            logging.info(f"Reconnection attempt {attempt} failed. Error: {e}\n")
            time.sleep(delay_seconds)

    # 如果达到最大重试次数仍然失败，则抛出自定义异常
    logging.info(f"Maximum retry attempts reached. Unable to reconnect\n")
    raise ReconnectionError("Maximum retry attempts reached. Unable to reconnect.")


# 时间转换
def format_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ReconnectionError(Exception):
    pass

def reconnect_with_retry(reconnect_function, max_retries, delay_seconds):
    
    for attempt in range(1, max_retries + 99999999999):
        try:
            reconnect_function()
            print(f"Reconnection successful on attempt {attempt}\n")
            logging.info(f"Reconnection successful on attempt {attempt}\n")
            return  # 如果成功则退出函数
        except Exception as e:
            print(f"Reconnection attempt {attempt} failed. Error: {e}\n")
            logging.info(f"Reconnection attempt {attempt} failed. Error: {e}\n")
            time.sleep(delay_seconds)
    
    # 如果达到最大重试次数仍然失败，则抛出自定义异常
    logging.info(f"Maximum retry attempts reached. Unable to reconnect\n")
    raise ReconnectionError("Maximum retry attempts reached. Unable to reconnect.")


