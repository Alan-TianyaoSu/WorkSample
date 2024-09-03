import sys
import os
import logging
import pymongo
from pymongo.errors import AutoReconnect, OperationFailure
from tqdm import tqdm
import argparse
from Clean_Functions import *

def Logger_Config(year):
    global logger

    log_directory = "Logger"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    Filename = os.path.join(log_directory, 'progress' + year[-2:] + '.log')
    logging.basicConfig(filename=Filename, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def main(Year, Size, Total_progress):
    # 设置
    global year, size, Progress, total_progress
    year = Year; size = Size; total_progress = Total_progress
    Progress = Initial(year, size)
    Logger_Config(year)
    elapsed_time_1 = 0
    elapsed_time_2 = 0
    LOAD = 0
    start_time_0 = time.time()
    logging.info(f"Cleaning start, Batch_id:{year}\n")
    with tqdm(total=total_progress, desc="Processing") as pbar:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
        while Progress < total_progress:
            try:
                elapsed_time_1,elapsed_time_2,Progress = Process(size,year)
            except (pymongo.errors.NetworkTimeout, AutoReconnect, OperationFailure) as e:
                logging.info(f"Network Error Occured: {e},\n")
                print(f"Caught a NetworkTimeout exception: {e}\n")
                # Writ_Data.save_error_index()
                try:
                    # 尝试10次重连，将这次的坏数据index保存
                    Progress = Initial(year, size)
                    elapsed_time_1,elapsed_time_2,Progress = Process(size,year)
                except ReconnectionError as e:
                    logging.info(f"Network Clashed: {e},\n")
                    print(f"Error: {e}")
                    break
            except Exception as e:
                # 捕获其他异常
                logging.info(f"Caught an unexpected exception: {e},\n")
                print(f"Caught an unexpected exception: {e}\n")
                raise(e)
                # Writ_Data.save_error_index()
                
            if IfEnd:
                print('清洗完成')
                break
            if LOAD == 0:
                
                pbar.update(Progress)
                pbar.update(size)
            else:
                pbar.update(size)
            LOAD += 1
            if LOAD % 50 == 0:

                Time_Step = round(time.time() - start_time_0)
                Time_Cur = format_seconds(Time_Step)
                Time_Tol = format_seconds(round((total_progress) / Progress * Time_Step))
                Progress_Cur = round(100 * Progress / total_progress)
                Speed = round(Progress / Time_Step,2)
                logging.info(f"50 Batch Completed. Processing:{Progress_Cur}%.[ {Time_Cur}/{Time_Tol}, {Speed}it/s ]\n")
            pbar.set_postfix({"Data retrieval time": elapsed_time_1, "Data process time": elapsed_time_2})
        logger.removeHandler(console_handler)
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time_0
    logging.info(f"WS_{year} CLeaning Completed. Time-Consuming: {format_seconds(round(elapsed_time))}.\n")
    print(f"WS_{year} CLeaning Completed. Time-Consuming: {format_seconds(round(elapsed_time))}.\n")
    print(f"运行时间：{format_seconds(round(elapsed_time))}")

def Process(size, year):
    start_time = time.time()  
    Progress = Get_Data(size)
    end_time_1 = time.time()  
    elapsed_time_1 = end_time_1 - start_time  
    start_time = time.time() 
    Clone_table()
    Head_Data_Cleaning()
    Process_Text()

    '''
    下方增加逻辑
    '''
    Judge_Name()
    Location()
    Process_Stage()
    Plaintiff_and_Deffendant()
    Amount_Involved()
    # Split_Plaintiff_and_Deffendant()
    '''
    上方增加逻辑
    '''
    save_into_DB(year)
    # save_into_DB()
    # save_to_jsonl()
    end_time_2 = time.time()
    elapsed_time_2 = end_time_2 - start_time
    return elapsed_time_1,elapsed_time_2,Progress


if __name__ == "__main__":
    year = sys.argv[1]
    data_as_json = sys.argv[2]
    data = json.loads(data_as_json)
    main(year, data['Size'], data['total_progress'])