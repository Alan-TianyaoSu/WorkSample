# 多进程执行
import subprocess
from multiprocessing import Process
import json
import argparse
import subprocess
from Years_Table import *

def run_script(script_path, year, data):
    data_as_json = json.dumps(data)
    subprocess.run(["python", script_path, year, data_as_json])

def main():
    global years
    # 创建一个进程列表，为每个年份运行Main.py脚本
    processes = [
        Process(target=run_script, args=("Main.py", year, data))
        for year, data in years.items()
    ]

    # 启动所有进程
    for process in processes:
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()