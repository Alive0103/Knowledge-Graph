# import wikidata_query as wq
# import json
# import os
# import glob
# import threading
# import sys


# import time
# import requests

# def safe_request(url):
#     while True:
#         response = requests.get(url)
#         if response.status_code == 429:  # 如果收到429错误
#             time.sleep(5)  # 等待5秒后重试
#             continue
#         return response


# sys.setrecursionlimit(3000)  # 设置递归调用的最大深度
# num_threads = 64  # 定义同时运行的线程数

# file_path = "military"  # 定义存放输入文件的目录
# if not os.path.exists(file_path):  # 如果目录不存在，则创建
#     os.mkdir(file_path)

# prefix = "new"  # 定义输出文件的前缀
# if not os.path.exists(prefix + "_" + file_path):  # 检查输出目录是否存在，如果不存在则创建
#     os.mkdir(prefix + "_" + file_path)

# # 查找指定目录下的JSONL文件
# files = glob.glob(file_path + "/*.jsonl")
# langs = ["zh", "en"]  # 定义要查询的语言

# # 遍历找到的每个文件
# for file in files:
#     if file[0] != ".":  # 忽略以"."开头的文件
#         open(prefix + "_" + file, 'w', encoding="utf-8")  # 创建输出文件
#         results = []  # 初始化结果列表

#         # 读取输入文件并解析每一行
#         with open(file, "r", encoding="utf-8") as f:
#             for line in f:
#                 results.append(json.loads(line))  # 将每一行的JSON数据加载到结果列表中


#             # 定义递归查询函数
#             def getQuery(index):
#                 global results  # 使用全局结果列表

#                 if len(results) > 0:  # 如果结果列表非空
#                     result = results[-1]  # 获取列表最后一个结果
#                     results.pop()  # 从列表中移除该结果

#                     # 如果结果中没有别名
#                     if "aliases" not in result:
#                         entity = "wd:" + result["entity"]["value"].split("/")[-1]  # 获取实体ID
#                         result["aliases"] = {}  # 初始化别名字典
#                         for lang in langs:  # 遍历语言
#                             aliases = wq.query_aliases(entity, lang)  # 查询别名
#                             result["aliases"][lang] = aliases  # 将别名存入结果中

#                     # 将结果写入输出文件
#                     with open(prefix + "_" + file, 'a', encoding='utf-8') as g:
#                         g.write(json.dumps(result, ensure_ascii=False) + "\n")  # 写入JSON格式结果

#                 # 递归调用，控制递归深度
#                 if index > 0:
#                     getQuery(index)  # 递归调用
#                 else:
#                     # 如果活动线程数少于最大线程数，则启动新线程
#                     for i in range(num_threads):  # - threading.active_count()):
#                         t = threading.Thread(target=getQuery, args=(i + 1,))  # 创建新线程
#                         t.start()  # 启动线程


#             getQuery(0)  # 初始化递归查询

import wikidata_query as wq
import json
import os
import glob
import threading
import sys
import time
import requests

def safe_request(url):
    while True:
        response = requests.get(url)
        if response.status_code == 429:  
            time.sleep(5)  
            continue
        return response

sys.setrecursionlimit(3000) 
num_threads = 64  

file_path = "military"  
if not os.path.exists(file_path):  
    os.mkdir(file_path)

prefix = "new"  
if not os.path.exists(prefix + "_" + file_path):  
    os.mkdir(prefix + "_" + file_path)

files = glob.glob(file_path + "/*.jsonl")
langs = ["zh", "en"]  

for file in files:
    if file[0] != ".":  
        open(prefix + "_" + file, 'w', encoding="utf-8")  
        results = []  

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line)) 

        # 定义查询函数
        def getQuery(index):
            global results  

            while len(results) > 0: 
                result = results[-1] 
                results.pop()  

                entity = "wd:" + result["entity"]["value"].split("/")[-1] 
                result["aliases"] = {}  
                aliases_found = False  

                for lang in langs: 
                    aliases = wq.query_aliases(entity, lang)  
                    if aliases:  
                        result["aliases"][lang] = aliases 
                        aliases_found = True


                if not aliases_found:
                    result["aliases"] = None

                with open(prefix + "_" + file, 'a', encoding='utf-8') as g:
                    g.write(json.dumps(result, ensure_ascii=False) + "\n")  

        threads = []  

        for i in range(num_threads):
            t = threading.Thread(target=getQuery, args=(i + 1,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
