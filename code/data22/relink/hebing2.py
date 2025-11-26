import os

def merge_jsonl_files(output_file):
    # 获取当前目录下所有.jsonl文件
    jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]

    # 如果没有找到.jsonl文件，提示用户
    if not jsonl_files:
        print("当前目录下没有找到任何.jsonl文件。")
        return

    # 打开目标文件用于写入
    with open(output_file, 'w', encoding='utf-8') as output:
        # 遍历所有.jsonl文件
        for file in jsonl_files:
            print(f"正在处理文件: {file}")
            with open(file, 'r', encoding='utf-8') as f:
                # 逐行读取并写入目标文件
                for line in f:
                    # 确保每一行后面都有换行符
                    output.write(line.strip() + "\n")

    print(f"所有.jsonl文件已合并到 {output_file}")

if __name__ == "__main__":
    output_file = "data2.jsonl"
    merge_jsonl_files(output_file)