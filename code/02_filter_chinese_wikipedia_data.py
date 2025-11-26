import os
import jsonlines

data2_folder = 'data2'
data3_folder = 'data3'

if not os.path.exists(data3_folder):
    os.makedirs(data3_folder)

for filename in os.listdir(data2_folder):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(data2_folder, filename)
        
        with jsonlines.open(file_path) as reader:
            for data in reader:
                if data.get('wikipediaLink_zh'):
                    new_file_path = os.path.join(data3_folder, filename)
                    with jsonlines.open(new_file_path, mode='a') as writer:
                        writer.write(data)

print("数据提取完成，非空的 wikipediaLink_zh 数据已保存到 data3 文件夹。")