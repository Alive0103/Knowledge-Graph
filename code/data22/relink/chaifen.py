import json

# 打开原始文件
input_file = 'deduplicated_links.jsonl'
output_base_name = 'deduplicated_links'

# 用于存储当前批次的数据
batch = []
batch_size = 10000  # 修改为 100 条
batch_count = 1

with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        # 将每行解析为 JSON 对象
        data = json.loads(line.strip())
        batch.append(data)
        
        # 如果当前批次达到 100 条，写入文件
        if len(batch) == batch_size:
            output_file = f'{output_base_name}{batch_count}.jsonl'
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for item in batch:
                    json.dump(item, outfile, ensure_ascii=False)
                    outfile.write('\n')
            print(f'Wrote {batch_size} lines to {output_file}')
            batch = []
            batch_count += 1

# 如果还有剩余的数据，写入最后一个文件
if batch:
    output_file = f'{output_base_name}{batch_count}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in batch:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')
    print(f'Wrote {len(batch)} lines to {output_file}')