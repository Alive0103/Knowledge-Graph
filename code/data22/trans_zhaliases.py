import json
import os
from transformers import MarianMTModel, AutoTokenizer
from langid import classify

# 初始化翻译模型
def initialize_translator(path, source, target):
    model_name = os.path.join(path, f'Helsinki-NLP-opus-mt-{source}-{target}')
    model = MarianMTModel.from_pretrained(model_name).to('cpu')  # 使用 CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# 翻译函数
def translate(texts, model, tokenizer):
    texts = list(set(texts))  # 去重
    translated_texts = []
    
    for text in texts:
        tokenized_text = tokenizer.encode(text, return_tensors="pt").to('cpu')  # 使用 CPU
        translated_tokens = model.generate(tokenized_text, no_repeat_ngram_size=1, repetition_penalty=1.2)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        if "I don't think so" in translated_text:
            print(f"翻译错误: {translated_text}")
        else:
            translated_texts.append(translated_text)
    
    return translated_texts

# 检查文本是否是繁体
def is_traditional(text):
    lang, _ = classify(text)
    return lang == 'zh' and any(ord(char) > 127 for char in text)

def complete(data_folder, model_path):
    zh_en_model, zh_en_tokenizer = initialize_translator(model_path, "zh", "en")
    en_zh_model, en_zh_tokenizer = initialize_translator(model_path, "en", "zh")

    for file_name in os.listdir(data_folder):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(data_folder, file_name)
            updated_data = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    task = json.loads(line.strip())  # 解析每一行JSON数据

                    zh_aliases = task.get("zh_aliases", [])
                    en_aliases = task.get("en_aliases", [])
                    label = task["label"]

                    # 如果 zh_aliases 为空，则翻译 en_aliases 并去重加入到 zh_aliases
                    if not task["zh_aliases"]:
                        translated_en_aliases = translate(en_aliases, en_zh_model, en_zh_tokenizer)
                        task["zh_aliases"] = list(set(translated_en_aliases))

                    updated_data.append(task)

            # 更新文件
            with open(file_path, 'w', encoding='utf-8') as f:
                for task in updated_data:
                    f.write(json.dumps(task, ensure_ascii=False) + "\n")

            print(f"文件 {file_name} 处理完成并已更新。")

if __name__ == "__main__":
    # 假设代码和文件在同一目录下
    current_dir = os.getcwd()  # 获取当前工作目录
    data_folder = current_dir  # 数据文件夹路径
    model_path = os.path.join(current_dir, "model")  # 翻译模型路径

    complete(data_folder, model_path)

    print("所有文件已更新完成。")