import json
import os
import tempfile
import shutil
from transformers import MarianMTModel, AutoTokenizer
from langid import classify

def initialize_translator(path, source, target):
    # 修改模型路径构建方式，以匹配实际下载的目录名
    model_name = os.path.join(path, f'opus-mt-{source}-{target}')
    print(f"尝试加载模型: {model_name}")
    
    # 检查模型目录是否存在
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"模型目录不存在: {model_name}")
    
    print(f"模型目录内容: {os.listdir(model_name)}")
    
    try:
        print(f"正在加载模型...")
        model = MarianMTModel.from_pretrained(model_name).to('cpu')  
        print(f"正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"模型 {model_name} 加载成功")
    except Exception as e:
        print(f"直接加载模型 {model_name} 失败: {e}")
        print("尝试复制到临时目录再加载...")
        # 尝试将模型复制到临时目录再加载
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_model_dir = os.path.join(temp_dir, f'opus-mt-{source}-{target}')
                print(f"复制模型到临时目录: {temp_model_dir}")
                shutil.copytree(model_name, temp_model_dir)
                print(f"临时目录内容: {os.listdir(temp_model_dir)}")
                print("从临时目录加载模型...")
                model = MarianMTModel.from_pretrained(temp_model_dir).to('cpu')
                tokenizer = AutoTokenizer.from_pretrained(temp_model_dir)
                print(f"从临时目录加载模型成功")
        except Exception as e2:
            print(f"从临时目录加载也失败: {e2}")
            raise e
    return model, tokenizer

def translate(texts, model, tokenizer):
    texts = list(set(texts))  
    translated_texts = []
    
    for text in texts:
        tokenized_text = tokenizer.encode(text, return_tensors="pt").to('cpu')  
        translated_tokens = model.generate(tokenized_text, no_repeat_ngram_size=1, repetition_penalty=1.2)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        if "I don't think so" in translated_text:
            print(f"翻译错误: {translated_text}")
        else:
            translated_texts.append(translated_text)
    
    return translated_texts

def is_traditional(text):
    lang, _ = classify(text)
    return lang == 'zh' and any(ord(char) > 127 for char in text)

def complete(data_folder, model_path):
    # 确保模型目录存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    
    print(f"模型根目录内容: {os.listdir(model_path)}")
    
    # 检查模型文件是否存在
    zh_en_model_path = os.path.join(model_path, 'opus-mt-zh-en')
    en_zh_model_path = os.path.join(model_path, 'opus-mt-en-zh')
    
    if not os.path.exists(zh_en_model_path):
        raise FileNotFoundError(f"中文到英文模型不存在: {zh_en_model_path}")
        
    if not os.path.exists(en_zh_model_path):
        raise FileNotFoundError(f"英文到中文模型不存在: {en_zh_model_path}")
    
    print(f"中文到英文模型目录内容: {os.listdir(zh_en_model_path)}")
    print(f"英文到中文模型目录内容: {os.listdir(en_zh_model_path)}")
    
    # 正确加载模型
    # en_zh_model: 英文到中文的翻译模型
    # zh_en_model: 中文到英文的翻译模型
    print("开始加载英文到中文模型...")
    en_zh_model, en_zh_tokenizer = initialize_translator(model_path, "en", "zh")
    print("开始加载中文到英文模型...")
    zh_en_model, zh_en_tokenizer = initialize_translator(model_path, "zh", "en")

    for file_name in os.listdir(data_folder):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(data_folder, file_name)
            updated_data = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    task = json.loads(line.strip())  

                    zh_aliases = task.get("zh_aliases", [])
                    en_aliases = task.get("en_aliases", [])
                    label = task["label"]

                    # 如果没有中文别名，则使用英文到中文的模型进行翻译
                    if not task["zh_aliases"]:
                        translated_en_aliases = translate(en_aliases, en_zh_model, en_zh_tokenizer)
                        task["zh_aliases"] = list(set(translated_en_aliases))

                    updated_data.append(task)

            with open(file_path, 'w', encoding='utf-8') as f:
                for task in updated_data:
                    f.write(json.dumps(task, ensure_ascii=False) + "\n")

            print(f"文件 {file_name} 处理完成并已更新。")

if __name__ == "__main__":
    current_dir = os.getcwd()  
    data_folder = current_dir  
    model_path = os.path.join(current_dir, "model")  

    complete(data_folder, model_path)

    print("所有文件已更新完成。")