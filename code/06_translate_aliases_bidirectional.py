import json
import os
from transformers import MarianMTModel, AutoTokenizer

# 初始化翻译模型
def initialize_translator(path, source, target):
    model_name = path + '/Helsinki-NLP-opus-mt-' + source + '-' + target
    model = MarianMTModel.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# 翻译函数
def translate(texts, model, tokenizer):
    texts = list(set(texts))  
    translated_texts = []
    
    for text in texts:
        tokenized_text = tokenizer.encode(text, return_tensors="pt").to('cuda')
        translated_tokens = model.generate(tokenized_text, no_repeat_ngram_size=1, repetition_penalty=1.2)  

        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        if "I don't think so" in translated_text:
            print(translated_text)  
        else:
            translated_texts.append(translated_text) 
    
    return translated_texts

def complete(data33_folder, model_path):
    zh_en_model, zh_en_tokenizer = initialize_translator(model_path, "zh", "en")
    en_zh_model, en_zh_tokenizer = initialize_translator(model_path, "en", "zh")

    for file_name in os.listdir(data33_folder):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(data33_folder, file_name)
            updated_data = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    task = json.loads(line)

                    zh_aliases = task.get("zh_aliases", [])
                    en_aliases = task.get("en_aliases", [])

                    en_aliases.append(task["itemLabel"]["value"])
                    added_en_aliases = translate(zh_aliases, zh_en_model, zh_en_tokenizer)
                    en_aliases = list(set(en_aliases + added_en_aliases))  

                    added_zh_aliases = translate(en_aliases, en_zh_model, en_zh_tokenizer)
                    zh_aliases = list(set(zh_aliases + added_zh_aliases))  

                    task["zh_aliases"] = zh_aliases
                    task["en_aliases"] = en_aliases

                    updated_data.append(task)

            with open(file_path, 'w', encoding='utf-8') as f:
                for task in updated_data:
                    f.write(json.dumps(task, ensure_ascii=False) + "\n")

            print(f"文件 {file_name} 处理完成并已更新。")

if __name__ == "__main__":
    data33_folder = "data33"  
    model_path = "model"  

    complete(data33_folder, model_path)

    print("所有文件已更新完成。")
