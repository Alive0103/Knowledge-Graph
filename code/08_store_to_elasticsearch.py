import os, re, json, requests, argparse
from SPARQLWrapper import SPARQLWrapper, JSON
from opencc import OpenCC
cc = OpenCC('t2s')

import urllib3
from urllib.parse import unquote, urlencode
from requests.adapters import HTTPAdapter, Retry
from transformers import MarianMTModel, AutoTokenizer
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from bs4 import BeautifulSoup
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="8c394eebd7e951bd95c9b27d92fbf579.xbTTVYgIAaKv220w")  # 请填写您自己的APIKey

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

ZH_WIKI_PREFIX = "https://zh.wikipedia.org/wiki/"
EN_WIKI_PREFIX = "https://en.wikipedia.org/wiki/"

def parse_query(query):
    sparql.setQuery(query)
    results = sparql.query().convert()
    results = results['results']['bindings']
    return results

def parse_returned_aliases(results, lang):
    aliases = []
    for result in results:
        aliases.append(result["label"]["value"])
        if "aliases" in result:
            aliases.append(result["aliases"]["value"])

    if lang == "zh":    
        string = "$$".join(aliases)
        string = cc.convert(string)
        aliases = string.split("$$")

    aliases = list(set(aliases))
    return aliases

def get_aliases(entity, lang):
    query ="""
    SELECT ?label ?aliases 
    WHERE {
        VALUES ?entity { ?entity_id }
        ?entity rdfs:label ?label.
        FILTER(LANG(?label) = "lang")

        OPTIONAL {
            ?entity skos:altLabel ?aliases.
            FILTER(LANG(?aliases) = "lang")
        }
    }
    """
    if "wd:" not in entity:
        entity = "wd:" + entity
    query = query.replace('?entity_id', entity).replace('"lang"', '"' + lang + '"')
    results = parse_query(query)
    aliases = parse_returned_aliases(results, lang)
    return aliases

def retrieve_wikidata_from_wikipedia(html=None, links=None, retries=3):
    if html:
        soup = BeautifulSoup(html, "html.parser")
        link = soup.select_one('#t-wikibase > a')
        if link:
            href = link.get('href')
            match = re.search(r'Q\d+', href)
            if match:
                qid = match.group(0)
                print("Extracted entity ID:", qid)
            else:
                print("No entity ID found.")
        else:
            print("No link found.")
    elif links:
        titles = {"zh": [], "en": []}
        for link in links:
            if link.startswith(ZH_WIKI_PREFIX):
                titles["zh"].append(unquote(link[len(ZH_WIKI_PREFIX):]))
            elif link.startswith(EN_WIKI_PREFIX):
                titles["en"].append(unquote(link[len(EN_WIKI_PREFIX):]))
            else:
                print("不支持的维基百科链接格式")
                return None, None

        results = {}
        for lang in titles.keys():
            api_url = "https://zh.wikipedia.org/w/api.php" if "zh" in lang else "https://en.wikipedia.org/w/api.php"
            
            params = {
                'action': 'query',
                'titles': "|".join(titles[lang]),
                'format': 'json',
                'prop': 'pageprops',
                'redirects': 1,  # 自动处理重定向
                'utf8': 1
            }
            
            params = urlencode(params)
            print(params)

            session = requests.Session()
            retries = Retry(total=retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retries)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            # try:
            headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0'}
            response = session.get(api_url, headers=headers, params=params, 
                                proxies=dict(http='socks5h://localhost:1080',
                                        https='socks5h://localhost:1080'), 
                                verify=False, timeout=5)
            response.raise_for_status()
            data = response.json()

            if 'query' in data:
                for key in data["query"]["pages"].keys():
                    page = data["query"]["pages"][key]
                    title = page["title"]
                    qid = page["pageprops"]["wikibase_item"]
                    if lang not in results:
                        results[lang] = {}
                    results[lang][title] = qid
        return results
    else:
        return None, None
    # except requests.RequestException as e:
    #     print(f"请求出错: {e}")    
    
def initialize_translator(path, source, target):
    model_name = path + '/Helsinki-NLP-opus-mt-' + source + '-' + target
    model = MarianMTModel.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
        
def read_tasks(file):
    tasks = []
    with open(file, "r", encoding="utf-8") as f:
        try: 
            tasks = json.loads(f.read())
        except:
            f.seek(0)
            for line in f:
                try:
                    tasks.append(json.loads(line))
                except:
                    print(line)
    return tasks

def is_traditional(text):
    """
    判断文本是否是简体中文
    """
    cc = OpenCC('s2t')  # 简体转繁体
    converted = cc.convert(text)
    return text == converted

def translate(texts, model, tokenizer):
    texts = list(set(texts))
    
    translated_texts = []
    for text in texts:
        # Tokenize the text
        tokenized_text = tokenizer.encode(text, return_tensors="pt").to('cuda')

        # Translate the tokenized text
        translated_tokens = model.generate(tokenized_text, no_repeat_ngram_size=1, repetition_penalty=1.2)  # Penalize repetition

        # Decode the translated tokens to a string
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        if "I don't think so" in translated_text:
            print(translated_text)
        else:
            translated_texts.append(translated_text)
    return translated_texts

def complete(): 
    zh_en_model, zh_en_tokenizer =  initialize_translator(args.translator_path, "zh", "en")
    en_zh_model, en_zh_tokenizer =  initialize_translator(args.translator_path, "en", "zh")
    
    def get_content(url):
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0'}
        response = requests.get(url, headers=headers, 
                            proxies=dict(http='socks5h://localhost:1080',
                                    https='socks5h://localhost:1080'), 
                            verify=False, timeout=5)
        response.raise_for_status()
        return response.text
    
    open(args.file + "-converted", "w")
    for idx, task in enumerate(read_tasks(args.file)):
        try:
            zh_aliases = task["zh_aliases"] if "zh_aliases" in task else []
            en_aliases = task["en_aliases"] if "en_aliases" in task else []

            label = task["itemLabel"]["value"]
            added_zh_aliases = [cc.convert(label)] if is_traditional(label) else [label]
            added_zh_aliases += translate(en_aliases, en_zh_model, en_zh_tokenizer)
            task["zh_aliases"] = list(set(zh_aliases + added_zh_aliases))

            added_en_aliases = translate(zh_aliases, zh_en_model, zh_en_tokenizer)
            task["en_aliases"] = list(set(en_aliases+ added_en_aliases))
            
            url = task["wikipediaLink"]
            if "content" in task:
                html = task["content"]
            else:
                task["content"] = get_content(url)
                
            description = get_description({"zh": task["zh_aliases"][0], "en": task["en_aliases"][0]}, html)
            
            if url.startswith(EN_WIKI_PREFIX):
                description = trans_by_api(description)
            else: 
                description = cc.convert(description)
                
            task["description_zh"] = description
        except:
            print(task)

        with open(args.file + "-converted", "a", encoding="utf-8") as f:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

def trans_by_api(text, target=None):

        # 创建一个包含实体类型判断和预测的请求
        response = client.chat.completions.create(
            model="glm-4-airx",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"你是一位专业的双语翻译员，擅长中英文互译。"
                        f"请将以下文本翻译为" + ("英文" if target == "en" else "中文") + "，确保准确无误且简洁：\n\n"
                        f"{text}\n\n"
                        f"请直接返回翻译后的文本，不需要解释或附加内容。"
                    )
                }
            ],
        )
        
        # 获取并返回模型的响应
        return "".join(response.choices[0].message.content.split('\n'))

def get_wikipedia_link(wikidata_id):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={wikidata_id}&format=json"

    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0'}
    response = requests.get(url, headers=headers, 
                        proxies=dict(http='socks5h://localhost:1080',
                                https='socks5h://localhost:1080'), 
                        verify=False, timeout=5)
    response.raise_for_status()
    data = response.json()    
    print(data)
    
    # Extracting the Wikipedia link
    if 'entities' in data and wikidata_id in data['entities']:
        sitelinks = data['entities'][wikidata_id].get('sitelinks', {})
        if 'enwiki' in sitelinks:
            return sitelinks['enwiki']['url'] if 'url' in sitelinks['enwiki'] else None
    return None

def get_description(labels, html, lang="zh"):
    soup = BeautifulSoup(html, "html.parser")
    
    def generate_desp(entity):
        if "别名" in entity:
            content = (f"什么是 '{entity}'？\n"
                       f"用中文回答关于该名词的详细解释，不用返回特点说明。\n")  
        else:
            content = (f"What is '{entity}'?\n"
                       f"Returns its detailed definition in English, with no additional answers explaining its functions or features.\n") 
        
        # 创建一个包含实体类型判断和预测的请求
        response = client.chat.completions.create(
            model="glm-4-airx",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
        )
        
        # 获取并返回模型的响应
        return "".join(response.choices[0].message.content.split('\n'))
        
    for table in soup.find_all('table'):
        table.decompose()

    try:
        description = "".join(soup.find('p').text.strip().split("\n"))
        
        if description:
            return description
        else:
            cmd = labels["zh"] + '（别名：' + labels["en"] + '）' if "zh" in lang \
                    else labels["en"] + ' (also known as "' + labels["zh"] + '" in Chinese)' 
            return generate_desp(cmd)
    except:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--file", default="wiki_complete.jsonl", type=str)
    parser.add_argument("-p", "--translator_path", default="../translate", type=str)
    args = parser.parse_args()
    
    # complete()
    get_wikipedia_link("Q48731975")