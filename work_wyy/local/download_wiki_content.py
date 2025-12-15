#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从find.xlsx中的wiki链接下载HTML内容并提取文本

功能：
1. 读取find.xlsx文件，获取第二列（wiki链接）
2. 访问每个链接，下载HTML内容
3. 提取HTML中的纯文本
4. 保存到本地文件夹，每个链接保存成一个txt文件
"""

import os
import sys
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse
from tqdm import tqdm
from datetime import datetime
import time

# 导入配置
try:
    from config import WORK_DIR
except ImportError:
    WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_find_excel(file_path):
    """读取find.xlsx文件，返回查询词和链接列表"""
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    links = df[1].tolist()
    
    # 清理数据，移除NaN
    queries = [str(q).strip() if pd.notna(q) else "" for q in queries]
    links = [str(link).strip() if pd.notna(link) else "" for link in links]
    
    # 过滤空值
    valid_pairs = [(q, link) for q, link in zip(queries, links) if q and link]
    
    print(f"从 {file_path} 读取了 {len(valid_pairs)} 个有效查询-链接对")
    return valid_pairs


def sanitize_filename(filename):
    """清理文件名，移除非法字符"""
    # 移除或替换非法字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除前后空格和点
    filename = filename.strip('. ')
    # 限制长度
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def get_filename_from_link(link, query=""):
    """从链接生成文件名"""
    if not link:
        return "unknown.txt"
    
    # 尝试从URL提取标题
    if "wikipedia.org/wiki/" in link:
        try:
            parts = link.split("/wiki/", 1)
            if len(parts) == 2:
                title = parts[1]
                # 移除URL参数
                if '#' in title:
                    title = title.split('#')[0]
                if '?' in title:
                    title = title.split('?')[0]
                try:
                    decoded_title = unquote(title)
                    filename = sanitize_filename(decoded_title) + ".txt"
                    return filename
                except:
                    filename = sanitize_filename(title) + ".txt"
                    return filename
        except:
            pass
    
    # 如果无法从URL提取，使用查询词
    if query:
        filename = sanitize_filename(query) + ".txt"
        return filename
    
    # 最后使用URL的一部分
    parsed = urlparse(link)
    path = parsed.path.strip('/')
    if path:
        filename = sanitize_filename(path.split('/')[-1]) + ".txt"
        return filename
    
    return "unknown.txt"


def download_wikipedia_page(url, retries=3, timeout=30):
    """下载维基百科页面"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                print(f"  下载失败: {e}")
                return None
    
    return None


def extract_text_from_html(html_content):
    """从HTML中提取纯文本"""
    if not html_content:
        return ""
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 移除script和style标签
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # 移除注释
        from bs4 import Comment
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()
        
        # 获取主要内容区域（维基百科通常使用id="content"或class="mw-body-content"）
        content = None
        
        # 尝试找到主要内容区域
        content = soup.find('div', {'id': 'content'}) or \
                  soup.find('div', {'id': 'bodyContent'}) or \
                  soup.find('div', {'class': 'mw-body-content'}) or \
                  soup.find('div', {'id': 'mw-content-text'})
        
        if content:
            text = content.get_text(separator='\n', strip=True)
        else:
            # 如果没有找到特定区域，获取body的所有文本
            body = soup.find('body')
            if body:
                text = body.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
        
        # 清理文本：移除多余的空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    except Exception as e:
        print(f"  提取文本失败: {e}")
        return ""


def download_and_save_content(find_pairs, output_dir):
    """下载并保存所有链接的内容"""
    print(f"\n开始下载并保存内容到: {output_dir}")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_links = []
    
    for query, link in tqdm(find_pairs, desc="下载内容"):
        try:
            # 生成文件名
            filename = get_filename_from_link(link, query)
            filepath = os.path.join(output_dir, filename)
            
            # 如果文件已存在，跳过
            if os.path.exists(filepath):
                skipped_count += 1
                continue
            
            # 下载HTML
            html_content = download_wikipedia_page(link)
            
            if not html_content:
                failed_count += 1
                failed_links.append((query, link))
                continue
            
            # 提取文本
            text_content = extract_text_from_html(html_content)
            
            if not text_content or len(text_content.strip()) < 50:
                print(f"  警告: {filename} 文本内容过短或为空")
                failed_count += 1
                failed_links.append((query, link))
                continue
            
            # 保存文件
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入元数据
                f.write(f"# 查询词: {query}\n")
                f.write(f"# 链接: {link}\n")
                f.write(f"# 下载时间: {datetime.now().isoformat()}\n")
                f.write("# " + "=" * 70 + "\n\n")
                # 写入正文
                f.write(text_content)
            
            success_count += 1
            
            # 添加延迟，避免请求过快
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  处理失败 ({query}): {e}")
            failed_count += 1
            failed_links.append((query, link))
            continue
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("下载完成统计")
    print("=" * 70)
    print(f"总链接数: {len(find_pairs)}")
    print(f"成功下载: {success_count} 个")
    print(f"跳过（已存在）: {skipped_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"成功率: {success_count/(len(find_pairs)-skipped_count)*100:.1f}%" if (len(find_pairs)-skipped_count) > 0 else "N/A")
    
    # 保存失败链接列表
    if failed_links:
        failed_file = os.path.join(output_dir, 'failed_links.txt')
        with open(failed_file, 'w', encoding='utf-8') as f:
            f.write("失败的链接列表\n")
            f.write("=" * 70 + "\n\n")
            for query, link in failed_links:
                f.write(f"查询: {query}\n")
                f.write(f"链接: {link}\n\n")
        print(f"\n失败链接列表已保存到: {failed_file}")
    
    return success_count, failed_count, skipped_count


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从find.xlsx中的wiki链接下载HTML内容并提取文本')
    parser.add_argument('--find-file', type=str, 
                       default=None,
                       help='find.xlsx文件路径（默认: work_wyy/data/find.xlsx）')
    parser.add_argument('--output-dir', type=str,
                       default=None,
                       help='输出目录（默认: work_wyy/data/find_wiki_content）')
    parser.add_argument('--retries', type=int, default=3,
                       help='下载失败重试次数（默认: 3）')
    parser.add_argument('--timeout', type=int, default=30,
                       help='请求超时时间（秒，默认: 30）')
    
    args = parser.parse_args()
    
    # 获取work_wyy目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_wyy_dir = os.path.dirname(script_dir)
    
    # 确定find文件路径
    if args.find_file:
        find_file = args.find_file
        if not os.path.isabs(find_file):
            find_file = os.path.abspath(find_file)
    else:
        find_file = os.path.join(work_wyy_dir, 'data', 'find.xlsx')
    
    if not os.path.exists(find_file):
        print(f"错误: 找不到find.xlsx文件: {find_file}")
        sys.exit(1)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir)
    else:
        output_dir = os.path.join(work_wyy_dir, 'data', 'find_wiki_content')
    
    print("=" * 70)
    print("从find.xlsx中的wiki链接下载HTML内容并提取文本")
    print("=" * 70)
    print(f"find.xlsx文件: {find_file}")
    print(f"输出目录: {output_dir}")
    print(f"重试次数: {args.retries}")
    print(f"超时时间: {args.timeout}秒")
    print("=" * 70)
    
    # 1. 读取find.xlsx
    find_pairs = read_find_excel(find_file)
    
    if not find_pairs:
        print("错误: find.xlsx中没有有效数据")
        sys.exit(1)
    
    # 2. 下载并保存内容
    success, failed, skipped = download_and_save_content(find_pairs, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ 完成！")
    print(f"内容已保存到: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
