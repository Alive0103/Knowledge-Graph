"""
向量检索效果对比脚本
对比启用和禁用向量检索的评测指标
"""
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from search_withllm import hybrid_search, generate_prompt_and_sort, normalize_url, clean_link
import os
import time

def read_excel(file_path):
    """读取评测文件"""
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    correct_links = df[1].tolist()
    return queries, correct_links

def calculate_metrics(queries, correct_links, use_vector=True, text_boost=1.0, vector_boost=0.8):
    """
    计算评测指标
    
    Args:
        queries: 查询列表
        correct_links: 正确答案列表
        use_vector: 是否使用向量检索
        text_boost: 文本检索boost
        vector_boost: 向量检索boost
    
    Returns:
        (mrr, hit_at_1, hit_at_5, hit_at_10, total_processed)
    """
    mrr = 0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    total_processed = 0

    def process_query(query, correct_link):
        try:
            # 使用指定的参数进行检索
            results = hybrid_search(query, top_k=20, text_boost=text_boost, 
                                   vector_boost=vector_boost, use_vector=use_vector)
            sorted_links = generate_prompt_and_sort(query, results)
            rank = None

            # 改进的链接匹配：支持双向匹配和清理后的匹配
            correct_link_cleaned = clean_link(str(correct_link))
            correct_link_normalized = normalize_url(correct_link_cleaned)
            
            for i, link in enumerate(sorted_links):
                link_cleaned = clean_link(str(link))
                link_normalized = normalize_url(link_cleaned)
                
                # 多种匹配方式
                # 1. 归一化后的URL匹配（处理URL编码问题）
                if correct_link_normalized == link_normalized:
                    rank = i + 1
                    break
                
                # 2. 清理后的精确匹配
                if correct_link_cleaned == link_cleaned:
                    rank = i + 1
                    break
                
                # 3. 双向子字符串匹配
                if correct_link_cleaned in link_cleaned or link_cleaned in correct_link_cleaned:
                    rank = i + 1
                    break
                
                # 4. 归一化后的双向匹配
                if correct_link_normalized in link_normalized or link_normalized in correct_link_normalized:
                    rank = i + 1
                    break

            if rank is not None:
                return 1 / rank, 1 if rank <= 1 else 0, 1 if rank <= 5 else 0, 1 if rank <= 10 else 0
            return 0, 0, 0, 0
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            return 0, 0, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_query, query, correct_link) 
                   for query, correct_link in zip(queries, correct_links)]
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(queries), 
                          desc=f"Processing queries ({'向量检索' if use_vector else '仅文本检索'})"):
            result = future.result()
            mrr += result[0]
            hit_at_1 += result[1]
            hit_at_5 += result[2]
            hit_at_10 += result[3]
            total_processed += 1

    if total_processed > 0:
        mrr /= total_processed
        hit_at_1 /= total_processed
        hit_at_5 /= total_processed
        hit_at_10 /= total_processed
    else:
        print("No queries were processed successfully.")
        return 0, 0, 0, 0, 0

    return mrr, hit_at_1, hit_at_5, hit_at_10, total_processed

def format_percentage(value):
    """格式化百分比"""
    return f"{value*100:.2f}%"

def format_improvement(old, new):
    """计算并格式化提升"""
    if old == 0:
        return "N/A" if new == 0 else "∞"
    improvement = ((new - old) / old) * 100
    sign = "+" if improvement >= 0 else ""
    return f"{sign}{improvement:.2f}%"

def main():
    print("=" * 80)
    print("向量检索效果对比")
    print("=" * 80)
    print()
    
    file_path = "data/find.xlsx"
    
    if not os.path.exists(file_path):
        print(f"错误: 未找到评测文件: {file_path}")
        print("请确保 find.xlsx 文件存在于当前目录")
        return
    
    try:
        queries, correct_links = read_excel(file_path)
        print(f"读取了 {len(queries)} 个查询")
        print()
        
        # 测试1: 仅文本检索（禁用向量检索）
        print("=" * 80)
        print("测试1: 仅文本检索（禁用向量检索）")
        print("=" * 80)
        start_time = time.time()
        mrr_text, hit1_text, hit5_text, hit10_text, total_text = calculate_metrics(
            queries, correct_links, use_vector=False, text_boost=1.0
        )
        time_text = time.time() - start_time
        
        print(f"\n仅文本检索结果:")
        print(f"  MRR: {mrr_text:.4f}")
        print(f"  Hit@1: {hit1_text:.4f} ({format_percentage(hit1_text)})")
        print(f"  Hit@5: {hit5_text:.4f} ({format_percentage(hit5_text)})")
        print(f"  Hit@10: {hit10_text:.4f} ({format_percentage(hit10_text)})")
        print(f"  处理时间: {time_text:.2f}秒")
        print()
        
        # 测试2: 混合检索（启用向量检索）
        print("=" * 80)
        print("测试2: 混合检索（文本 + 向量检索）")
        print("=" * 80)
        start_time = time.time()
        mrr_hybrid, hit1_hybrid, hit5_hybrid, hit10_hybrid, total_hybrid = calculate_metrics(
            queries, correct_links, use_vector=True, text_boost=1.0, vector_boost=0.8
        )
        time_hybrid = time.time() - start_time
        
        print(f"\n混合检索结果:")
        print(f"  MRR: {mrr_hybrid:.4f}")
        print(f"  Hit@1: {hit1_hybrid:.4f} ({format_percentage(hit1_hybrid)})")
        print(f"  Hit@5: {hit5_hybrid:.4f} ({format_percentage(hit5_hybrid)})")
        print(f"  Hit@10: {hit10_hybrid:.4f} ({format_percentage(hit10_hybrid)})")
        print(f"  处理时间: {time_hybrid:.2f}秒")
        print()
        
        # 对比结果
        print("=" * 80)
        print("对比结果")
        print("=" * 80)
        print(f"{'指标':<15} {'仅文本检索':<20} {'混合检索':<20} {'提升':<15}")
        print("-" * 80)
        print(f"{'MRR':<15} {mrr_text:.4f} ({format_percentage(mrr_text):<6}) {mrr_hybrid:.4f} ({format_percentage(mrr_hybrid):<6}) {format_improvement(mrr_text, mrr_hybrid):<15}")
        print(f"{'Hit@1':<15} {hit1_text:.4f} ({format_percentage(hit1_text):<6}) {hit1_hybrid:.4f} ({format_percentage(hit1_hybrid):<6}) {format_improvement(hit1_text, hit1_hybrid):<15}")
        print(f"{'Hit@5':<15} {hit5_text:.4f} ({format_percentage(hit5_text):<6}) {hit5_hybrid:.4f} ({format_percentage(hit5_hybrid):<6}) {format_improvement(hit5_text, hit5_hybrid):<15}")
        print(f"{'Hit@10':<15} {hit10_text:.4f} ({format_percentage(hit10_text):<6}) {hit10_hybrid:.4f} ({format_percentage(hit10_hybrid):<6}) {format_improvement(hit10_text, hit10_hybrid):<15}")
        print(f"{'处理时间':<15} {time_text:.2f}秒{'':<13} {time_hybrid:.2f}秒{'':<13} {format_improvement(time_text, time_hybrid):<15}")
        print()
        
        # 计算绝对提升
        print("=" * 80)
        print("绝对提升")
        print("=" * 80)
        print(f"MRR提升: {mrr_hybrid - mrr_text:+.4f} (相对提升: {format_improvement(mrr_text, mrr_hybrid)})")
        print(f"Hit@1提升: {hit1_hybrid - hit1_text:+.4f} (相对提升: {format_improvement(hit1_text, hit1_hybrid)})")
        print(f"Hit@5提升: {hit5_hybrid - hit5_text:+.4f} (相对提升: {format_improvement(hit5_text, hit5_hybrid)})")
        print(f"Hit@10提升: {hit10_hybrid - hit10_text:+.4f} (相对提升: {format_improvement(hit10_text, hit10_hybrid)})")
        print(f"时间增加: {time_hybrid - time_text:+.2f}秒 (相对增加: {format_improvement(time_text, time_hybrid)})")
        print()
        
        # 总结
        print("=" * 80)
        print("总结")
        print("=" * 80)
        if mrr_hybrid > mrr_text:
            print("✓ 向量检索提升了整体性能")
            print(f"  - MRR从 {mrr_text:.4f} 提升到 {mrr_hybrid:.4f}")
            print(f"  - 相对提升: {format_improvement(mrr_text, mrr_hybrid)}")
        elif mrr_hybrid == mrr_text:
            print("= 向量检索与文本检索性能相同")
        else:
            print("✗ 向量检索未提升性能（可能需要调整权重或检查向量字段）")
        
        if hit10_hybrid > hit10_text:
            print(f"✓ Hit@10从 {format_percentage(hit10_text)} 提升到 {format_percentage(hit10_hybrid)}")
            print(f"  - 相对提升: {format_improvement(hit10_text, hit10_hybrid)}")
        
        print(f"\n处理时间: 混合检索比文本检索多用了 {time_hybrid - time_text:.2f}秒")
        print("=" * 80)
        
    except Exception as e:
        print(f"评测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

