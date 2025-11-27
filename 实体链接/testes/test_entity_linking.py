"""
实体链接诊断工具
用于分析为什么评测指标这么低
"""
import pandas as pd
from search_withllm import hybrid_search, generate_prompt_and_sort, normalize_url, clean_link
from tqdm import tqdm
import json

def diagnose_sample_queries(file_path="find.xlsx", sample_size=10):
    """诊断样本查询，找出问题所在"""
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    correct_links = df[1].tolist()
    
    print("=" * 80)
    print("实体链接诊断报告")
    print("=" * 80)
    print(f"总查询数: {len(queries)}")
    print(f"诊断样本数: {min(sample_size, len(queries))}\n")
    
    # 统计信息
    stats = {
        "total": 0,
        "found_in_search": 0,  # 在搜索结果中找到正确答案
        "found_after_rerank": 0,  # 重排序后找到正确答案
        "not_found_in_search": 0,  # 搜索结果中没有正确答案
        "not_found_after_rerank": 0,  # 重排序后没有正确答案
        "llm_parse_errors": 0,  # LLM解析错误
        "link_format_mismatch": []  # 链接格式不匹配的案例
    }
    
    # 详细诊断前N个查询
    for idx in tqdm(range(min(sample_size, len(queries))), desc="诊断中"):
        query = queries[idx]
        correct_link = str(correct_links[idx])
        stats["total"] += 1
        
        print(f"\n{'='*80}")
        print(f"查询 #{idx+1}: {query}")
        print(f"正确答案: {correct_link}")
        print(f"{'='*80}")
        
        try:
            # 步骤1: 执行搜索
            results = hybrid_search(query, top_k=20)
            print(f"\n[步骤1] 搜索结果: 找到 {len(results)} 个候选实体")
            
            # 检查正确答案是否在搜索结果中
            found_in_search = False
            search_rank = None
            correct_link_cleaned = clean_link(str(correct_link))
            correct_link_normalized = normalize_url(correct_link_cleaned)
            
            for i, result in enumerate(results):
                result_link = result.get('link', '')
                result_link_cleaned = clean_link(str(result_link))
                result_link_normalized = normalize_url(result_link_cleaned)
                
                # 使用改进的匹配逻辑（与主代码一致）
                # 1. 归一化后的URL匹配（处理URL编码问题）
                if correct_link_normalized == result_link_normalized:
                    found_in_search = True
                    search_rank = i + 1
                    print(f"  ✓ 在搜索结果第 {search_rank} 位找到正确答案")
                    print(f"    匹配的链接: {result_link}")
                    stats["found_in_search"] += 1
                    break
                
                # 2. 清理后的精确匹配
                if correct_link_cleaned == result_link_cleaned:
                    found_in_search = True
                    search_rank = i + 1
                    print(f"  ✓ 在搜索结果第 {search_rank} 位找到正确答案")
                    print(f"    匹配的链接: {result_link}")
                    stats["found_in_search"] += 1
                    break
                
                # 3. 双向子字符串匹配
                if correct_link_cleaned in result_link_cleaned or result_link_cleaned in correct_link_cleaned:
                    found_in_search = True
                    search_rank = i + 1
                    print(f"  ✓ 在搜索结果第 {search_rank} 位找到正确答案")
                    print(f"    匹配的链接: {result_link}")
                    stats["found_in_search"] += 1
                    break
                
                # 4. 归一化后的双向匹配
                if correct_link_normalized in result_link_normalized or result_link_normalized in correct_link_normalized:
                    found_in_search = True
                    search_rank = i + 1
                    print(f"  ✓ 在搜索结果第 {search_rank} 位找到正确答案")
                    print(f"    匹配的链接: {result_link}")
                    stats["found_in_search"] += 1
                    break
            
            if not found_in_search:
                print(f"  ✗ 搜索结果中未找到正确答案")
                stats["not_found_in_search"] += 1
                print(f"  前5个搜索结果:")
                for i, result in enumerate(results[:5], 1):
                    print(f"    {i}. {result.get('label', 'N/A')} -> {result.get('link', 'N/A')}")
            
            # 步骤2: LLM重排序
            print(f"\n[步骤2] LLM重排序...")
            try:
                sorted_links = generate_prompt_and_sort(query, results)
                print(f"  重排序后得到 {len(sorted_links)} 个链接")
                
                # 检查正确答案是否在重排序结果中
                found_after_rerank = False
                rerank_rank = None
                correct_link_cleaned = clean_link(str(correct_link))
                correct_link_normalized = normalize_url(correct_link_cleaned)
                
                for i, link in enumerate(sorted_links):
                    link_cleaned = clean_link(str(link))
                    link_normalized = normalize_url(link_cleaned)
                    
                    # 使用改进的匹配逻辑（与主代码一致）
                    # 1. 归一化后的URL匹配（处理URL编码问题）
                    if correct_link_normalized == link_normalized:
                        found_after_rerank = True
                        rerank_rank = i + 1
                        print(f"  ✓ 重排序后第 {rerank_rank} 位找到正确答案")
                        print(f"    匹配的链接: {link}")
                        stats["found_after_rerank"] += 1
                        break
                    
                    # 2. 清理后的精确匹配
                    if correct_link_cleaned == link_cleaned:
                        found_after_rerank = True
                        rerank_rank = i + 1
                        print(f"  ✓ 重排序后第 {rerank_rank} 位找到正确答案")
                        print(f"    匹配的链接: {link}")
                        stats["found_after_rerank"] += 1
                        break
                    
                    # 3. 双向子字符串匹配
                    if correct_link_cleaned in link_cleaned or link_cleaned in correct_link_cleaned:
                        found_after_rerank = True
                        rerank_rank = i + 1
                        print(f"  ✓ 重排序后第 {rerank_rank} 位找到正确答案")
                        print(f"    匹配的链接: {link}")
                        stats["found_after_rerank"] += 1
                        break
                    
                    # 4. 归一化后的双向匹配
                    if correct_link_normalized in link_normalized or link_normalized in correct_link_normalized:
                        found_after_rerank = True
                        rerank_rank = i + 1
                        print(f"  ✓ 重排序后第 {rerank_rank} 位找到正确答案")
                        print(f"    匹配的链接: {link}")
                        stats["found_after_rerank"] += 1
                        break
                
                if not found_after_rerank:
                    print(f"  ✗ 重排序后未找到正确答案")
                    stats["not_found_after_rerank"] += 1
                    print(f"  前5个重排序结果:")
                    for i, link in enumerate(sorted_links[:5], 1):
                        print(f"    {i}. {link}")
                    
                    # 检查是否是格式问题
                    print(f"\n  链接格式检查:")
                    print(f"    正确答案格式: '{correct_link}' (长度: {len(correct_link)})")
                    print(f"    归一化后: '{correct_link_normalized}'")
                    print(f"    重排序结果中的链接示例:")
                    for i, link in enumerate(sorted_links[:3], 1):
                        link_normalized = normalize_url(clean_link(str(link)))
                        print(f"      {i}. '{link}' (长度: {len(link)})")
                        print(f"         归一化后: '{link_normalized}'")
                        # 使用归一化后的URL进行比较
                        if correct_link_normalized != link_normalized:
                            print(f"        ⚠ 归一化后仍不匹配!")
                            if idx < 5:  # 只记录前5个不匹配的案例
                                stats["link_format_mismatch"].append({
                                    "query": query,
                                    "correct": correct_link,
                                    "correct_normalized": correct_link_normalized,
                                    "got": link,
                                    "got_normalized": link_normalized
                                })
                        else:
                            print(f"        ✓ 归一化后匹配!")
                
                # 对比搜索和重排序的排名变化
                if found_in_search and found_after_rerank:
                    if search_rank != rerank_rank:
                        print(f"\n  📊 排名变化: 搜索第{search_rank}位 -> 重排序第{rerank_rank}位")
                        if rerank_rank > search_rank:
                            print(f"     ⚠ LLM重排序把正确答案排到了更后面!")
                    else:
                        print(f"\n  ✓ 排名未变化: 都是第{search_rank}位")
                
            except Exception as e:
                print(f"  ✗ LLM重排序失败: {e}")
                stats["llm_parse_errors"] += 1
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"\n  ✗ 处理查询时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 输出统计报告
    print(f"\n\n{'='*80}")
    print("诊断统计报告")
    print(f"{'='*80}")
    print(f"总查询数: {stats['total']}")
    print(f"在搜索结果中找到: {stats['found_in_search']} ({stats['found_in_search']/stats['total']*100:.1f}%)")
    print(f"重排序后找到: {stats['found_after_rerank']} ({stats['found_after_rerank']/stats['total']*100:.1f}%)")
    print(f"搜索结果中未找到: {stats['not_found_in_search']} ({stats['not_found_in_search']/stats['total']*100:.1f}%)")
    print(f"重排序后未找到: {stats['not_found_after_rerank']} ({stats['not_found_after_rerank']/stats['total']*100:.1f}%)")
    print(f"LLM解析错误: {stats['llm_parse_errors']} ({stats['llm_parse_errors']/stats['total']*100:.1f}%)")
    
    if stats['link_format_mismatch']:
        print(f"\n链接格式不匹配案例 (前5个):")
        for case in stats['link_format_mismatch'][:5]:
            print(f"  查询: {case['query']}")
            print(f"    正确答案: '{case['correct']}'")
            print(f"    实际得到: '{case['got']}'")
    
    # 分析可能的问题
    print(f"\n{'='*80}")
    print("问题分析")
    print(f"{'='*80}")
    
    if stats['not_found_in_search'] / stats['total'] > 0.5:
        print("⚠ 主要问题: 搜索结果中找不到正确答案")
        print("  建议: 检查Elasticsearch索引数据，确认正确答案是否在知识库中")
    
    if stats['not_found_after_rerank'] > stats['not_found_in_search']:
        print("⚠ 主要问题: LLM重排序导致正确答案丢失")
        print("  建议: 检查LLM返回的链接格式，确保与正确答案格式一致")
    
    if stats['llm_parse_errors'] / stats['total'] > 0.1:
        print("⚠ 主要问题: LLM解析错误率较高")
        print("  建议: 检查LLM prompt，确保返回格式稳定")
    
    if stats['found_after_rerank'] / stats['total'] < 0.1:
        print("⚠ 主要问题: 整体准确率过低")
        print("  建议: 检查搜索策略和LLM重排序逻辑")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    diagnose_sample_queries(sample_size=20)  # 诊断前20个查询

