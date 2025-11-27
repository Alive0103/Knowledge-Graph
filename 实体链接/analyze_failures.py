"""
分析评测失败案例，提供改进建议
"""
import json
from collections import defaultdict
from search_withllm import hybrid_search, normalize_url, clean_link

def load_evaluation_log(log_file):
    """加载评测日志"""
    queries = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

def analyze_failures(log_file):
    """分析失败案例"""
    queries = load_evaluation_log(log_file)
    
    # 分类失败案例
    empty_search = []  # 搜索结果为空
    not_in_results = []  # 正确答案不在搜索结果中
    low_rank = []  # 排名较低（>10）
    
    for item in queries:
        result = item['result']
        query = item['query']
        correct_link = item['correct_link']
        
        if not result.get('found', False):
            if '搜索结果为空' in result.get('warnings', []):
                empty_search.append({
                    'query': query,
                    'correct_link': correct_link,
                    'warnings': result.get('warnings', [])
                })
            elif '正确答案不在搜索结果中' in result.get('warnings', []):
                not_in_results.append({
                    'query': query,
                    'correct_link': correct_link,
                    'warnings': result.get('warnings', [])
                })
        elif result.get('rank', 0) > 10:
            low_rank.append({
                'query': query,
                'correct_link': correct_link,
                'rank': result.get('rank'),
                'warnings': result.get('warnings', [])
            })
    
    return empty_search, not_in_results, low_rank

def diagnose_query(query_text, correct_link):
    """诊断单个查询，尝试找出问题"""
    print(f"\n{'='*80}")
    print(f"诊断查询: '{query_text}'")
    print(f"正确答案: {correct_link}")
    print(f"{'='*80}")
    
    # 1. 尝试搜索
    try:
        results = hybrid_search(query_text, top_k=20)
        print(f"\n【搜索结果】找到 {len(results)} 个候选实体")
        
        if len(results) == 0:
            print("  ❌ 搜索结果为空！")
            print("\n【可能原因】")
            print("  1. 查询文本太短或太模糊")
            print("  2. 知识库中可能没有相关实体")
            print("  3. 查询文本与实体名称差异太大")
            print("\n【建议】")
            print("  - 尝试使用更完整的查询文本")
            print("  - 检查知识库中是否存在相关实体")
            print("  - 考虑添加查询扩展或同义词")
            return
        
        # 2. 检查正确答案是否在搜索结果中
        correct_link_cleaned = clean_link(str(correct_link))
        correct_link_normalized = normalize_url(correct_link_cleaned)
        
        found_in_search = False
        search_rank = None
        
        for i, result in enumerate(results):
            result_link = result.get('link', '')
            result_link_cleaned = clean_link(str(result_link))
            result_link_normalized = normalize_url(result_link_cleaned)
            
            if (correct_link_normalized == result_link_normalized or
                correct_link_cleaned in result_link_cleaned or
                result_link_cleaned in correct_link_cleaned):
                found_in_search = True
                search_rank = i + 1
                break
        
        if found_in_search:
            print(f"  ✓ 正确答案在搜索结果中（排名: {search_rank}）")
            print(f"\n【问题分析】")
            print(f"  正确答案在搜索结果中，但LLM重排序后可能排到了后面")
            print(f"\n【建议】")
            print("  - 改进LLM重排序的prompt")
            print("  - 增加正确答案的特征信息")
            print("  - 检查LLM返回的排序结果")
        else:
            print("  ❌ 正确答案不在搜索结果中")
            print(f"\n【前5个搜索结果】")
            for i, result in enumerate(results[:5], 1):
                print(f"  {i}. {result.get('label', 'N/A')}")
                print(f"     链接: {result.get('link', 'N/A')[:80]}...")
            
            print(f"\n【可能原因】")
            print("  1. 查询文本与实体名称不匹配")
            print("  2. 实体在知识库中的名称与查询文本差异较大")
            print("  3. 向量检索或文本检索都未能召回该实体")
            print(f"\n【建议】")
            print("  - 检查知识库中该实体的实际名称和别名")
            print("  - 尝试使用实体的其他名称或别名进行查询")
            print("  - 考虑改进查询扩展策略")
        
    except Exception as e:
        print(f"  ❌ 搜索失败: {e}")

def main():
    log_file = "evaluation_log_20251127_135720.jsonl"
    
    print("="*80)
    print("失败案例分析工具")
    print("="*80)
    
    # 分析失败案例
    empty_search, not_in_results, low_rank = analyze_failures(log_file)
    
    print(f"\n【失败统计】")
    print(f"  搜索结果为空: {len(empty_search)} 个")
    print(f"  正确答案不在搜索结果中: {len(not_in_results)} 个")
    print(f"  排名较低(>10): {len(low_rank)} 个")
    
    # 详细分析搜索结果为空的案例
    if empty_search:
        print(f"\n{'='*80}")
        print("【搜索结果为空的案例】")
        print(f"{'='*80}")
        print("\n这些查询的搜索策略需要改进：")
        for i, case in enumerate(empty_search[:10], 1):
            print(f"\n{i}. 查询: '{case['query']}'")
            print(f"   正确答案: {case['correct_link'][:80]}...")
            diagnose_query(case['query'], case['correct_link'])
    
    # 详细分析正确答案不在搜索结果中的案例
    if not_in_results:
        print(f"\n{'='*80}")
        print("【正确答案不在搜索结果中的案例（前5个）】")
        print(f"{'='*80}")
        for i, case in enumerate(not_in_results[:5], 1):
            print(f"\n{i}. 查询: '{case['query']}'")
            print(f"   正确答案: {case['correct_link'][:80]}...")
            diagnose_query(case['query'], case['correct_link'])
    
    # 提供改进建议
    print(f"\n{'='*80}")
    print("【总体改进建议】")
    print(f"{'='*80}")
    print("\n1. 【搜索结果为空的问题】")
    print("   - 对于短查询（如'FARA', 'Mk41'），考虑添加查询扩展")
    print("   - 使用更宽松的搜索策略（降低匹配阈值）")
    print("   - 检查知识库中是否存在这些实体")
    print("   - 考虑使用模糊匹配或部分匹配")
    
    print("\n2. 【正确答案不在搜索结果中的问题】")
    print("   - 改进查询文本的预处理（提取关键词）")
    print("   - 增强向量检索的召回能力")
    print("   - 检查实体别名是否完整")
    print("   - 考虑使用更丰富的实体描述信息")
    
    print("\n3. 【排名较低的问题】")
    print("   - 改进LLM重排序的prompt")
    print("   - 增加正确答案的特征权重")
    print("   - 优化混合检索的权重配置")
    
    print("\n4. 【数据质量改进】")
    print("   - 检查知识库中实体的完整性和准确性")
    print("   - 确保实体别名列表完整")
    print("   - 验证实体描述的准确性")

if __name__ == "__main__":
    main()

