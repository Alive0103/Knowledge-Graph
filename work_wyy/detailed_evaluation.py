"""
详细评测脚本 - 记录所有异常和详细信息
"""
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from search_withllm import hybrid_search, generate_prompt_and_sort, normalize_url, clean_link, set_error_callback
import os
import time
import json
from datetime import datetime
from collections import defaultdict

class EvaluationLogger:
    """评测日志记录器"""
    def __init__(self, log_file="evaluation_log.jsonl", summary_file="evaluation_summary.json"):
        self.log_file = log_file
        self.summary_file = summary_file
        self.stats = {
            "total_queries": 0,
            "successful": 0,
            "failed": 0,
            "errors": defaultdict(int),
            "error_details": [],
            "query_details": [],
            "llm_errors": [],
            "vector_errors": [],
            "search_errors": [],
            "ranking_errors": []
        }
        self.start_time = None
        
    def log_query(self, query, correct_link, result):
        """记录单个查询的详细信息"""
        self.stats["query_details"].append({
            "query": query,
            "correct_link": correct_link,
            "rank": result.get("rank"),
            "found": result.get("found", False),
            "mrr": result.get("mrr", 0),
            "hit_at_1": result.get("hit_at_1", 0),
            "hit_at_5": result.get("hit_at_5", 0),
            "hit_at_10": result.get("hit_at_10", 0),
            "processing_time": result.get("processing_time", 0),
            "errors": result.get("errors", []),
            "warnings": result.get("warnings", [])
        })
        
        # 写入JSONL文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "correct_link": correct_link,
                "result": result
            }, ensure_ascii=False) + "\n")
    
    def log_error(self, error_type, query, error_msg, details=None):
        """记录错误"""
        self.stats["errors"][error_type] += 1
        error_info = {
            "type": error_type,
            "query": query,
            "error": str(error_msg),
            "timestamp": datetime.now().isoformat()
        }
        if details:
            error_info["details"] = details
        
        if error_type.startswith("llm"):
            self.stats["llm_errors"].append(error_info)
        elif error_type.startswith("vector"):
            self.stats["vector_errors"].append(error_info)
        elif error_type.startswith("search"):
            self.stats["search_errors"].append(error_info)
        elif error_type.startswith("ranking"):
            self.stats["ranking_errors"].append(error_info)
        
        self.stats["error_details"].append(error_info)
    
    def save_summary(self):
        """保存统计摘要"""
        if self.start_time:
            self.stats["total_time"] = time.time() - self.start_time
        
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "=" * 80)
        print("评测统计摘要")
        print("=" * 80)
        print(f"总查询数: {self.stats['total_queries']}")
        print(f"成功: {self.stats['successful']} ({self.stats['successful']/self.stats['total_queries']*100:.2f}%)")
        print(f"失败: {self.stats['failed']} ({self.stats['failed']/self.stats['total_queries']*100:.2f}%)")
        
        if self.stats["errors"]:
            print("\n错误统计:")
            for error_type, count in sorted(self.stats["errors"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")
        
        if self.start_time:
            print(f"\n总耗时: {self.stats.get('total_time', 0):.2f}秒")
            print(f"平均每个查询: {self.stats.get('total_time', 0)/self.stats['total_queries']:.2f}秒")
        
        print("=" * 80)

def read_excel(file_path):
    """读取评测文件"""
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    correct_links = df[1].tolist()
    return queries, correct_links

def process_query_detailed(query, correct_link, logger):
    """
    详细处理单个查询，记录所有异常
    """
    result = {
        "rank": None,
        "found": False,
        "mrr": 0,
        "hit_at_1": 0,
        "hit_at_5": 0,
        "hit_at_10": 0,
        "processing_time": 0,
        "errors": [],
        "warnings": []
    }
    
    start_time = time.time()
    
    try:
        # 步骤1: 执行搜索
        try:
            results = hybrid_search(query, top_k=20)
            if not results:
                result["warnings"].append("搜索结果为空")
                logger.log_error("search_empty", query, "搜索结果为空")
        except Exception as e:
            error_msg = f"搜索失败: {str(e)}"
            result["errors"].append(error_msg)
            logger.log_error("search_failed", query, str(e), {"error_type": type(e).__name__})
            return result
        
        # 步骤2: LLM重排序
        try:
            sorted_links = generate_prompt_and_sort(query, results)
            if not sorted_links:
                result["warnings"].append("重排序结果为空，使用原始结果")
                sorted_links = [r['link'] for r in results]
        except Exception as e:
            error_msg = f"LLM重排序失败: {str(e)}"
            result["errors"].append(error_msg)
            logger.log_error("ranking_failed", query, str(e), {"error_type": type(e).__name__})
            # 使用原始搜索结果
            sorted_links = [r['link'] for r in results]
            result["warnings"].append("使用原始搜索结果顺序")
        
        # 步骤3: 查找正确答案
        correct_link_cleaned = clean_link(str(correct_link))
        correct_link_normalized = normalize_url(correct_link_cleaned)
        
        for i, link in enumerate(sorted_links):
            link_cleaned = clean_link(str(link))
            link_normalized = normalize_url(link_cleaned)
            
            # 多种匹配方式
            matched = False
            match_method = None
            
            # 1. 归一化后的URL匹配
            if correct_link_normalized == link_normalized:
                matched = True
                match_method = "normalized_exact"
            
            # 2. 清理后的精确匹配
            if not matched and correct_link_cleaned == link_cleaned:
                matched = True
                match_method = "cleaned_exact"
            
            # 3. 双向子字符串匹配
            if not matched and (correct_link_cleaned in link_cleaned or link_cleaned in correct_link_cleaned):
                matched = True
                match_method = "substring"
            
            # 4. 归一化后的双向匹配
            if not matched and (correct_link_normalized in link_normalized or link_normalized in correct_link_normalized):
                matched = True
                match_method = "normalized_substring"
            
            if matched:
                result["rank"] = i + 1
                result["found"] = True
                result["match_method"] = match_method
                break
        
        # 计算指标
        if result["rank"] is not None:
            result["mrr"] = 1 / result["rank"]
            result["hit_at_1"] = 1 if result["rank"] <= 1 else 0
            result["hit_at_5"] = 1 if result["rank"] <= 5 else 0
            result["hit_at_10"] = 1 if result["rank"] <= 10 else 0
        else:
            result["warnings"].append("未找到正确答案")
            # 检查是否在搜索结果中
            found_in_search = False
            for r in results:
                r_link = r.get('link', '')
                if (correct_link_normalized == normalize_url(clean_link(r_link)) or
                    correct_link_cleaned in clean_link(r_link) or
                    clean_link(r_link) in correct_link_cleaned):
                    found_in_search = True
                    result["warnings"].append(f"正确答案在搜索结果中但未在重排序结果中（搜索排名: {results.index(r)+1}）")
                    break
            if not found_in_search:
                result["warnings"].append("正确答案不在搜索结果中")
        
    except Exception as e:
        error_msg = f"处理查询时发生未知错误: {str(e)}"
        result["errors"].append(error_msg)
        logger.log_error("unknown_error", query, str(e), {
            "error_type": type(e).__name__,
            "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
        })
    
    finally:
        result["processing_time"] = time.time() - start_time
    
    return result

def calculate_metrics_detailed(queries, correct_links, logger):
    """
    详细计算评测指标，记录所有异常
    """
    mrr = 0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    total_processed = 0
    
    logger.start_time = time.time()
    logger.stats["total_queries"] = len(queries)
    
    def process_with_logging(query, correct_link):
        result = process_query_detailed(query, correct_link, logger)
        
        if result["errors"]:
            logger.stats["failed"] += 1
        else:
            logger.stats["successful"] += 1
        
        logger.log_query(query, correct_link, result)
        
        return result["mrr"], result["hit_at_1"], result["hit_at_5"], result["hit_at_10"]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_with_logging, query, correct_link): (query, correct_link) 
                  for query, correct_link in zip(queries, correct_links)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(queries), 
                          desc="详细评测中"):
            try:
                result = future.result()
                mrr += result[0]
                hit_at_1 += result[1]
                hit_at_5 += result[2]
                hit_at_10 += result[3]
                total_processed += 1
            except Exception as e:
                query, correct_link = futures[future]
                logger.log_error("future_error", query, str(e))
    
    if total_processed > 0:
        mrr /= total_processed
        hit_at_1 /= total_processed
        hit_at_5 /= total_processed
        hit_at_10 /= total_processed
    
    return mrr, hit_at_1, hit_at_5, hit_at_10

def print_error_report(logger):
    """打印详细的错误报告"""
    print("\n" + "=" * 80)
    print("详细错误报告")
    print("=" * 80)
    
    # LLM错误
    if logger.stats["llm_errors"]:
        print(f"\n【LLM错误】共 {len(logger.stats['llm_errors'])} 个:")
        for i, error in enumerate(logger.stats["llm_errors"][:10], 1):  # 只显示前10个
            print(f"  {i}. 查询: '{error['query']}'")
            print(f"     错误: {error['error']}")
        if len(logger.stats["llm_errors"]) > 10:
            print(f"     ... 还有 {len(logger.stats['llm_errors']) - 10} 个错误")
    
    # 向量错误
    if logger.stats["vector_errors"]:
        print(f"\n【向量生成错误】共 {len(logger.stats['vector_errors'])} 个:")
        for i, error in enumerate(logger.stats["vector_errors"][:10], 1):
            print(f"  {i}. 查询: '{error['query']}'")
            print(f"     错误: {error['error']}")
    
    # 搜索错误
    if logger.stats["search_errors"]:
        print(f"\n【搜索错误】共 {len(logger.stats['search_errors'])} 个:")
        for i, error in enumerate(logger.stats["search_errors"][:10], 1):
            print(f"  {i}. 查询: '{error['query']}'")
            print(f"     错误: {error['error']}")
    
    # 重排序错误
    if logger.stats["ranking_errors"]:
        print(f"\n【重排序错误】共 {len(logger.stats['ranking_errors'])} 个:")
        for i, error in enumerate(logger.stats["ranking_errors"][:10], 1):
            print(f"  {i}. 查询: '{error['query']}'")
            print(f"     错误: {error['error']}")
    
    print("=" * 80)

def print_warning_summary(logger):
    """打印警告摘要"""
    warnings = defaultdict(int)
    for query_detail in logger.stats["query_details"]:
        for warning in query_detail.get("warnings", []):
            warnings[warning] += 1
    
    if warnings:
        print("\n" + "=" * 80)
        print("警告统计")
        print("=" * 80)
        for warning, count in sorted(warnings.items(), key=lambda x: x[1], reverse=True):
            print(f"  {warning}: {count} 次")
        print("=" * 80)

def analyze_failed_queries(logger):
    """分析失败的查询"""
    failed_queries = [qd for qd in logger.stats["query_details"] if not qd.get("found", False)]
    
    if failed_queries:
        print("\n" + "=" * 80)
        print(f"失败查询分析（共 {len(failed_queries)} 个）")
        print("=" * 80)
        
        # 按警告类型分组
        by_warning = defaultdict(list)
        for qd in failed_queries:
            warnings = qd.get("warnings", [])
            if warnings:
                key = "; ".join(warnings)
                by_warning[key].append(qd["query"])
            else:
                by_warning["无警告信息"].append(qd["query"])
        
        for warning, queries in sorted(by_warning.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n{warning}: {len(queries)} 个查询")
            print(f"  示例: {', '.join(queries[:5])}")
            if len(queries) > 5:
                print(f"  ... 还有 {len(queries) - 5} 个")
        
        print("=" * 80)

def main():
    print("=" * 80)
    print("详细评测 - 记录所有异常情况")
    print("=" * 80)
    print()
    
    file_path = "data/find.xlsx"
    
    if not os.path.exists(file_path):
        print(f"错误: 未找到评测文件: {file_path}")
        return
    
    # 创建日志记录器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = EvaluationLogger(
        log_file=f"evaluation_log_{timestamp}.jsonl",
        summary_file=f"evaluation_summary_{timestamp}.json"
    )
    
    # 设置错误回调，让search_withllm模块的错误也能被记录
    def error_callback(error_type, query, error_msg, details=None):
        logger.log_error(error_type, query, error_msg, details)
    
    set_error_callback(error_callback)
    
    try:
        queries, correct_links = read_excel(file_path)
        print(f"读取了 {len(queries)} 个查询")
        print(f"日志文件: {logger.log_file}")
        print(f"摘要文件: {logger.summary_file}")
        print()
        
        print("开始详细评测...")
        mrr, hit_at_1, hit_at_5, hit_at_10 = calculate_metrics_detailed(queries, correct_links, logger)
        
        # 保存摘要
        logger.save_summary()
        
        # 打印结果
        print(f"\n{'='*80}")
        print("评测结果")
        print(f"{'='*80}")
        print(f"MRR: {mrr:.4f}")
        print(f"Hit@1: {hit_at_1:.4f}")
        print(f"Hit@5: {hit_at_5:.4f}")
        print(f"Hit@10: {hit_at_10:.4f}")
        
        # 打印统计摘要
        logger.print_summary()
        
        # 打印错误报告
        print_error_report(logger)
        
        # 打印警告摘要
        print_warning_summary(logger)
        
        # 分析失败查询
        analyze_failed_queries(logger)
        
        print(f"\n详细日志已保存到: {logger.log_file}")
        print(f"统计摘要已保存到: {logger.summary_file}")
        
    except Exception as e:
        print(f"评测失败: {e}")
        import traceback
        traceback.print_exc()
        logger.save_summary()

if __name__ == "__main__":
    main()

