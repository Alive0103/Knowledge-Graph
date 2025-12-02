import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_evaluation_reports():
    """加载所有评估报告"""
    reports = []
    for filename in os.listdir('.'):
        if filename.startswith('evaluation_report_') and filename.endswith('.json'):
            with open(filename, 'r', encoding='utf-8') as f:
                report = json.load(f)
                reports.append(report)
    return reports

def plot_metrics_comparison(reports):
    """绘制不同检索模式的指标对比图"""
    modes = []
    mrr_scores = []
    hit1_scores = []
    hit5_scores = []
    hit10_scores = []
    
    for report in reports:
        mode = report['search_mode']
        metrics = report['metrics']
        
        modes.append(mode)
        mrr_scores.append(metrics['mrr'])
        hit1_scores.append(metrics['hit@1'])
        hit5_scores.append(metrics['hit@5'])
        hit10_scores.append(metrics['hit@10'])
    
    # 创建柱状图
    x = np.arange(len(modes))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - 1.5*width, mrr_scores, width, label='MRR')
    rects2 = ax.bar(x - 0.5*width, hit1_scores, width, label='Hit@1')
    rects3 = ax.bar(x + 0.5*width, hit5_scores, width, label='Hit@5')
    rects4 = ax.bar(x + 1.5*width, hit10_scores, width, label='Hit@10')
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    ax.set_xlabel('检索模式')
    ax.set_ylabel('分数')
    ax.set_title('不同检索模式的评估指标对比')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_rank_distribution(reports):
    """绘制排名分布图"""
    fig, axes = plt.subplots(1, len(reports), figsize=(5*len(reports), 6))
    if len(reports) == 1:
        axes = [axes]
    
    for idx, report in enumerate(reports):
        mode = report['search_mode']
        detailed_results = report['detailed_results']
        
        # 统计排名分布（只统计前10名）
        rank_counts = defaultdict(int)
        total_valid = 0
        
        for result in detailed_results:
            if 'rank' in result and result['rank'] is not None:
                rank = min(result['rank'], 10)  # 将10名之后的都归为第10名
                rank_counts[rank] += 1
                total_valid += 1
        
        # 计算百分比
        ranks = list(range(1, 11))
        percentages = [rank_counts[r]/total_valid*100 for r in ranks]
        
        # 绘制柱状图
        axes[idx].bar(ranks, percentages, color='skyblue')
        axes[idx].set_xlabel('排名')
        axes[idx].set_ylabel('百分比 (%)')
        axes[idx].set_title(f'{mode} 检索模式排名分布')
        axes[idx].set_xticks(ranks)
        
        # 添加数值标签
        for i, v in enumerate(percentages):
            axes[idx].text(ranks[i], v + 0.5, f'{v:.1f}%', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('rank_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_processing_time_trend():
    """绘制处理时间趋势图（如果日志中有相关信息）"""
    # 这个功能需要从日志中解析处理时间信息
    # 作为示例，我们暂时不实现这个功能
    pass

def generate_summary_table(reports):
    """生成结果汇总表"""
    modes = []
    mrr_scores = []
    hit1_scores = []
    hit5_scores = []
    hit10_scores = []
    
    for report in reports:
        mode = report['search_mode']
        metrics = report['metrics']
        
        modes.append(mode)
        mrr_scores.append(metrics['mrr'])
        hit1_scores.append(metrics['hit@1'])
        hit5_scores.append(metrics['hit@5'])
        hit10_scores.append(metrics['hit@10'])
    
    # 打印表格
    print("\n评估结果汇总表:")
    print("=" * 60)
    print(f"{'模式':<12} {'MRR':<12} {'Hit@1':<12} {'Hit@5':<12} {'Hit@10':<12}")
    print("-" * 60)
    for i in range(len(modes)):
        print(f"{modes[i]:<12} {mrr_scores[i]:<12.4f} {hit1_scores[i]:<12.4f} {hit5_scores[i]:<12.4f} {hit10_scores[i]:<12.4f}")

def main():
    # 加载评估报告
    reports = load_evaluation_reports()
    
    if not reports:
        print("未找到评估报告文件，请先运行评估程序生成报告。")
        return
    
    print(f"找到 {len(reports)} 个评估报告")
    
    # 生成汇总表
    generate_summary_table(reports)
    
    # 绘制指标对比图
    print("\n正在生成指标对比图...")
    plot_metrics_comparison(reports)
    
    # 绘制排名分布图
    print("正在生成排名分布图...")
    plot_rank_distribution(reports)
    
    print("\n图表已生成并保存到当前目录。")

if __name__ == "__main__":
    main()