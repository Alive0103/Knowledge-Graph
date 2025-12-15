#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化训练与测试流水线
1. 训练NER模型
2. 测试模型效果
3. 使用测试通过的模型对数据集进行向量化处理并存入ES
4. 正式测试向量检索系统

使用方法:
    python auto_pipeline.py                    # 完整流程（默认）
    python auto_pipeline.py --from test        # 从测试阶段开始
    python auto_pipeline.py --from extract      # 从实体词提取开始
    python auto_pipeline.py --from vectorize   # 从向量化开始
    python auto_pipeline.py --from final_test  # 只运行最终测试
    python auto_pipeline.py --skip check       # 跳过前置检查
    python auto_pipeline.py --skip train       # 跳过训练
    python auto_pipeline.py --skip test        # 跳过测试
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime

# 获取脚本所在目录（work_wyy）
script_dir = os.path.dirname(os.path.abspath(__file__))

# 创建 trainlog 文件夹（如果不存在）
trainlog_dir = os.path.join(script_dir, 'trainlog')
os.makedirs(trainlog_dir, exist_ok=True)

# 创建日志文件名（保存到 trainlog 文件夹）
log_filename = os.path.join(trainlog_dir, f'auto_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建Tee类，同时输出到控制台和文件
class Tee:
    """同时将输出写入文件和控制台"""
    def __init__(self, file_path, mode='a', encoding='utf-8'):
        self.file = open(file_path, mode, encoding=encoding)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, text):
        self.file.write(text)
        self.file.flush()  # 立即刷新到文件
        self.stdout.write(text)
        self.stdout.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        if self.file:
            self.file.close()

# 重定向stdout和stderr到文件和控制台（保存到 trainlog 文件夹）
console_log_file = os.path.join(trainlog_dir, f'auto_pipeline_console_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
tee = Tee(console_log_file, mode='w')
sys.stdout = tee
sys.stderr = tee

logger.info(f"所有控制台输出将同时保存到: {console_log_file}")
logger.info(f"日志信息保存到: {log_filename}")


def run_command(cmd, cwd=None, description="", capture_output=False):
    """
    运行命令并记录输出
    
    Args:
        cmd: 命令列表
        cwd: 工作目录
        description: 命令描述
        capture_output: 是否捕获输出（False时实时显示，True时缓冲后显示）
    
    Returns:
        (success: bool, output: str): 成功标志和输出内容
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"执行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")
    logger.info(f"{'=' * 70}\n")
    
    try:
        if capture_output:
            # 捕获输出模式（用于需要检查输出的场景）
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            logger.info(f"✅ {description} 成功完成")
            return True, output
        else:
            # 实时输出模式（用于显示进度条等）
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                # 不使用 capture_output，让输出直接流到控制台
                text=True,
                encoding='utf-8'
            )
            logger.info(f"\n✅ {description} 成功完成")
            return True, ""
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} 失败")
        logger.error(f"错误代码: {e.returncode}")
        if capture_output:
            if e.stdout:
                logger.error(f"标准输出:\n{e.stdout}")
            if e.stderr:
                logger.error(f"错误输出:\n{e.stderr}")
            return False, e.stderr or e.stdout or ""
        else:
            return False, ""


def check_prerequisites():
    """检查前置条件"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤 0: 检查前置条件")
    logger.info("=" * 70)
    
    check_script = os.path.join(script_dir, "ner", "check_prerequisites.py")
    
    # 检查NER前置条件（实时显示输出）
    success, _ = run_command(
        [sys.executable, check_script],
        cwd=script_dir,
        description="检查NER训练前置条件",
        capture_output=False  # 实时显示
    )
    
    if not success:
        logger.warning("⚠️  前置条件检查有警告，但继续执行...")
    
    return success


def train_ner_model():
    """训练NER模型"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤 1: 训练NER模型")
    logger.info("=" * 70)
    
    ner_script = os.path.join(script_dir, "ner", "finetune_ner_model.py")
    
    success, output = run_command(
        [sys.executable, ner_script],
        cwd=script_dir,
        description="训练NER模型",
        capture_output=False  # 实时显示进度条和训练过程
    )
    
    if not success:
        logger.error("❌ NER模型训练失败，终止流水线")
        return False
    
    # 检查模型是否生成（相对于 work_wyy 目录）
    # 模型保存在 ner/finetune_ner_model.py 中，路径是 './../model/ner_finetuned'
    # 相对于 ner/ 目录，所以相对于 work_wyy 应该是 model/ner_finetuned
    model_path = os.path.join(script_dir, "model", "ner_finetuned")
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型文件不存在: {model_path}")
        logger.error(f"   请检查模型是否已成功保存")
        return False
    
    logger.info(f"✅ 模型已保存到: {model_path}")
    return True


def test_ner_model():
    """测试NER模型效果"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤 2: 测试NER模型效果")
    logger.info("=" * 70)
    logger.info("注意: 如果模型测试不通过，流水线将终止")
    logger.info("=" * 70)
    
    diagnose_script = os.path.join(script_dir, "ner", "diagnose_ner_model.py")
    
    # 先实时显示输出（让用户看到评测过程）
    logger.info("\n开始运行NER模型测试（实时输出）...\n")
    success, _ = run_command(
        [sys.executable, diagnose_script],
        cwd=script_dir,
        description="测试NER模型（实时输出）",
        capture_output=False  # 实时显示
    )
    
    if not success:
        logger.error("❌ NER模型测试执行失败（命令执行错误）")
        logger.error("   流水线终止")
        return False
    
    # 再次运行以捕获输出进行详细检查
    logger.info("\n检查测试结果...\n")
    success, output = run_command(
        [sys.executable, diagnose_script],
        cwd=script_dir,
        description="检查NER模型测试结果",
        capture_output=True  # 捕获输出用于检查
    )
    
    if not success:
        logger.error("❌ NER模型测试失败（命令执行错误）")
        logger.error("   流水线终止")
        return False
    
    # 更准确的评测逻辑：检查多个指标
    evaluation_passed = False
    evaluation_reasons = []
    critical_errors = []
    
    # 检查1: 模型是否成功加载（关键检查）
    if "❌ 微调模型不存在" in output or "模型加载失败" in output:
        critical_errors.append("模型加载失败 - 这是关键错误")
        evaluation_passed = False
    elif "✅ 模型加载成功" in output or "成功加载标签映射" in output:
        evaluation_reasons.append("✅ 模型加载成功")
    elif "加载模型:" in output:
        evaluation_reasons.append("模型加载尝试（需进一步确认）")
    
    # 检查2: 是否提取到实体（关键检查）
    found_entities = False
    if "提取的实体:" in output:
        # 查找所有"提取的实体:"后面的内容
        parts = output.split("提取的实体:")
        if len(parts) > 1:
            # 检查每个提取结果
            for part in parts[1:]:
                # 提取接下来的几行
                next_lines = part.split('\n')[:5]  # 取前5行
                for line in next_lines:
                    line = line.strip()
                    # 检查是否包含实体（不是空列表）
                    if line and line != "[]" and not line.startswith("提取的实体:") and len(line) > 2:
                        # 检查是否包含中文字符或常见实体格式（如 ['实体1', '实体2']）
                        if (any(c >= '\u4e00' and c <= '\u9fff' for c in line) or 
                            ('[' in line and ']' in line and line != "[]") or
                            (line.startswith("'") and line.endswith("'"))):
                            found_entities = True
                            break
                if found_entities:
                    break
    
    if found_entities:
        evaluation_passed = True
        evaluation_reasons.append("✅ 成功提取到实体")
    else:
        critical_errors.append("未成功提取到实体 - 这是关键错误")
    
    # 检查3: 是否有严重错误
    if "❌" in output and ("模型" in output or "失败" in output):
        # 检查是否是关键错误
        error_lines = [line for line in output.split('\n') if '❌' in line]
        for error_line in error_lines:
            if any(keyword in error_line for keyword in ['模型', '加载', '不存在', '失败']):
                critical_errors.append(f"检测到关键错误: {error_line.strip()}")
    
    # 输出评测结果
    logger.info("\n" + "=" * 70)
    logger.info("NER模型测试结果评估")
    logger.info("=" * 70)
    
    if critical_errors:
        logger.error("❌ 检测到关键错误:")
        for error in critical_errors:
            logger.error(f"   - {error}")
        logger.error("\n❌ NER模型测试不通过，流水线终止")
        logger.error("   请检查模型训练是否成功，或手动运行 diagnose_ner_model.py 查看详细错误")
        return False
    
    if evaluation_passed:
        logger.info("✅ NER模型测试通过")
        for reason in evaluation_reasons:
            logger.info(f"   - {reason}")
        return True
    else:
        logger.warning("⚠️  NER模型测试结果不确定")
        for reason in evaluation_reasons:
            logger.warning(f"   - {reason}")
        if not found_entities:
            logger.error("❌ 未成功提取到实体，这可能是严重问题")
            logger.error("   流水线终止，请检查模型训练是否成功")
            return False
        else:
            logger.warning("   但继续执行流水线...")
            return True


def extract_entity_words():
    """提取实体词并向量化（使用NER模型）"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤 3a: 提取实体词并向量化")
    logger.info("=" * 70)
    
    # 检查find_top_k.py是否存在
    find_top_k_script = os.path.join(script_dir, "data", "find_top_k.py")
    if not os.path.exists(find_top_k_script):
        logger.error(f"❌ 实体词提取脚本不存在: {find_top_k_script}")
        logger.info("请确保 data/find_top_k.py 文件存在")
        return False
    
    logger.info("运行实体词提取脚本（使用NER模型提取实体词并向量化）...")
    logger.info("这将从每条数据的中文描述中提取实体词，并使用微调后的模型进行向量化")
    
    success, output = run_command(
        [sys.executable, find_top_k_script],
        cwd=script_dir,
        description="提取实体词并向量化"
    )
    
    if not success:
        logger.warning("⚠️  实体词提取失败，但继续执行...")
        return False
    
    # 检查输出文件是否生成（现在保存在 data/ 目录）
    data_dir = os.path.join(script_dir, "data")
    output_files = [
        os.path.join(data_dir, "entity_words_zh.jsonl"),  # 新位置：data/ 目录
        os.path.join(data_dir, "entity_words_en.jsonl"),  # 新位置：data/ 目录
        os.path.join(script_dir, "entity_words_zh.jsonl"),  # 旧位置：兼容检查
        os.path.join(script_dir, "entity_words_en.jsonl")   # 旧位置：兼容检查
    ]
    
    found_files = [f for f in output_files if os.path.exists(f)]
    if found_files:
        logger.info(f"✅ 实体词提取完成，生成文件: {len(set(found_files))} 个")
        for f in set(found_files):
            logger.info(f"   - {f}")
        return True
    else:
        logger.warning("⚠️  未找到输出文件，实体词提取可能未完成")
        logger.warning(f"   检查的路径:")
        for f in output_files[:2]:  # 只显示新位置
            logger.warning(f"     - {f}")
        return False


def vectorize_and_store_to_es():
    """向量化数据集并存入ES"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤 3b: 向量化数据集并存入ES")
    logger.info("=" * 70)
    
    # 检查vector2ES.py是否存在
    vector_script = os.path.join(script_dir, "vector", "vector2ES.py")
    if not os.path.exists(vector_script):
        logger.error(f"❌ 向量化脚本不存在: {vector_script}")
        logger.info("请确保vector2ES.py文件存在")
        return False
    
    # 检查输入文件是否存在（entity_words_zh.jsonl 和 entity_words_en.jsonl）
    data_dir = os.path.join(script_dir, "data")
    input_files = [
        os.path.join(data_dir, "entity_words_zh.jsonl"),
        os.path.join(data_dir, "entity_words_en.jsonl"),
        os.path.join(script_dir, "entity_words_zh.jsonl"),  # 兼容旧位置
        os.path.join(script_dir, "entity_words_en.jsonl")   # 兼容旧位置
    ]
    
    found_input_files = [f for f in input_files if os.path.exists(f)]
    if not found_input_files:
        logger.warning("⚠️  未找到实体词文件，请先运行实体词提取步骤")
        logger.warning("   期望的文件:")
        logger.warning(f"     - {os.path.join(data_dir, 'entity_words_zh.jsonl')}")
        logger.warning(f"     - {os.path.join(data_dir, 'entity_words_en.jsonl')}")
        return False
    
    logger.info(f"✅ 找到 {len(set(found_input_files))} 个实体词文件:")
    for f in set(found_input_files):
        logger.info(f"   - {f}")
    
    logger.info("\n开始运行向量化脚本...")
    logger.info("注意: vector2ES.py 会自动查找并处理 entity_words_zh.jsonl 和 entity_words_en.jsonl 文件")
    
    # 自动运行向量化脚本
    success, output = run_command(
        [sys.executable, vector_script],
        cwd=script_dir,  # 在 work_wyy 目录运行，vector2ES.py 会自动查找文件
        description="向量化数据集并存入ES",
        capture_output=False  # 实时显示进度
    )
    
    if not success:
        logger.error("❌ 向量化失败，请检查错误信息")
        return False
    
    logger.info("✅ 向量化并存入ES完成")
    return True


def run_final_test():
    """运行正式测试"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤 4: 运行正式测试（向量检索系统）")
    logger.info("=" * 70)
    
    test_script = os.path.join(script_dir, "search_vllm.py")
    
    if not os.path.exists(test_script):
        logger.error(f"❌ 测试脚本不存在: {test_script}")
        return False
    
    logger.info("运行向量检索系统测试（评估所有5种检索方案）...")
    logger.info("这将评估以下方案：")
    logger.info("  1. vector_only - 纯向量检索")
    logger.info("  2. es_text_only - 纯ES文本搜索")
    logger.info("  3. llm_only - 纯LLM判断")
    logger.info("  4. vector_with_llm_always - 向量+LLM（始终重排序）")
    logger.info("  5. vector_with_llm - 向量+LLM（智能混合模式，推荐）")
    logger.info("\n详细说明请参考：检索方案对比说明.md\n")
    
    success, output = run_command(
        [sys.executable, test_script],
        cwd=script_dir,
        description="运行向量检索系统测试（评估所有5种检索方案）"
    )
    
    if success:
        logger.info("✅ 正式测试完成")
        return True
    else:
        logger.warning("⚠️  测试过程中有错误，请检查输出")
        return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='自动化训练与测试流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                           # 完整流程（默认）
  %(prog)s --from test               # 从测试阶段开始（跳过训练）
  %(prog)s --from extract            # 从实体词提取开始（跳过训练和测试）
  %(prog)s --from vectorize           # 从向量化开始（跳过训练、测试、实体词提取）
  %(prog)s --from final_test         # 只运行最终测试
  %(prog)s --skip check              # 跳过前置检查
  %(prog)s --skip train              # 跳过训练
  %(prog)s --skip test               # 跳过测试
  %(prog)s --skip extract            # 跳过实体词提取
  %(prog)s --skip vectorize          # 跳过向量化
  %(prog)s --from extract --skip vectorize  # 从提取开始，但跳过向量化
  %(prog)s --list-steps              # 列出所有可用阶段
        """
    )
    
    parser.add_argument(
        '--from',
        dest='start_from',
        choices=['check', 'train', 'test', 'extract', 'vectorize', 'final_test'],
        help='从指定阶段开始运行（跳过之前的阶段）'
    )
    
    parser.add_argument(
        '--skip',
        dest='skip_steps',
        action='append',
        choices=['check', 'train', 'test', 'extract', 'vectorize', 'final_test'],
        help='跳过指定阶段（可以多次使用，如 --skip train --skip test）'
    )
    
    parser.add_argument(
        '--list-steps',
        action='store_true',
        help='列出所有可用的阶段并退出'
    )
    
    return parser.parse_args()


def list_steps():
    """列出所有可用的阶段"""
    steps_info = [
        ("check", "检查前置条件", "检查环境、数据、模型等前置条件"),
        ("train", "训练NER模型", "使用训练数据微调NER模型"),
        ("test", "测试NER模型", "测试NER模型效果，确保能正确提取实体"),
        ("extract", "提取实体词并向量化", "使用NER模型提取实体词并进行向量化"),
        ("vectorize", "向量化并存入ES", "将向量化后的数据存入Elasticsearch"),
        ("final_test", "运行正式测试", "运行向量检索系统测试（评估所有5种检索方案）"),
    ]
    
    print("\n" + "=" * 70)
    print("可用阶段列表")
    print("=" * 70)
    for step_id, step_name, step_desc in steps_info:
        print(f"  {step_id:12s} - {step_name:20s} - {step_desc}")
    print("=" * 70 + "\n")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果只是列出阶段，则退出
    if args.list_steps:
        list_steps()
        return
    
    logger.info("\n" + "=" * 70)
    logger.info("自动化训练与测试流水线")
    logger.info("=" * 70)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示运行模式
    if args.start_from:
        logger.info(f"运行模式: 从 '{args.start_from}' 阶段开始")
    if args.skip_steps:
        logger.info(f"跳过阶段: {', '.join(args.skip_steps)}")
    if not args.start_from and not args.skip_steps:
        logger.info("运行模式: 完整流程（所有阶段）")
    
    # 切换到工作目录（work_wyy）
    os.chdir(script_dir)
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info(f"日志保存目录: {trainlog_dir}")
    logger.info("=" * 70)
    
    # 定义所有步骤（使用ID便于匹配）
    all_steps = [
        ("check", "检查前置条件", check_prerequisites, True),  # 警告不影响继续
        ("train", "训练NER模型", train_ner_model, False),  # 失败则终止
        ("test", "测试NER模型", test_ner_model, False),  # 失败则终止（关键步骤）
        ("extract", "提取实体词并向量化", extract_entity_words, True),  # 可以跳过
        ("vectorize", "向量化并存入ES", vectorize_and_store_to_es, True),  # 可以跳过
        ("final_test", "运行正式测试", run_final_test, True),  # 可以跳过
    ]
    
    # 根据参数决定从哪个阶段开始
    start_index = 0
    if args.start_from:
        for i, (step_id, _, _, _) in enumerate(all_steps):
            if step_id == args.start_from:
                start_index = i
                break
        else:
            logger.error(f"❌ 无效的起始阶段: {args.start_from}")
            logger.info("使用 --list-steps 查看所有可用阶段")
            return
    
    # 根据参数决定跳过哪些阶段
    skip_steps = set(args.skip_steps or [])
    
    # 过滤步骤
    steps_to_run = []
    for i, (step_id, step_name, step_func, allow_skip) in enumerate(all_steps):
        if i < start_index:
            logger.info(f"⏭️  跳过阶段: {step_name} (在起始阶段之前)")
            continue
        if step_id in skip_steps:
            logger.info(f"⏭️  跳过阶段: {step_name} (用户指定跳过)")
            continue
        steps_to_run.append((step_name, step_func, allow_skip))
    
    if not steps_to_run:
        logger.warning("⚠️  没有要执行的步骤，请检查参数")
        return
    
    logger.info(f"\n将执行 {len(steps_to_run)} 个步骤:")
    for step_name, _, _ in steps_to_run:
        logger.info(f"  - {step_name}")
    logger.info("")
    
    # 执行步骤
    results = {}
    
    for step_name, step_func, allow_skip in steps_to_run:
        try:
            result = step_func()
            results[step_name] = result
            
            if not result and not allow_skip:
                logger.error(f"\n❌ 关键步骤 '{step_name}' 失败，终止流水线")
                break
            
            if not result and allow_skip:
                logger.warning(f"\n⚠️  步骤 '{step_name}' 失败或跳过，继续执行")
        
        except KeyboardInterrupt:
            logger.info("\n\n用户中断执行")
            break
        except Exception as e:
            logger.error(f"\n❌ 步骤 '{step_name}' 执行异常: {e}")
            if not allow_skip:
                break
    
    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("流水线执行总结")
    logger.info("=" * 70)
    for step_name, result in results.items():
        status = "✅ 成功" if result else "❌ 失败/跳过"
        logger.info(f"{step_name}: {status}")
    
    logger.info(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # 恢复stdout和stderr，并关闭文件
    if 'tee' in globals():
        original_stdout = tee.stdout
        original_stderr = tee.stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee.close()
        print(f"\n✅ 控制台输出已保存到: {console_log_file}")
        print(f"✅ 日志信息已保存到: {log_filename}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n用户中断执行")
    except Exception as e:
        logger.error(f"\n执行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保文件被关闭
        if 'tee' in globals():
            original_stdout = tee.stdout
            original_stderr = tee.stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            tee.close()

