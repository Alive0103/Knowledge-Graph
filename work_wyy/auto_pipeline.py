#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化训练与测试流水线
1. 训练NER模型
2. 测试模型效果
3. 使用测试通过的模型对数据集进行向量化处理并存入ES
4. 正式测试向量检索系统
"""

import os
import sys
import subprocess
import logging
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


def run_command(cmd, cwd=None, description=""):
    """
    运行命令并记录输出
    
    Args:
        cmd: 命令列表
        cwd: 工作目录
        description: 命令描述
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"执行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")
    logger.info(f"{'=' * 70}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        logger.info(f"✅ {description} 成功完成")
        if result.stdout:
            logger.info(f"输出:\n{result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} 失败")
        logger.error(f"错误代码: {e.returncode}")
        if e.stdout:
            logger.error(f"标准输出:\n{e.stdout}")
        if e.stderr:
            logger.error(f"错误输出:\n{e.stderr}")
        return False, e.stderr


def check_prerequisites():
    """检查前置条件"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤 0: 检查前置条件")
    logger.info("=" * 70)
    
    check_script = os.path.join(script_dir, "ner", "check_prerequisites.py")
    
    # 检查NER前置条件
    success, _ = run_command(
        [sys.executable, check_script],
        cwd=script_dir,
        description="检查NER训练前置条件"
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
        description="训练NER模型"
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
    
    diagnose_script = os.path.join(script_dir, "ner", "diagnose_ner_model.py")
    
    success, output = run_command(
        [sys.executable, diagnose_script],
        cwd=script_dir,
        description="测试NER模型"
    )
    
    if not success:
        logger.warning("⚠️  NER模型测试失败，但继续执行...")
        return False
    
    # 检查输出中是否包含成功的实体提取
    if "提取的实体:" in output and "[]" not in output.split("提取的实体:")[-1][:100]:
        logger.info("✅ NER模型测试通过，能够正确提取实体")
        return True
    else:
        logger.warning("⚠️  NER模型可能未正确提取实体，但继续执行...")
        return True  # 仍然继续，让用户决定


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
    
    # 检查输出文件是否生成
    output_files = [
        os.path.join(script_dir, "entity_words_zh.jsonl"),
        os.path.join(script_dir, "entity_words_en.jsonl")
    ]
    
    found_files = [f for f in output_files if os.path.exists(f)]
    if found_files:
        logger.info(f"✅ 实体词提取完成，生成文件: {len(found_files)} 个")
        for f in found_files:
            logger.info(f"   - {f}")
        return True
    else:
        logger.warning("⚠️  未找到输出文件，实体词提取可能未完成")
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
    
    logger.info("⚠️  向量化脚本需要手动配置数据源路径")
    logger.info("请运行以下命令进行向量化:")
    logger.info(f"  cd {os.path.join(script_dir, 'vector')}")
    logger.info(f"  python vector2ES.py")
    logger.info("\n或者直接运行:")
    logger.info(f"  python {vector_script}")
    logger.info("\n注意: vector2ES.py 会自动查找 entity_words_zh.jsonl 和 entity_words_en.jsonl 文件")
    
    # 询问用户是否已手动完成
    response = input("\n是否已完成向量化并存入ES? (y/n): ").strip().lower()
    if response == 'y':
        logger.info("✅ 向量化完成")
        return True
    else:
        logger.warning("⚠️  跳过向量化步骤，请稍后手动完成")
        return False


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


def main():
    """主函数"""
    logger.info("\n" + "=" * 70)
    logger.info("自动化训练与测试流水线")
    logger.info("=" * 70)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 切换到工作目录（work_wyy）
    os.chdir(script_dir)
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info(f"日志保存目录: {trainlog_dir}")
    
    steps = [
        ("检查前置条件", check_prerequisites, True),  # 警告不影响继续
        ("训练NER模型", train_ner_model, False),  # 失败则终止
        ("测试NER模型", test_ner_model, True),  # 警告不影响继续
        ("提取实体词并向量化", extract_entity_words, True),  # 可以跳过
        ("向量化并存入ES", vectorize_and_store_to_es, True),  # 可以跳过
        ("运行正式测试", run_final_test, True),  # 可以跳过
    ]
    
    results = {}
    
    for step_name, step_func, allow_skip in steps:
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

