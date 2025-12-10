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
    python auto_pipeline.py --compare-datasets  # 对比多种数据集配置（推荐）
    python auto_pipeline.py --from test         # 从测试阶段开始
    python auto_pipeline.py --from extract      # 从实体词提取开始
    python auto_pipeline.py --from vectorize    # 从向量化开始
    python auto_pipeline.py --from final_test   # 只运行最终测试
    python auto_pipeline.py --skip check        # 跳过前置检查
    python auto_pipeline.py --skip train        # 跳过训练
    python auto_pipeline.py --skip test         # 跳过测试

数据集对比模式（--compare-datasets）:
    对6种不同的数据集配置分别运行完整流水线，并生成对比表格：
    1. 仅traindata
    2. 仅CCKS数据
    3. 仅train.txt
    4. traindata + CCKS
    5. 全部（除MSRA）
    6. 全部数据（包括MSRA）
    
    每种配置会：
    - 使用独立模型目录（model/ner_finetuned_<config_id>）
    - 使用独立ES索引（data_<config_id>）
    - 运行完整流水线（训练→测试→提取→向量化→最终测试）
    - 提取并记录评估指标
    - 最后生成汇总对比表格
"""

import os
import sys
import subprocess
import logging
import argparse
import json
import re
import shutil
from datetime import datetime

# 导入分步骤日志管理系统
from step_logger import StepLogger

# 获取脚本所在目录（local）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 工作目录（work_wyy）是 local 的父目录
work_dir = os.path.dirname(script_dir)

# 创建 trainlog 文件夹（如果不存在）
# 日志保存在 work_wyy/trainlog，而不是 local/trainlog
trainlog_dir = os.path.join(work_dir, 'trainlog')
os.makedirs(trainlog_dir, exist_ok=True)

# 创建日志文件名（保存到 trainlog 文件夹）- 保留用于向后兼容
log_filename = os.path.join(trainlog_dir, f'auto_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# 配置日志（保留用于向后兼容，但主要使用step_logger）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建分步骤日志管理器（全局，在main函数中初始化）
step_logger = None


def run_command(cmd, cwd=None, description="", capture_output=False):
    """
    运行命令并记录输出
    
    Args:
        cmd: 命令列表
        cwd: 工作目录
        description: 命令描述
        capture_output: 是否捕获输出（False时实时显示并保存，True时缓冲后显示）
    
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
            # 使用Popen实时捕获子进程输出并写入Tee对象
            # 这样所有print输出（包括HuggingFace Trainer的训练进度）都会被捕获
            # 设置PYTHONUNBUFFERED=1确保Python输出不被缓冲，立即刷新
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # 使用Popen实时捕获输出
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并stderr到stdout
                text=True,
                encoding='utf-8',
                env=env,
                bufsize=1  # 行缓冲
            )
            
            # 实时读取并写入Tee对象
            try:
                for line in process.stdout:
                    # 写入Tee对象（会同时写入文件和控制台）
                    sys.stdout.write(line)
                    sys.stdout.flush()
                
                # 等待进程完成
                return_code = process.wait()
                
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)
                
            except Exception as e:
                process.kill()
                process.wait()
                raise
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
    
    check_script = os.path.join(script_dir, "check_prerequisites.py")
    
    # 检查NER前置条件（实时显示输出）
    # 需要在 work_wyy 目录运行，以便正确找到数据文件
    success, _ = run_command(
        [sys.executable, check_script],
        cwd=work_dir,  # 在 work_wyy 目录运行
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
    
    ner_script = os.path.join(script_dir, "finetune_ner_model.py")
    
    success, output = run_command(
        [sys.executable, ner_script],
        cwd=work_dir,  # 在 work_wyy 目录运行
        description="训练NER模型",
        capture_output=False  # 实时显示进度条和训练过程
    )
    
    if not success:
        logger.error("❌ NER模型训练失败，终止流水线")
        return False
    
    # 检查模型是否生成（相对于 work_wyy 目录）
    # 模型保存在 work_wyy/model/ner_finetuned
    model_path = os.path.join(work_dir, "model", "ner_finetuned")
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
    
    diagnose_script = os.path.join(script_dir, "diagnose_ner_model.py")
    
    # 先实时显示输出（让用户看到评测过程）
    logger.info("\n开始运行NER模型测试（实时输出）...\n")
    success, _ = run_command(
        [sys.executable, diagnose_script],
        cwd=work_dir,  # 在 work_wyy 目录运行
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
        cwd=work_dir,  # 在 work_wyy 目录运行
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


def extract_entity_words(config_id=None):
    """提取实体词并向量化（使用NER模型）"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤 3a: 提取实体词并向量化")
    logger.info("=" * 70)
    
    # 如果指定了config_id，需要临时修改模型路径
    if config_id:
        config_path = os.path.join(script_dir, "config.py")
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        old_pattern = r"FINETUNED_MODEL_PATH = os\.path\.join\(MODEL_DIR, '[^']+'\)"
        new_path = f"FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_finetuned_{config_id}')"
        new_content = re.sub(old_pattern, new_path, content)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        if 'config' in sys.modules:
            import importlib
            import config
            importlib.reload(config)
    
    # 检查find_top_k.py是否存在
    find_top_k_script = os.path.join(script_dir, "find_top_k.py")
    if not os.path.exists(find_top_k_script):
        logger.error(f"❌ 实体词提取脚本不存在: {find_top_k_script}")
        logger.info("请确保 find_top_k.py 文件存在")
        return False
    
    logger.info("运行实体词提取脚本（使用NER模型提取实体词并向量化）...")
    logger.info("这将从每条数据的中文描述中提取实体词，并使用微调后的模型进行向量化")
    
    success, output = run_command(
        [sys.executable, find_top_k_script],
        cwd=work_dir,  # 在 work_wyy 目录运行
        description="提取实体词并向量化"
    )
    
    if not success:
        logger.warning("⚠️  实体词提取失败，但继续执行...")
        return False
    
    # 检查输出文件是否生成（现在保存在 data/ 目录）
    data_dir = os.path.join(work_dir, "data")
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


def vectorize_and_store_to_es(config_id=None):
    """向量化数据集并存入ES"""
    logger.info("\n" + "=" * 70)
    logger.info("步骤 3b: 向量化数据集并存入ES")
    logger.info("=" * 70)
    
    # 如果指定了config_id，需要临时修改ES索引名称和模型路径
    if config_id:
        config_path = os.path.join(script_dir, "config.py")
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修改ES索引名称
        old_index_pattern = r"ES_INDEX_NAME = '[^']+'"
        new_index = f"ES_INDEX_NAME = 'data_{config_id}'"
        content = re.sub(old_index_pattern, new_index, content)
        
        # 修改模型路径
        old_pattern = r"FINETUNED_MODEL_PATH = os\.path\.join\(MODEL_DIR, '[^']+'\)"
        new_path = f"FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_finetuned_{config_id}')"
        content = re.sub(old_pattern, new_path, content)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if 'config' in sys.modules:
            import importlib
            import config
            importlib.reload(config)
        
        logger.info(f"✅ 已切换到ES索引: data_{config_id}")
    
    # 检查vector2ES.py是否存在
    vector_script = os.path.join(script_dir, "vector2ES.py")
    if not os.path.exists(vector_script):
        logger.error(f"❌ 向量化脚本不存在: {vector_script}")
        logger.info("请确保vector2ES.py文件存在")
        return False
    
    # 检查输入文件是否存在（entity_words_zh.jsonl 和 entity_words_en.jsonl）
    data_dir = os.path.join(work_dir, "data")
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
        cwd=work_dir,  # 在 work_wyy 目录运行，vector2ES.py 会自动查找文件
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
        cwd=work_dir,  # 在 work_wyy 目录运行
        description="运行向量检索系统测试（评估所有5种检索方案）"
    )
    
    if success:
        logger.info("✅ 正式测试完成")
        return True
    else:
        logger.warning("⚠️  测试过程中有错误，请检查输出")
        return False


# ============================================================================
# 数据集配置方案定义
# ============================================================================

DATASET_CONFIGS = {
    "config_1_traindata_only": {
        "name": "仅traindata",
        "description": "只使用traindata目录的数据",
        "switches": {
            'traindata': True,
            'ccks_json': False,
            'ccks_validate': False,
            'ccks_fold0': False,
            'ccks_fold1': False,
            'ccks_fold2': False,
            'ccks_fold3': False,
            'ccks_fold4': False,
            'train_txt': False,
            'msra_train': False,
            'msra_test': False,
        }
    },
    "config_2_ccks_only": {
        "name": "仅CCKS数据",
        "description": "只使用CCKS军事领域数据（JSON和BIO格式）",
        "switches": {
            'traindata': False,
            'ccks_json': True,
            'ccks_validate': True,
            'ccks_fold0': True,
            'ccks_fold1': True,
            'ccks_fold2': True,
            'ccks_fold3': True,
            'ccks_fold4': True,
            'train_txt': False,
            'msra_train': False,
            'msra_test': False,
        }
    },
    "config_3_train_txt_only": {
        "name": "仅train.txt",
        "description": "只使用train.txt的JSONL格式数据",
        "switches": {
            'traindata': False,
            'ccks_json': False,
            'ccks_validate': False,
            'ccks_fold0': False,
            'ccks_fold1': False,
            'ccks_fold2': False,
            'ccks_fold3': False,
            'ccks_fold4': False,
            'train_txt': True,
            'msra_train': False,
            'msra_test': False,
        }
    },
    "config_4_traindata_ccks": {
        "name": "traindata + CCKS",
        "description": "traindata目录 + CCKS数据",
        "switches": {
            'traindata': True,
            'ccks_json': True,
            'ccks_validate': True,
            'ccks_fold0': True,
            'ccks_fold1': True,
            'ccks_fold2': True,
            'ccks_fold3': True,
            'ccks_fold4': True,
            'train_txt': False,
            'msra_train': False,
            'msra_test': False,
        }
    },
    "config_5_all_except_msra": {
        "name": "全部（除MSRA）",
        "description": "traindata + CCKS + train.txt（不含MSRA通用数据）",
        "switches": {
            'traindata': True,
            'ccks_json': True,
            'ccks_validate': True,
            'ccks_fold0': True,
            'ccks_fold1': True,
            'ccks_fold2': True,
            'ccks_fold3': True,
            'ccks_fold4': True,
            'train_txt': True,
            'msra_train': False,
            'msra_test': False,
        }
    },
    "config_6_all_data": {
        "name": "全部数据",
        "description": "所有数据源（包括MSRA通用NER数据）",
        "switches": {
            'traindata': True,
            'ccks_json': True,
            'ccks_validate': True,
            'ccks_fold0': True,
            'ccks_fold1': True,
            'ccks_fold2': True,
            'ccks_fold3': True,
            'ccks_fold4': True,
            'train_txt': True,
            'msra_train': True,
            'msra_test': True,
        }
    },
}


def update_config_data_sources(config_switches):
    """临时更新config.py中的数据源开关"""
    config_path = os.path.join(script_dir, "config.py")
    
    # 读取原始config.py
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原始文件
    backup_path = config_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(config_path, backup_path)
    
    # 替换DATA_SOURCE_SWITCHES部分
    pattern = r"DATA_SOURCE_SWITCHES = \{.*?\}"
    
    switches_str = "DATA_SOURCE_SWITCHES = {\n"
    for key, value in config_switches.items():
        switches_str += f"    '{key}': {value},\n"
    switches_str += "}"
    
    new_content = re.sub(pattern, switches_str, content, flags=re.DOTALL)
    
    # 写入新内容
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # 重新导入config模块（清除缓存）
    if 'config' in sys.modules:
        import importlib
        import config
        importlib.reload(config)


def restore_config_data_sources():
    """恢复config.py的原始数据源开关"""
    config_path = os.path.join(script_dir, "config.py")
    backup_path = config_path + ".backup"
    
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, config_path)
        # 重新导入config模块
        if 'config' in sys.modules:
            import importlib
            import config
            importlib.reload(config)


def extract_ner_evaluation_metrics(output):
    """从diagnose_ner_model.py的输出中提取评估指标"""
    metrics = {
        'train_samples': None,
        'dev_samples': None,
        'entity_types': None,
        'total_entities': None,
        'avg_entities_per_sample': None,
        'test_samples': None,
        'successful_extractions': None,
    }
    
    # 提取训练样本数
    match = re.search(r'训练样本数[：:]\s*([\d,]+)', output)
    if match:
        metrics['train_samples'] = int(match.group(1).replace(',', ''))
    
    # 提取验证样本数
    match = re.search(r'验证样本数[：:]\s*([\d,]+)', output)
    if match:
        metrics['dev_samples'] = int(match.group(1).replace(',', ''))
    
    # 提取实体类型数
    match = re.search(r'实体类型总数[：:]\s*(\d+)', output)
    if match:
        metrics['entity_types'] = int(match.group(1))
    
    # 提取总实体数
    match = re.search(r'总实体数[：:]\s*([\d,]+)', output)
    if match:
        metrics['total_entities'] = int(match.group(1).replace(',', ''))
    
    # 提取平均实体数
    match = re.search(r'平均每个样本的实体数[：:]\s*([\d.]+)', output)
    if match:
        metrics['avg_entities_per_sample'] = float(match.group(1))
    
    # 提取测试样本数和成功提取数
    match = re.search(r'测试样本数[：:]\s*(\d+)', output)
    if match:
        metrics['test_samples'] = int(match.group(1))
    
    match = re.search(r'成功提取实体[：:]\s*(\d+)/(\d+)', output)
    if match:
        metrics['successful_extractions'] = int(match.group(1))
        if metrics['test_samples'] is None:
            metrics['test_samples'] = int(match.group(2))
    
    return metrics


def extract_search_evaluation_metrics(output):
    """从search_vllm.py的输出中提取评估指标"""
    metrics = {}
    
    # 提取所有检索模式的结果
    modes = ['vector_only', 'es_text_only', 'llm_only', 'vector_with_llm_always', 'vector_with_llm']
    
    for mode in modes:
        # 查找该模式的MRR, Hit@1, Hit@5, Hit@10
        pattern = rf"{mode}.*?MRR[：:]\s*([\d.]+).*?Hit@1[：:]\s*([\d.]+).*?Hit@5[：:]\s*([\d.]+).*?Hit@10[：:]\s*([\d.]+)"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            metrics[mode] = {
                'mrr': float(match.group(1)),
                'hit@1': float(match.group(2)),
                'hit@5': float(match.group(3)),
                'hit@10': float(match.group(4)),
            }
        else:
            # 尝试从汇总表格中提取
            pattern = rf"方案\d+[：:].*?{mode}.*?([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            match = re.search(pattern, output)
            if match:
                metrics[mode] = {
                    'mrr': float(match.group(1)),
                    'hit@1': float(match.group(2)),
                    'hit@5': float(match.group(3)),
                    'hit@10': float(match.group(4)),
                }
    
    return metrics


def train_ner_model_with_config(config_id, config_info):
    """使用指定配置训练NER模型（创建独立模型目录）"""
    # 为每种配置创建独立的模型目录
    model_dir = os.path.join(work_dir, "model", f"ner_finetuned_{config_id}")
    
    # 临时修改config.py中的FINETUNED_MODEL_PATH
    config_path = os.path.join(script_dir, "config.py")
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换FINETUNED_MODEL_PATH
    old_pattern = r"FINETUNED_MODEL_PATH = os\.path\.join\(MODEL_DIR, '[^']+'\)"
    new_path = f"FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_finetuned_{config_id}')"
    new_content = re.sub(old_pattern, new_path, content)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # 重新导入config
    if 'config' in sys.modules:
        import importlib
        import config
        importlib.reload(config)
    
    # 调用训练脚本
    ner_script = os.path.join(script_dir, "finetune_ner_model.py")
    
    success, output = run_command(
        [sys.executable, ner_script],
        cwd=work_dir,
        description=f"训练NER模型（配置: {config_id}）",
        capture_output=False
    )
    
    if not success:
        logger.error(f"❌ 配置 {config_id} 的NER模型训练失败")
        return False, model_dir
    
    # 检查模型是否生成
    if not os.path.exists(model_dir):
        logger.error(f"❌ 模型文件不存在: {model_dir}")
        return False, model_dir
    
    logger.info(f"✅ 模型已保存到: {model_dir}")
    return True, model_dir


def test_ner_model_with_config(config_id):
    """使用指定配置测试NER模型"""
    # 确保使用正确的模型路径
    model_dir = os.path.join(work_dir, "model", f"ner_finetuned_{config_id}")
    
    # 临时修改config.py中的FINETUNED_MODEL_PATH
    config_path = os.path.join(script_dir, "config.py")
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_pattern = r"FINETUNED_MODEL_PATH = os\.path\.join\(MODEL_DIR, '[^']+'\)"
    new_path = f"FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_finetuned_{config_id}')"
    new_content = re.sub(old_pattern, new_path, content)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # 重新导入config
    if 'config' in sys.modules:
        import importlib
        import config
        importlib.reload(config)
    
    # 调用原始测试函数并捕获输出
    diagnose_script = os.path.join(script_dir, "diagnose_ner_model.py")
    success, output = run_command(
        [sys.executable, diagnose_script],
        cwd=work_dir,
        description=f"测试NER模型（配置: {config_id}）",
        capture_output=True
    )
    
    if success:
        metrics = extract_ner_evaluation_metrics(output)
        return True, metrics, output
    else:
        return False, {}, output


def run_final_test_with_config(config_id):
    """使用指定配置运行最终测试"""
    # 确保使用正确的模型路径和ES索引
    model_dir = os.path.join(work_dir, "model", f"ner_finetuned_{config_id}")
    
    # 临时修改config.py中的FINETUNED_MODEL_PATH和ES_INDEX_NAME
    config_path = os.path.join(script_dir, "config.py")
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修改模型路径
    old_pattern = r"FINETUNED_MODEL_PATH = os\.path\.join\(MODEL_DIR, '[^']+'\)"
    new_path = f"FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_finetuned_{config_id}')"
    content = re.sub(old_pattern, new_path, content)
    
    # 修改ES索引名称
    old_index_pattern = r"ES_INDEX_NAME = '[^']+'"
    new_index = f"ES_INDEX_NAME = 'data_{config_id}'"
    content = re.sub(old_index_pattern, new_index, content)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 重新导入config
    if 'config' in sys.modules:
        import importlib
        import config
        importlib.reload(config)
    
    logger.info(f"✅ 已切换到ES索引: data_{config_id}")
    
    # 调用原始测试函数并捕获输出
    test_script = os.path.join(script_dir, "search_vllm.py")
    success, output = run_command(
        [sys.executable, test_script],
        cwd=work_dir,
        description=f"运行最终测试（配置: {config_id}）",
        capture_output=True
    )
    
    if success:
        metrics = extract_search_evaluation_metrics(output)
        return True, metrics, output
    else:
        return False, {}, output


def generate_comparison_table(all_results):
    """生成汇总对比表格"""
    logger.info("\n" + "=" * 100)
    logger.info("数据集配置对比汇总表")
    logger.info("=" * 100)
    
    # 准备表格数据
    table_data = []
    for config_id, result in all_results.items():
        config_info = DATASET_CONFIGS[config_id]
        row = {
            'config_id': config_id,
            'config_name': config_info['name'],
            'ner_metrics': result.get('ner_metrics', {}),
            'search_metrics': result.get('search_metrics', {}),
            'success': result.get('success', False),
        }
        table_data.append(row)
    
    # 打印NER评估结果表格
    logger.info("\n【NER模型评估结果】")
    logger.info("-" * 100)
    logger.info(f"{'配置名称':<20} {'训练样本':<12} {'验证样本':<12} {'实体类型':<12} {'总实体数':<12} {'测试成功':<12}")
    logger.info("-" * 100)
    
    for row in table_data:
        ner = row['ner_metrics']
        train_samples = ner.get('train_samples', 'N/A')
        dev_samples = ner.get('dev_samples', 'N/A')
        entity_types = ner.get('entity_types', 'N/A')
        total_entities = ner.get('total_entities', 'N/A')
        successful = f"{ner.get('successful_extractions', 0)}/{ner.get('test_samples', 0)}"
        
        logger.info(f"{row['config_name']:<20} {str(train_samples):<12} {str(dev_samples):<12} {str(entity_types):<12} {str(total_entities):<12} {successful:<12}")
    
    # 打印检索系统评估结果表格
    logger.info("\n【检索系统评估结果（MRR）】")
    logger.info("-" * 100)
    logger.info(f"{'配置名称':<20} {'纯向量':<12} {'ES文本':<12} {'纯LLM':<12} {'向量+LLM(始终)':<18} {'向量+LLM(智能)':<18}")
    logger.info("-" * 100)
    
    for row in table_data:
        search = row['search_metrics']
        vector_only = search.get('vector_only', {}).get('mrr', 'N/A')
        es_text = search.get('es_text_only', {}).get('mrr', 'N/A')
        llm_only = search.get('llm_only', {}).get('mrr', 'N/A')
        vector_llm_always = search.get('vector_with_llm_always', {}).get('mrr', 'N/A')
        vector_llm = search.get('vector_with_llm', {}).get('mrr', 'N/A')
        
        def format_metric(m):
            if m == 'N/A':
                return 'N/A'
            return f"{m:.4f}"
        
        logger.info(f"{row['config_name']:<20} {format_metric(vector_only):<12} {format_metric(es_text):<12} {format_metric(llm_only):<12} {format_metric(vector_llm_always):<18} {format_metric(vector_llm):<18}")
    
    # 保存详细结果到JSON文件
    summary_file = os.path.join(trainlog_dir, f'dataset_comparison_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ 详细对比结果已保存到: {summary_file}")
    logger.info("=" * 100)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='自动化训练与测试流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                           # 完整流程（默认）
  %(prog)s --compare-datasets        # 对比多种数据集配置（推荐）
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
        '--compare-datasets',
        action='store_true',
        help='对比多种数据集配置（对每种配置运行完整流水线并生成对比表格）'
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


def run_dataset_comparison():
    """运行数据集配置对比"""
    global step_logger
    
    # 创建数据集对比模式的日志管理器
    step_logger = StepLogger(
        base_dir=trainlog_dir,
        session_name="dataset_comparison",
        filter_tqdm=True
    )
    
    try:
        with step_logger.step("数据集配置对比模式"):
            step_logger.info(f"将对比 {len(DATASET_CONFIGS)} 种数据集配置")
            
            # 显示所有配置
            step_logger.info("\n数据集配置列表:")
            for config_id, config_info in DATASET_CONFIGS.items():
                step_logger.info(f"  - {config_id}: {config_info['name']} - {config_info['description']}")
            
            # 切换到工作目录
            os.chdir(work_dir)
            step_logger.info(f"\n工作目录: {os.getcwd()}")
            step_logger.info(f"日志保存目录: {trainlog_dir}")
            
            # 保存原始config.py
            config_path = os.path.join(script_dir, "config.py")
            original_config_backup = config_path + ".original_backup"
            if not os.path.exists(original_config_backup):
                shutil.copy2(config_path, original_config_backup)
                step_logger.info(f"已备份原始config.py到: {original_config_backup}")
            
            all_results = {}
            
            # 对每种配置运行完整流水线
            # 获取会话开始时间戳（用于统一命名）
            session_timestamp = step_logger.session_start_time.strftime('%Y%m%d_%H%M%S')
            
            for config_id, config_info in DATASET_CONFIGS.items():
                # 为每个配置创建一个统一的日志文件（直接保存在trainlog目录下）
                safe_config_name = re.sub(r'[^\w\s-]', '', config_id).strip()
                safe_config_name = re.sub(r'[-\s]+', '_', safe_config_name)
                config_log_file = os.path.join(
                    trainlog_dir,
                    f"{safe_config_name}_{session_timestamp}.log"
                )
                
                # 记录日志文件路径
                step_logger.info(f"配置日志文件: {config_log_file}")
                
                # 使用统一的日志文件记录该配置的所有步骤
                # 所有步骤的输出都会写入同一个config_log_file
                config_result = {
                    'config_id': config_id,
                    'config_name': config_info['name'],
                    'config_description': config_info['description'],
                    'success': False,
                    'ner_metrics': {},
                    'search_metrics': {},
                    'error': None,
                }
                
                try:
                    # 1. 更新数据源开关
                    with step_logger.step("步骤1: 更新数据源配置", log_file=config_log_file):
                        update_config_data_sources(config_info['switches'])
                        step_logger.info("✅ 数据源配置已更新")
                        
                        # 显示启用的数据源
                        enabled_sources = [k for k, v in config_info['switches'].items() if v]
                        step_logger.info(f"启用的数据源: {', '.join(enabled_sources)}")
                    
                    # 2. 检查前置条件（可选，但建议执行）
                    with step_logger.step("步骤2: 检查前置条件", log_file=config_log_file):
                        check_prerequisites()
                    
                    # 3. 训练NER模型（使用独立模型目录）
                    with step_logger.step("步骤3: 训练NER模型", log_file=config_log_file):
                        train_success, model_dir = train_ner_model_with_config(config_id, config_info)
                        if not train_success:
                            config_result['error'] = "NER模型训练失败"
                            step_logger.error(f"❌ 配置 {config_id} 的NER模型训练失败，跳过后续步骤")
                            all_results[config_id] = config_result
                            continue
                    
                    # 4. 测试NER模型并提取评估指标
                    with step_logger.step("步骤4: 测试NER模型并提取评估指标", log_file=config_log_file):
                        test_success, ner_metrics, ner_output = test_ner_model_with_config(config_id)
                        if test_success:
                            config_result['ner_metrics'] = ner_metrics
                            step_logger.info(f"✅ NER评估指标已提取: {ner_metrics}")
                        else:
                            step_logger.warning(f"⚠️  NER模型测试失败，但继续执行")
                    
                    # 5. 提取实体词并向量化
                    with step_logger.step("步骤5: 提取实体词并向量化", log_file=config_log_file):
                        extract_success = extract_entity_words(config_id)
                        if not extract_success:
                            step_logger.warning(f"⚠️  实体词提取失败，但继续执行")
                    
                    # 6. 向量化并存入ES
                    with step_logger.step("步骤6: 向量化并存入ES", log_file=config_log_file):
                        vectorize_success = vectorize_and_store_to_es(config_id)
                        if not vectorize_success:
                            step_logger.warning(f"⚠️  向量化失败，但继续执行")
                    
                    # 7. 运行最终测试并提取评估指标
                    with step_logger.step("步骤7: 运行最终测试并提取评估指标", log_file=config_log_file):
                        final_test_success, search_metrics, search_output = run_final_test_with_config(config_id)
                        if final_test_success:
                            config_result['search_metrics'] = search_metrics
                            step_logger.info(f"✅ 检索系统评估指标已提取")
                            for mode, metrics in search_metrics.items():
                                step_logger.info(f"  {mode}: MRR={metrics.get('mrr', 'N/A'):.4f}, Hit@1={metrics.get('hit@1', 'N/A'):.4f}")
                        else:
                            step_logger.warning(f"⚠️  最终测试失败")
                    
                    config_result['success'] = True
                    step_logger.info(f"\n✅ 配置 {config_id} 处理完成")
                    
                except Exception as e:
                    step_logger.error(f"\n❌ 配置 {config_id} 处理异常: {e}")
                    config_result['error'] = str(e)
                    import traceback
                    traceback.print_exc()
                
                finally:
                    all_results[config_id] = config_result
            
            # 恢复原始config.py
            with step_logger.step("恢复原始config.py配置"):
                config_path = os.path.join(script_dir, "config.py")
                original_config_backup = config_path + ".original_backup"
                if os.path.exists(original_config_backup):
                    shutil.copy2(original_config_backup, config_path)
                    if 'config' in sys.modules:
                        import importlib
                        import config
                        importlib.reload(config)
                    step_logger.info("✅ 原始配置已恢复")
            
            # 生成对比表格
            with step_logger.step("生成对比汇总表格"):
                generate_comparison_table(all_results)
    
    finally:
        if step_logger:
            step_logger.close()


def main():
    """主函数"""
    global step_logger
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果只是列出阶段，则退出
    if args.list_steps:
        list_steps()
        return
    
    # 如果启用数据集对比模式
    if args.compare_datasets:
        run_dataset_comparison()
        return
    
    # 创建普通模式的日志管理器
    step_logger = StepLogger(
        base_dir=trainlog_dir,
        session_name="auto_pipeline",
        filter_tqdm=True
    )
    
    try:
        with step_logger.step("自动化训练与测试流水线"):
            # 显示运行模式
            if args.start_from:
                step_logger.info(f"运行模式: 从 '{args.start_from}' 阶段开始")
            if args.skip_steps:
                step_logger.info(f"跳过阶段: {', '.join(args.skip_steps)}")
            if not args.start_from and not args.skip_steps:
                step_logger.info("运行模式: 完整流程（所有阶段）")
            
            # 切换到工作目录（work_wyy）
            os.chdir(work_dir)
            step_logger.info(f"工作目录: {os.getcwd()}")
            step_logger.info(f"日志保存目录: {trainlog_dir}")
            
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
                    step_logger.error(f"❌ 无效的起始阶段: {args.start_from}")
                    step_logger.info("使用 --list-steps 查看所有可用阶段")
                    return
            
            # 根据参数决定跳过哪些阶段
            skip_steps = set(args.skip_steps or [])
            
            # 过滤步骤
            steps_to_run = []
            for i, (step_id, step_name, step_func, allow_skip) in enumerate(all_steps):
                if i < start_index:
                    step_logger.info(f"⏭️  跳过阶段: {step_name} (在起始阶段之前)")
                    continue
                if step_id in skip_steps:
                    step_logger.info(f"⏭️  跳过阶段: {step_name} (用户指定跳过)")
                    continue
                steps_to_run.append((step_name, step_func, allow_skip))
            
            if not steps_to_run:
                step_logger.warning("⚠️  没有要执行的步骤，请检查参数")
                return
            
            step_logger.info(f"\n将执行 {len(steps_to_run)} 个步骤:")
            for step_name, _, _ in steps_to_run:
                step_logger.info(f"  - {step_name}")
            step_logger.info("")
            
            # 执行步骤
            results = {}
            
            for step_name, step_func, allow_skip in steps_to_run:
                with step_logger.step(step_name):
                    try:
                        result = step_func()
                        results[step_name] = result
                        
                        if not result and not allow_skip:
                            step_logger.error(f"\n❌ 关键步骤 '{step_name}' 失败，终止流水线")
                            break
                        
                        if not result and allow_skip:
                            step_logger.warning(f"\n⚠️  步骤 '{step_name}' 失败或跳过，继续执行")
                    
                    except KeyboardInterrupt:
                        step_logger.info("\n\n用户中断执行")
                        break
                    except Exception as e:
                        step_logger.error(f"\n❌ 步骤 '{step_name}' 执行异常: {e}")
                        if not allow_skip:
                            break
            
            # 总结
            with step_logger.step("流水线执行总结"):
                for step_name, result in results.items():
                    status = "✅ 成功" if result else "❌ 失败/跳过"
                    step_logger.info(f"{step_name}: {status}")
    
    finally:
        if step_logger:
            step_logger.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if step_logger:
            step_logger.info("\n\n用户中断执行")
    except Exception as e:
        if step_logger:
            step_logger.error(f"\n执行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保日志管理器被关闭
        if step_logger:
            step_logger.close()

