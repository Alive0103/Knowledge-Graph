#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER模型微调前置条件检验脚本
在运行微调之前，检查所有必要的条件和数据
"""

import os
import json
import glob
import sys
from pathlib import Path

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")


class PrerequisitesChecker:
    """前置条件检查器"""
    
    def __init__(self):
        # 获取配置文件中的路径
        try:
            from config import DATA_DIR, MODEL_DIR
            self.base_dir = Path(__file__).parent  # local 目录
            self.data_dir = Path(DATA_DIR)  # work_wyy/data
            self.model_dir = Path(MODEL_DIR)  # work_wyy/model
        except ImportError:
            # 如果无法导入配置，使用默认路径
            self.base_dir = Path(__file__).parent
            self.data_dir = self.base_dir.parent / "data"
            self.model_dir = self.base_dir.parent / "model"
        self.errors = []
        self.warnings = []
        
    def check_model_exists(self):
        """检查基础模型是否存在"""
        print_header("1. 检查基础模型")
        
        model_path = self.model_dir / "chinese-roberta-wwm-ext-large"
        if model_path.exists():
            # 检查必要的文件
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            all_exist = all((model_path / f).exists() for f in required_files)
            
            if all_exist:
                print_success(f"基础模型存在: {model_path}")
                return True
            else:
                missing = [f for f in required_files if not (model_path / f).exists()]
                print_error(f"模型文件不完整，缺少: {', '.join(missing)}")
                self.errors.append(f"模型文件不完整: {model_path}")
                return False
        else:
            print_error(f"基础模型不存在: {model_path}")
            self.errors.append(f"基础模型不存在: {model_path}")
            return False
    
    def check_traindata(self):
        """检查traindata目录"""
        print_header("2. 检查traindata目录")
        
        traindata_dir = self.data_dir / "traindata"
        if not traindata_dir.exists():
            print_warning(f"traindata目录不存在: {traindata_dir}")
            self.warnings.append("traindata目录不存在")
            return False
        
        train_files = list(traindata_dir.glob("*_ner_train.json"))
        dev_files = list(traindata_dir.glob("*_ner_dev.json"))
        
        if not train_files and not dev_files:
            print_warning("traindata目录中没有找到训练或验证文件")
            self.warnings.append("traindata目录为空")
            return False
        
        print_success(f"找到 {len(train_files)} 个训练文件")
        print_success(f"找到 {len(dev_files)} 个验证文件")
        
        # 检查文件格式
        total_samples = 0
        for file_path in train_files + dev_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        total_samples += len(data)
                        print_info(f"  {file_path.name}: {len(data)} 条数据")
                    else:
                        print_warning(f"  {file_path.name}: 不是JSON数组格式")
            except Exception as e:
                print_error(f"  {file_path.name}: 读取失败 - {e}")
                self.errors.append(f"文件读取失败: {file_path.name}")
        
        if total_samples > 0:
            print_success(f"总计: {total_samples} 条数据")
        
        return len(train_files) > 0 or len(dev_files) > 0
    
    def check_ccks_data(self):
        """检查CCKS数据"""
        print_header("3. 检查CCKS数据")
        
        ccks_dir = self.data_dir / "ccks_ner" / "militray" / "PreModel_Encoder_CRF" / "ccks_8_data_v2"
        
        if not ccks_dir.exists():
            print_warning(f"CCKS数据目录不存在: {ccks_dir}")
            self.warnings.append("CCKS数据目录不存在")
            return False
        
        train_dir = ccks_dir / "train"
        validate_file = ccks_dir / "validate_data.json"
        
        has_data = False
        
        if train_dir.exists():
            json_files = list(train_dir.glob("*.json"))
            if json_files:
                print_success(f"找到 {len(json_files)} 个CCKS训练文件")
                has_data = True
                
                # 检查前几个文件的编码
                checked = 0
                encoding_issues = 0
                for json_file in json_files[:5]:
                    checked += 1
                    try:
                        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
                        success = False
                        for enc in encodings:
                            try:
                                with open(json_file, 'r', encoding=enc) as f:
                                    json.load(f)
                                success = True
                                break
                            except:
                                continue
                        if not success:
                            encoding_issues += 1
                    except:
                        encoding_issues += 1
                
                if encoding_issues > 0:
                    print_warning(f"前{checked}个文件中有{encoding_issues}个存在编码问题（将自动处理）")
            else:
                print_warning("CCKS train目录中没有JSON文件")
        
        if validate_file.exists():
            print_success("找到validate_data.json")
            has_data = True
        
        if not has_data:
            self.warnings.append("CCKS数据目录为空")
        
        return has_data
    
    def check_nlp_datasets(self):
        """检查nlp_datasets数据（可选）"""
        print_header("4. 检查nlp_datasets数据（可选）")
        
        msra_dir = self.data_dir / "nlp_datasets" / "ner" / "msra"
        
        if not msra_dir.exists():
            print_info("nlp_datasets目录不存在（这是可选的）")
            return True
        
        train_file = msra_dir / "msra_train_bio.txt"
        test_file = msra_dir / "msra_test_bio.txt"
        
        if train_file.exists():
            print_info(f"找到MSRA训练文件: {train_file.name}")
            print_warning("注意: MSRA是通用NER数据，实体类型与军事领域不匹配")
        
        if test_file.exists():
            print_info(f"找到MSRA测试文件: {test_file.name}")
        
        return True
    
    def check_output_directory(self):
        """检查输出目录"""
        print_header("5. 检查输出目录")
        
        output_dir = self.model_dir / "ner_finetuned"
        
        if output_dir.exists():
            print_warning(f"输出目录已存在: {output_dir}")
            print_info("训练将覆盖现有模型")
        else:
            print_success(f"输出目录将创建: {output_dir}")
        
        return True
    
    def check_dependencies(self):
        """检查Python依赖"""
        print_header("6. 检查Python依赖")
        
        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'Transformers',
            'sklearn': 'scikit-learn',
            'numpy': 'NumPy',
            'tqdm': 'tqdm'
        }
        
        all_ok = True
        for package, name in required_packages.items():
            try:
                __import__(package)
                print_success(f"{name} 已安装")
            except ImportError:
                print_error(f"{name} 未安装 (pip install {package})")
                self.errors.append(f"缺少依赖: {name}")
                all_ok = False
        
        # 检查CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print_success(f"CUDA可用: {torch.cuda.get_device_name(0)}")
            else:
                print_warning("CUDA不可用，将使用CPU训练（速度较慢）")
        except:
            pass
        
        return all_ok
    
    def check_data_statistics(self):
        """统计数据信息"""
        print_header("7. 数据统计")
        
        stats = {
            'traindata': {'train': 0, 'dev': 0, 'types': set()},
            'ccks': {'files': 0, 'types': set()},
            'total_samples': 0
        }
        
        # 统计traindata
        traindata_dir = self.data_dir / "traindata"
        if traindata_dir.exists():
            for file_path in traindata_dir.glob("*_ner_train.json"):
                try:
                    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
                    data = None
                    for enc in encodings:
                        try:
                            with open(file_path, 'r', encoding=enc) as f:
                                data = json.load(f)
                            break
                        except:
                            continue
                    
                    if data and isinstance(data, list):
                        stats['traindata']['train'] += len(data)
                        for item in data:
                            for entity in item.get('entities', []):
                                entity_type = entity.get('type', '')
                                if entity_type:
                                    stats['traindata']['types'].add(entity_type)
                except Exception as e:
                    print_warning(f"  读取 {file_path.name} 失败: {e}")
            
            for file_path in traindata_dir.glob("*_ner_dev.json"):
                try:
                    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
                    data = None
                    for enc in encodings:
                        try:
                            with open(file_path, 'r', encoding=enc) as f:
                                data = json.load(f)
                            break
                        except:
                            continue
                    
                    if data and isinstance(data, list):
                        stats['traindata']['dev'] += len(data)
                        for item in data:
                            for entity in item.get('entities', []):
                                entity_type = entity.get('type', '')
                                if entity_type:
                                    stats['traindata']['types'].add(entity_type)
                except Exception as e:
                    print_warning(f"  读取 {file_path.name} 失败: {e}")
        
        # 统计CCKS数据
        ccks_dir = self.data_dir / "ccks_ner" / "militray" / "PreModel_Encoder_CRF" / "ccks_8_data_v2" / "train"
        if ccks_dir.exists():
            json_files = list(ccks_dir.glob("*.json"))
            stats['ccks']['files'] = len(json_files)
            # 检查前几个文件的实体类型
            for json_file in json_files[:10]:
                try:
                    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
                    data = None
                    for enc in encodings:
                        try:
                            with open(json_file, 'r', encoding=enc) as f:
                                data = json.load(f)
                            break
                        except:
                            continue
                    
                    if data and isinstance(data, dict):
                        entities = data.get('entities', [])
                        for entity in entities:
                            entity_type = entity.get('label_type', entity.get('type', ''))
                            if entity_type:
                                stats['ccks']['types'].add(entity_type)
                except:
                    pass
        
        stats['total_samples'] = stats['traindata']['train'] + stats['traindata']['dev']
        
        print_info(f"训练数据: {stats['traindata']['train']} 条")
        print_info(f"验证数据: {stats['traindata']['dev']} 条")
        if stats['ccks']['files'] > 0:
            print_info(f"CCKS数据: {stats['ccks']['files']} 个文件")
        print_info(f"总计: {stats['total_samples']} 条")
        
        all_types = stats['traindata']['types'].union(stats['ccks']['types'])
        if all_types:
            print_info(f"实体类型 ({len(all_types)} 种): {', '.join(sorted(all_types))}")
        
        if stats['total_samples'] == 0:
            print_error("没有找到任何训练数据！")
            self.errors.append("没有训练数据")
        elif stats['total_samples'] < 100:
            print_warning(f"训练数据较少 ({stats['total_samples']} 条)，建议至少100条")
            self.warnings.append(f"训练数据较少: {stats['total_samples']} 条")
        
        return stats['total_samples'] > 0
    
    def run_all_checks(self):
        """运行所有检查"""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}NER模型微调前置条件检查{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
        
        checks = [
            ("基础模型", self.check_model_exists),
            ("训练数据", self.check_traindata),
            ("CCKS数据", self.check_ccks_data),
            ("NLP数据集", self.check_nlp_datasets),
            ("输出目录", self.check_output_directory),
            ("Python依赖", self.check_dependencies),
            ("数据统计", self.check_data_statistics),
        ]
        
        results = {}
        for name, check_func in checks:
            try:
                results[name] = check_func()
            except Exception as e:
                print_error(f"检查 {name} 时出错: {e}")
                results[name] = False
                self.errors.append(f"检查失败: {name}")
        
        # 总结
        print_header("检查总结")
        
        if self.errors:
            print_error(f"发现 {len(self.errors)} 个错误:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print_warning(f"发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors:
            print_success("所有必要检查通过！可以开始训练。")
            print(f"\n{Colors.GREEN}{Colors.BOLD}运行训练: python finetune_ner_model.py{Colors.END}\n")
            return True
        else:
            print_error("存在错误，请先解决上述问题再开始训练。")
            return False


def main():
    checker = PrerequisitesChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

