#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
下载实体链接所需的翻译模型
修复版：解决网络连接和依赖检测问题
"""

import os
import sys
import time
import subprocess
import importlib
import requests
from transformers import AutoTokenizer, MarianMTModel

# 设置环境变量
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # 禁用符号链接警告
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用HF镜像站点
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 确保在线模式


def install_package(package_name):
    """安装必要的Python包"""
    try:
        print(f"正在安装 {package_name}...")
        # 使用国内镜像源加速安装
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            package_name, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/",
            "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
        ])
        print(f"✓ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装 {package_name} 失败: {e}")
        # 尝试使用默认源
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✓ {package_name} 安装成功")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ 使用默认源安装 {package_name} 也失败")
            return False


def check_dependency(package_name):
    """更健壮的依赖检查函数"""
    try:
        # 提取基础包名（去掉版本说明）
        base_name = package_name.split('>')[0].split('<')[0].split('=')[0].split('[')[0].strip()

        # 特殊处理protobuf，因为它可能有不同的导入名
        if base_name.lower() == 'protobuf':
            try:
                import google.protobuf
                print(f"✓ {package_name} 已安装 (通过google.protobuf)")
                return True
            except ImportError:
                pass

        # 正常导入检查
        importlib.import_module(base_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError as e:
        # 特殊处理：protobuf有时以不同名称存在
        if base_name.lower() == 'protobuf':
            try:
                # 检查是否已安装但名称不同
                result = subprocess.run([
                    sys.executable, "-c",
                    "import pkg_resources; print([pkg.project_name for pkg in pkg_resources.working_set if 'protobuf' in pkg.project_name.lower()])"
                ], capture_output=True, text=True)

                if 'protobuf' in result.stdout.lower():
                    print(f"✓ {package_name} 已安装 (通过pkg_resources检测)")
                    return True
            except:
                pass

        print(f"❌ {package_name} 未安装: {e}")
        return False


def check_and_install_dependencies():
    """检查并安装必要的依赖"""
    dependencies = [
        "transformers>=4.20.0",
        "torch>=1.9.0",
        "sentencepiece>=0.1.91",
        "protobuf>=3.20.0"
    ]

    print("检查依赖安装情况...")

    missing_deps = []
    for dep in dependencies:
        if not check_dependency(dep):
            missing_deps.append(dep)

    if missing_deps:
        print(f"\n缺少以下依赖: {', '.join(missing_deps)}")
        response = input("是否自动安装? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            for dep in missing_deps:
                # 只安装基础包名
                base_name = dep.split('>')[0].split('<')[0].split('=')[0].split('[')[0].strip()
                if not install_package(base_name):
                    print(f"安装 {base_name} 失败，请手动安装: pip install {base_name}")
                    return False
        else:
            print("请手动安装依赖:")
            for dep in missing_deps:
                base_name = dep.split('>')[0].split('<')[0].split('=')[0].split('[')[0].strip()
                print(f"pip install {base_name}")
            return False

    return True


def test_network_connection():
    """测试网络连接"""
    test_urls = [
        "https://hf-mirror.com",
        "https://huggingface.co",
        "https://www.baidu.com"
    ]

    print("测试网络连接...")

    for url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✓ 可以访问: {url}")
                return True
            else:
                print(f"⚠ 可以访问但状态异常: {url} (状态码: {response.status_code})")
        except Exception as e:
            print(f"❌ 无法访问 {url}: {e}")

    print("\n网络连接测试失败，请检查:")
    print("1. 网络连接是否正常")
    print("2. 是否使用了代理，需要设置代理环境变量")
    print("3. 防火墙设置")

    # 询问是否设置代理
    try:
        response = input("\n是否设置HTTP代理? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            proxy = input("请输入代理地址 (如 http://127.0.0.1:1080): ").strip()
            if proxy:
                os.environ['HTTP_PROXY'] = proxy
                os.environ['HTTPS_PROXY'] = proxy
                print(f"已设置代理: {proxy}")
                return test_network_connection()
    except:
        pass

    return False


def download_with_retry(model_name, save_path, max_retries=3):
    """带重试机制的模型下载函数"""
    print(f"下载模型: {model_name}")

    # 检查是否已存在模型文件
    config_path = os.path.join(save_path, "config.json")
    if os.path.exists(config_path):
        print(f"模型已存在，跳过下载: {save_path}")
        return True

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    for attempt in range(max_retries):
        try:
            print(f"├─ 尝试 #{attempt + 1}/{max_retries}")

            # 动态设置超时
            timeout = 30 + (attempt * 30)  # 30, 60, 90秒

            print("├─ 下载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, timeout=timeout)

            print("├─ 下载模型...")
            model = MarianMTModel.from_pretrained(model_name, timeout=timeout)

            print("└─ 保存到本地...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            print(f"✓ 模型已保存至: {save_path}")
            return True

        except Exception as e:
            error_msg = str(e)
            print(f"├─ 下载失败: {error_msg}")

            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"├─ 等待 {wait_time} 秒后重试...")

                # 在重试前尝试切换端点
                if attempt == 1:  # 第二次重试时切换
                    current = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
                    new = 'https://hf-mirror.com' if current == 'https://huggingface.co' else 'https://huggingface.co'
                    print(f"├─ 切换到: {new}")
                    os.environ['HF_ENDPOINT'] = new

                time.sleep(wait_time)
            else:
                print("❌ 下载失败，已达最大重试次数")
                return False

    return False


def manual_download_option():
    """提供手动下载选项"""
    print("\n" + "=" * 60)
    print("自动下载失败，请尝试手动下载:")
    print("1. 访问以下链接手动下载模型文件:")
    print("   - https://hf-mirror.com/Helsinki-NLP/opus-mt-zh-en")
    print("   - https://hf-mirror.com/Helsinki-NLP/opus-mt-en-zh")
    print("2. 下载所有文件到对应目录")
    print("3. 确保目录结构如下:")
    print("   models/")
    print("   ├── opus-mt-zh-en/")
    print("   │   ├── pytorch_model.bin")
    print("   │   ├── config.json")
    print("   │   ├── *.spm 等所有文件")
    print("   └── opus-mt-en-zh/")
    print("       ├── pytorch_model.bin")
    print("       ├── config.json")
    print("       └── *.spm 等所有文件")
    print("=" * 60)


def main():
    print("=" * 60)
    print("HuggingFace翻译模型下载脚本")
    print("修复网络连接和依赖检测问题")
    print("=" * 60)

    # 测试网络连接
    if not test_network_connection():
        print("❌ 网络连接测试失败")
        manual_download_option()
        return

    # 检查并安装依赖
    if not check_and_install_dependencies():
        print("❌ 依赖检查失败")
        return

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")

    print(f"\n模型将保存到: {model_dir}")

    # 定义需要下载的模型
    models_to_download = [
        {
            "name": "Helsinki-NLP/opus-mt-zh-en",
            "save_path": os.path.join(model_dir, "opus-mt-zh-en"),
            "description": "中文到英文翻译模型"
        },
        {
            "name": "Helsinki-NLP/opus-mt-en-zh",
            "save_path": os.path.join(model_dir, "opus-mt-en-zh"),
            "description": "英文到中文翻译模型"
        }
    ]

    # 显示下载计划
    print("准备下载以下模型:")
    for i, model_info in enumerate(models_to_download, 1):
        print(f"{i}. {model_info['description']}")
        print(f"   模型: {model_info['name']}")
        print(f"   路径: {model_info['save_path']}")

    # 确认下载
    try:
        response = input("\n是否开始下载? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("下载已取消")
            return
    except:
        print("下载已取消")
        return

    # 下载所有模型
    success_count = 0
    for model_info in models_to_download:
        print(f"\n{'=' * 60}")
        print(f"[{model_info['description']}]")
        print(f"模型: {model_info['name']}")
        print(f"保存到: {model_info['save_path']}")
        print('=' * 60)

        if download_with_retry(model_info["name"], model_info["save_path"]):
            success_count += 1
        else:
            print(f"❌ {model_info['description']} 下载失败")

    # 输出总结
    print("=" * 60)
    print(f"下载完成! 成功下载 {success_count}/{len(models_to_download)} 个模型.")

    if success_count == len(models_to_download):
        print("✓ 所有模型均已成功下载")
        print("现在可以运行 add_zhaliases.py 了")
    else:
        print("⚠ 部分模型下载失败")
        manual_download_option()


if __name__ == "__main__":
    main()