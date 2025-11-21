"""
调试和运行脚本
用于检查环境、修复配置并运行项目
"""
import os
import sys
import json
from pathlib import Path

def check_elasticsearch():
    """检查Elasticsearch连接"""
    print("=" * 50)
    print("步骤1: 检查Elasticsearch连接")
    print("=" * 50)
    try:
        from 实体链接.es_client import es
        
        if es.ping():
            print("✓ Elasticsearch连接成功")
            # 获取ES版本信息
            info = es.info()
            print(f"  ES版本: {info['version']['number']}")
            print(f"  集群名称: {info['cluster_name']}")
            return True
        else:
            print("✗ Elasticsearch连接失败")
            return False
    except Exception as e:
        print(f"✗ Elasticsearch连接错误: {e}")
        print("  请检查阿里云Elasticsearch配置")
        return False

def check_data_files():
    """检查数据文件"""
    print("\n" + "=" * 50)
    print("步骤2: 检查数据文件")
    print("=" * 50)
    
    data_files = {
        "zh_wiki_v2.jsonl": ["实体链接/zh_wiki_v2.jsonl", "中英文维基-部分/zh_wiki_v2.jsonl"],
        "en_wiki_v3.jsonl": ["实体链接/en_wiki_v3.jsonl", "中英文维基-部分/en_wiki_v3.jsonl"]
    }
    
    found_files = {}
    for name, paths in data_files.items():
        for path in paths:
            full_path = Path(path)
            if full_path.exists():
                size = full_path.stat().st_size / (1024 * 1024)  # MB
                print(f"✓ 找到 {name}: {path} ({size:.2f} MB)")
                found_files[name] = str(full_path)
                break
        else:
            print(f"✗ 未找到 {name}")
    
    # 检查数据格式
    if found_files:
        print("\n检查数据格式...")
        for name, path in found_files.items():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    data = json.loads(first_line)
                    print(f"✓ {name} 格式正确")
                    print(f"  字段: {list(data.keys())[:5]}...")
            except Exception as e:
                print(f"✗ {name} 格式错误: {e}")
    
    return found_files

def check_model_path():
    """检查模型路径"""
    print("\n" + "=" * 50)
    print("步骤3: 检查模型路径")
    print("=" * 50)
    
    model_paths = {
        "Chinese-RoBERTa": "D:/model/chinese-roberta-wwm-ext-large",
        "LaBSE": "跨语言实体对齐/data/LaBSE"
    }
    
    model_status = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"✓ {name} 模型路径存在: {path}")
            model_status[name] = path
        else:
            print(f"✗ {name} 模型路径不存在: {path}")
            print(f"  提示: 可以从HuggingFace下载模型")
            model_status[name] = None
    
    return model_status

def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 50)
    print("步骤4: 检查依赖包")
    print("=" * 50)
    
    required_packages = {
        "torch": "torch",
        "transformers": "transformers",
        "elasticsearch": "elasticsearch",
        "pandas": "pandas",
        "tqdm": "tqdm",
        "zhipuai": "zhipuai",
        "faiss": "faiss-cpu",
        "flask": "flask"
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n缺少的包: {', '.join(missing_packages)}")
        print("安装命令: pip install " + " ".join(missing_packages))
    
    return len(missing_packages) == 0

def fix_config_files(data_files):
    """修复配置文件"""
    print("\n" + "=" * 50)
    print("步骤5: 修复配置文件")
    print("=" * 50)
    
    # 修复toesdata.py
    toesdata_path = Path("实体链接/toesdata.py")
    if toesdata_path.exists() and data_files:
        print("修复 toesdata.py...")
        content = toesdata_path.read_text(encoding='utf-8')
        
        # 检查是否需要修复数据文件路径
        if 'import_data_to_es("data2.jsonl"' in content:
            # 使用找到的数据文件
            zh_file = data_files.get("zh_wiki_v2.jsonl")
            if zh_file:
                # 创建备份
                backup_path = toesdata_path.with_suffix('.py.bak')
                if not backup_path.exists():
                    backup_path.write_text(content, encoding='utf-8')
                    print(f"  已创建备份: {backup_path}")
                
                # 修复路径（注释掉旧路径，添加新路径）
                new_content = content.replace(
                    'import_data_to_es("data2.jsonl", INDEX_MILITARY, transform_military_data)',
                    f'import_data_to_es(r"{zh_file}", INDEX_MILITARY, transform_military_data)'
                )
                toesdata_path.write_text(new_content, encoding='utf-8')
                print(f"  ✓ 已更新数据文件路径为: {zh_file}")
    
    # 检查search_withllm.py中的模型路径
    search_path = Path("实体链接/search_withllm.py")
    if search_path.exists():
        print("检查 search_withllm.py...")
        content = search_path.read_text(encoding='utf-8')
        if 'D:/model/chinese-roberta-wwm-ext-large' in content:
            print("  提示: 模型路径为硬编码，如需修改请手动编辑")
            print("  当前路径: D:/model/chinese-roberta-wwm-ext-large")

def create_es_index():
    """创建Elasticsearch索引"""
    print("\n" + "=" * 50)
    print("步骤6: 创建Elasticsearch索引")
    print("=" * 50)
    
    try:
        sys.path.insert(0, "实体链接")
        from createES import es
        
        if es.ping():
            print("✓ 正在创建索引...")
            # 导入createES模块并执行
            import importlib.util
            spec = importlib.util.spec_from_file_location("createES", "实体链接/createES.py")
            createES = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(createES)
            print("✓ 索引创建完成（如果已存在会被删除重建）")
            return True
        else:
            print("✗ Elasticsearch未连接")
            return False
    except Exception as e:
        print(f"✗ 创建索引失败: {e}")
        return False

def import_data(data_files):
    """导入数据"""
    print("\n" + "=" * 50)
    print("步骤7: 导入数据到Elasticsearch")
    print("=" * 50)
    
    if not data_files:
        print("✗ 没有可用的数据文件")
        return False
    
    try:
        sys.path.insert(0, "实体链接")
        from toesdata import import_data_to_es, transform_military_data, INDEX_MILITARY
        
        zh_file = data_files.get("zh_wiki_v2.jsonl")
        if zh_file:
            print(f"正在导入数据: {zh_file}")
            import_data_to_es(zh_file, INDEX_MILITARY, transform_military_data, batch_size=100)
            print("✓ 数据导入完成")
            return True
        else:
            print("✗ 未找到中文数据文件")
            return False
    except Exception as e:
        print(f"✗ 数据导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search():
    """测试搜索功能"""
    print("\n" + "=" * 50)
    print("步骤8: 测试搜索功能")
    print("=" * 50)
    
    try:
        sys.path.insert(0, "实体链接")
        from search_label_aliases import hybrid_search
        
        test_queries = ["AK47", "F-16", "步枪"]
        print("测试查询:")
        for query in test_queries:
            try:
                results = hybrid_search(query, top_k=5)
                print(f"  查询: {query}")
                print(f"  结果数量: {len(results)}")
                if results:
                    print(f"  第一个结果: {results[0].get('label', 'N/A')}")
                print()
            except Exception as e:
                print(f"  ✗ 查询 '{query}' 失败: {e}")
        
        print("✓ 搜索测试完成")
        return True
    except Exception as e:
        print(f"✗ 搜索测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("知识图谱项目 - 调试和运行脚本")
    print("=" * 60 + "\n")
    
    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}\n")
    
    results = {}
    
    # 1. 检查Elasticsearch
    results['elasticsearch'] = check_elasticsearch()
    
    # 2. 检查数据文件
    data_files = check_data_files()
    results['data_files'] = len(data_files) > 0
    
    # 3. 检查模型路径
    model_status = check_model_path()
    results['models'] = any(model_status.values())
    
    # 4. 检查依赖
    results['dependencies'] = check_dependencies()
    
    # 5. 修复配置
    if data_files:
        fix_config_files(data_files)
    
    # 6. 创建索引（如果ES可用）
    if results['elasticsearch']:
        results['create_index'] = create_es_index()
    
    # 7. 导入数据（如果ES可用且有数据）
    if results['elasticsearch'] and data_files:
        user_input = input("\n是否导入数据到Elasticsearch? (y/n): ")
        if user_input.lower() == 'y':
            results['import_data'] = import_data(data_files)
    
    # 8. 测试搜索（如果数据已导入）
    if results.get('import_data'):
        user_input = input("\n是否测试搜索功能? (y/n): ")
        if user_input.lower() == 'y':
            results['test_search'] = test_search()
    
    # 总结
    print("\n" + "=" * 60)
    print("运行总结")
    print("=" * 60)
    for step, status in results.items():
        status_str = "✓ 通过" if status else "✗ 失败"
        print(f"{step:20s}: {status_str}")
    
    print("\n" + "=" * 60)
    print("调试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

