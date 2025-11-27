"""
检查原始数据源中是否存在该实体
"""
import pandas as pd
from urllib.parse import unquote

def check_in_data_source(data_file, target_link):
    """检查数据源文件中是否存在该实体"""
    print("="*80)
    print("检查原始数据源")
    print("="*80)
    
    try:
        # 尝试读取Excel文件
        df = pd.read_excel(data_file)
        print(f"✓ 成功读取数据文件: {data_file}")
        print(f"  总行数: {len(df)}")
        
        # URL解码目标链接
        decoded_link = unquote(target_link)
        title = decoded_link.split("/wiki/")[1] if "/wiki/" in decoded_link else ""
        
        print(f"\n查找目标实体:")
        print(f"  链接: {target_link}")
        print(f"  标题: {title}")
        
        # 检查link列
        if 'link' in df.columns:
            # 精确匹配
            exact_match = df[df['link'] == target_link]
            if len(exact_match) > 0:
                print(f"\n✓ 找到精确匹配（link列）:")
                print(f"  行数: {len(exact_match)}")
                for idx, row in exact_match.iterrows():
                    print(f"\n  行 {idx}:")
                    for col in df.columns:
                        print(f"    {col}: {str(row[col])[:100]}...")
                return True
            
            # 模糊匹配（包含标题）
            if title:
                fuzzy_match = df[df['link'].str.contains(title, na=False)]
                if len(fuzzy_match) > 0:
                    print(f"\n✓ 找到模糊匹配（link列包含'{title}'）:")
                    print(f"  行数: {len(fuzzy_match)}")
                    for idx, row in fuzzy_match.head(5).iterrows():
                        print(f"\n  行 {idx}:")
                        print(f"    link: {row['link']}")
                        if 'label' in df.columns:
                            print(f"    label: {row.get('label', 'N/A')}")
        
        # 检查label列
        if 'label' in df.columns and title:
            label_match = df[df['label'].str.contains("堪培拉", na=False) & 
                            df['label'].str.contains("两栖", na=False)]
            if len(label_match) > 0:
                print(f"\n✓ 找到包含'堪培拉'和'两栖'的实体（label列）:")
                print(f"  行数: {len(label_match)}")
                for idx, row in label_match.head(5).iterrows():
                    print(f"\n  行 {idx}:")
                    print(f"    label: {row.get('label', 'N/A')}")
                    print(f"    link: {row.get('link', 'N/A')[:80]}...")
        
        print(f"\n❌ 未在数据源中找到目标实体")
        return False
        
    except FileNotFoundError:
        print(f"❌ 数据文件不存在: {data_file}")
        print("\n请检查数据文件路径")
        return False
    except Exception as e:
        print(f"❌ 读取数据文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_all_entities_with_keyword(data_file, keyword):
    """查找所有包含关键词的实体"""
    print(f"\n{'='*80}")
    print(f"查找所有包含'{keyword}'的实体")
    print(f"{'='*80}")
    
    try:
        df = pd.read_excel(data_file)
        
        # 在label列中搜索
        if 'label' in df.columns:
            matches = df[df['label'].str.contains(keyword, na=False)]
            print(f"在label列中找到 {len(matches)} 个实体:")
            for idx, row in matches.head(10).iterrows():
                print(f"  {row.get('label', 'N/A')} -> {row.get('link', 'N/A')[:60]}...")
        
        # 在link列中搜索
        if 'link' in df.columns:
            matches = df[df['link'].str.contains(keyword, na=False)]
            print(f"\n在link列中找到 {len(matches)} 个实体:")
            for idx, row in matches.head(10).iterrows():
                print(f"  {row.get('label', 'N/A')} -> {row.get('link', 'N/A')[:60]}...")
                
    except Exception as e:
        print(f"❌ 搜索失败: {e}")

def main():
    # 目标实体链接
    target_link = "https://zh.wikipedia.org/wiki/%E5%A0%AA%E5%9F%B9%E6%8B%89%E7%BA%A7%E4%B8%A4%E6%A3%B2%E6%94%BB%E5%87%BB%E8%88%B0"
    
    # 常见的数据文件路径
    data_files = [
        "data.xlsx",
        "实体数据.xlsx",
        "toesdata.xlsx",
        "../data.xlsx",
        "实体链接/data.xlsx"
    ]
    
    found = False
    for data_file in data_files:
        if check_in_data_source(data_file, target_link):
            found = True
            break
    
    if not found:
        print(f"\n{'='*80}")
        print("数据源检查结果")
        print(f"{'='*80}")
        print("❌ 目标实体不在数据源中")
        print("\n【解决方案】")
        print("1. 检查数据导入过程，确认是否遗漏了该实体")
        print("2. 如果该实体确实不在原始数据中，需要：")
        print("   - 从Wikipedia获取该实体数据")
        print("   - 添加到数据源文件")
        print("   - 重新导入到ES")
        print("3. 或者检查是否有其他数据文件包含该实体")
    
    # 查找所有包含"堪培拉"的实体
    for data_file in data_files:
        try:
            check_all_entities_with_keyword(data_file, "堪培拉")
            break
        except:
            continue

if __name__ == "__main__":
    main()

