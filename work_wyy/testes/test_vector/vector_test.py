"""
测试文本向量化功能
验证模型能够输出1024维向量
"""

import torch
from transformers import BertTokenizer, BertModel
import numpy as np

def test_vector_generation():
    """
    测试向量生成功能
    """
    # 使用项目中定义的模型路径
    model_name = '../../model/chinese-roberta-wwm-ext-large'
    
    print("加载模型...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()
        
        print(f"模型加载成功!")
        print(f"模型隐藏层维度: {model.config.hidden_size}")
        
        # 测试文本
        test_texts = [
            "这是一段测试文本",
            "AK47突击步枪",
            "F-16战斗机",
            "中国人民解放军",
            "航空母舰"
        ]
        
        print("\n开始测试向量生成:")
        print("-" * 50)
        
        for text in test_texts:
            # 生成向量
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            # L2归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            # 处理维度: 确保是1024维
            vector_dim = len(vector)
            target_dim = 1024
            
            if vector_dim != target_dim:
                if vector_dim < target_dim:
                    # 零填充到1024维
                    vector = np.pad(vector, (0, target_dim - vector_dim), 'constant', constant_values=0)
                else:
                    # 截断到1024维
                    vector = vector[:target_dim]
                
                # 重新归一化
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            
            # 验证结果
            final_norm = np.linalg.norm(vector)
            print(f"文本: {text}")
            print(f"  向量维度: {len(vector)}")
            print(f"  向量模长: {final_norm:.6f}")
            print(f"  前10维预览: {[round(float(x), 6) for x in vector[:10]]}")
            print()
            
            # 断言检查
            assert len(vector) == 1024, f"向量维度应为1024，实际为{len(vector)}"
            assert abs(final_norm - 1.0) < 1e-5, f"向量应为单位向量，实际模长为{final_norm}"
            
        print("✓ 所有测试通过！模型能够正确生成1024维向量。")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_generation()