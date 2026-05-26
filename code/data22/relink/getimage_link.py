#从demo和demo_relink获取图片链接

import json

def extract_images(input_files, output_file):
    # 用于存储所有图片链接
    all_images = set()  # 使用集合去重

    # 遍历所有输入文件
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 提取 images 和 images_link 字段
                images = data.get("images", [])
                images_link = data.get("images_link", [])
                # 将图片链接添加到集合中
                all_images.update(images)
                all_images.update(images_link)

    # 将去重后的图片链接写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        for image in all_images:
            f.write(json.dumps({"link": image}, ensure_ascii=False) + "\n")

    print(f"已提取 {len(all_images)} 条图片链接到 {output_file}")

if __name__ == "__main__":
    # 输入文件列表
    input_files = ["demo.jsonl", "demo_relink.jsonl"]
    # 输出文件
    output_file = "demo_image.jsonl"
    # 执行提取操作
    extract_images(input_files, output_file)

