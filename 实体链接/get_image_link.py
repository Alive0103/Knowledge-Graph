import json

def extract_images(input_files, output_file):
    all_images = set() 

    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                images = data.get("images", [])
                images_link = data.get("images_link", [])
                all_images.update(images)
                all_images.update(images_link)

    with open(output_file, "w", encoding="utf-8") as f:
        for image in all_images:
            f.write(json.dumps({"link": image}, ensure_ascii=False) + "\n")

    print(f"已提取 {len(all_images)} 条图片链接到 {output_file}")

if __name__ == "__main__":
    input_files = ["demo.jsonl", "demo_relink.jsonl"]
    output_file = "demo_image.jsonl"
    extract_images(input_files, output_file)

