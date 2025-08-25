import json
import os

def check_coco_annotations(annotation_file):
    """检查COCO标注文件的格式和内容"""
    if not os.path.exists(annotation_file):
        print(f"文件不存在: {annotation_file}")
        return
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"检查文件: {annotation_file}")
    print(f"图像数量: {len(data.get('images', []))}")
    print(f"标注数量: {len(data.get('annotations', []))}")
    print(f"类别数量: {len(data.get('categories', []))}")
    
    # 检查类别
    if 'categories' in data:
        print("\n类别信息:")
        for cat in data['categories']:
            print(f"  ID: {cat['id']}, Name: {cat['name']}")
    
    # 检查前几个标注
    if 'annotations' in data and len(data['annotations']) > 0:
        print("\n前3个标注示例:")
        for i, ann in enumerate(data['annotations'][:3]):
            print(f"  标注{i+1}: image_id={ann['image_id']}, category_id={ann['category_id']}, bbox={ann['bbox']}")
    
    # 检查前几个图像
    if 'images' in data and len(data['images']) > 0:
        print("\n前3个图像示例:")
        for i, img in enumerate(data['images'][:3]):
            print(f"  图像{i+1}: id={img['id']}, file_name={img['file_name']}, size=({img['width']}, {img['height']})")

# 检查所有标注文件
annotation_dir = './datasets/DUT/annotations'
for filename in ['instances_train2017.json', 'instances_val2017.json', 'instances_test2017.json']:
    filepath = os.path.join(annotation_dir, filename)
    check_coco_annotations(filepath)
    print("-" * 50)