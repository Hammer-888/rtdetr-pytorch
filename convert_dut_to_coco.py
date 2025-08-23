import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from PIL import Image
import argparse

def parse_xml(xml_file):
    """解析XML文件，提取标注信息"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图像信息
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # 获取所有目标框
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }

def create_coco_format():
    """创建COCO格式的基础结构"""
    coco_format = {
        "info": {
            "description": "DUT Anti-UAV Detection Dataset in COCO format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "DUT Dataset Converter",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown License",
                "url": ""
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "UAV",
                "supercategory": "vehicle"
            }
        ],
        "images": [],
        "annotations": []
    }
    return coco_format

def convert_dataset(dataset_path, split):
    """转换指定分割的数据集"""
    print(f"Converting {split} dataset...")
    
    img_dir = os.path.join(dataset_path, split, 'img')
    xml_dir = os.path.join(dataset_path, split, 'xml')
    
    if not os.path.exists(img_dir) or not os.path.exists(xml_dir):
        print(f"Warning: {split} directory not found, skipping...")
        return None
    
    coco_data = create_coco_format()
    
    image_id = 1
    annotation_id = 1
    
    # 获取所有XML文件
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    xml_files.sort()
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        
        try:
            # 解析XML
            annotation_data = parse_xml(xml_path)
            
            # 检查对应的图像文件是否存在
            img_path = os.path.join(img_dir, annotation_data['filename'])
            if not os.path.exists(img_path):
                print(f"Warning: Image {annotation_data['filename']} not found, skipping...")
                continue
            
            # 添加图像信息
            image_info = {
                "id": image_id,
                "width": annotation_data['width'],
                "height": annotation_data['height'],
                "file_name": annotation_data['filename'],
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            }
            coco_data['images'].append(image_info)
            
            # 添加标注信息
            for obj in annotation_data['objects']:
                xmin, ymin, xmax, ymax = obj['bbox']
                width = xmax - xmin
                height = ymax - ymin
                area = width * height
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # UAV类别ID为1
                    "segmentation": [],
                    "area": area,
                    "bbox": [xmin, ymin, width, height],  # COCO格式: [x, y, width, height]
                    "iscrowd": 0
                }
                coco_data['annotations'].append(annotation)
                annotation_id += 1
            
            image_id += 1
            
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
            continue
    
    print(f"Processed {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations for {split}")
    return coco_data

def main():
    parser = argparse.ArgumentParser(description='Convert DUT dataset XML annotations to COCO format')
    parser.add_argument('--dataset_path', type=str, default='./datasets/DUT', 
                       help='Path to DUT dataset directory')
    parser.add_argument('--output_dir', type=str, default='./datasets/DUT/annotations', 
                       help='Output directory for COCO annotation files')
    
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换各个分割
    splits = ['train', 'val', 'test']
    
    for split in splits:
        coco_data = convert_dataset(dataset_path, split)
        
        if coco_data is not None:
            # 保存COCO格式的标注文件
            if split == 'val':
                output_file = os.path.join(output_dir, 'instances_val2017.json')
            elif split == 'train':
                output_file = os.path.join(output_dir, 'instances_train2017.json')
            else:
                output_file = os.path.join(output_dir, f'instances_{split}2017.json')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {split} annotations to {output_file}")
            print(f"  - Images: {len(coco_data['images'])}")
            print(f"  - Annotations: {len(coco_data['annotations'])}")
            print()
    
    print("Conversion completed!")
    print(f"\nCOCO annotation files saved in: {output_dir}")
    print("\nFile structure:")
    print("  - instances_train2017.json (training set)")
    print("  - instances_val2017.json (validation set, if exists)")
    print("  - instances_test2017.json (test set)")

if __name__ == '__main__':
    main()