import json
import re
import argparse
import os

def extract_question_from_messages(messages):
    """从messages中提取用户的问题"""
    for message in messages:
        if message.get('role') == 'user':
            content = message.get('content', '')
            # 查找 "The original question from the user is:" 后面的内容
            match = re.search(r'The original question from the user is:\s*(.+)', content)
            if match:
                return match.group(1).strip()
    return None

def convert_jsonl_format(input_file_path, output_file_path, dataset_name="converted_dataset"):
    """
    从JSONL文件读取数据并转换格式
    
    Args:
        input_file_path: 输入JSONL文件路径
        output_file_path: 输出JSONL文件路径
        dataset_name: 数据集名称
    """
    converted_data = []
    skipped_count = 0
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            for line_num, line in enumerate(input_file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析每行JSON
                    item = json.loads(line)
                    
                    # 提取基本信息
                    uuid = item.get('meta', {}).get('uuid', '')
                    case = item.get('meta', {}).get('case', '')
                    
                    # 从messages中提取问题
                    messages = item.get('data', {}).get('messages', [])
                    question = extract_question_from_messages(messages)
                    
                    # 提取答案
                    gold_answer = item.get('gold_answer', '')
                    
                    # 如果没有找到问题，跳过这条数据
                    if not question:
                        print(f"Warning: No question found in line {line_num}, skipping...")
                        skipped_count += 1
                        continue
                    
                    # 构建转换后的数据
                    converted_item = {
                        "id": f"{dataset_name}_{len(converted_data)}",
                        "question": question,
                        "answers": [gold_answer] if gold_answer else [],
                        "metadata": {
                            "dataset": dataset_name,
                            "index": len(converted_data),
                            "original_uuid": uuid,
                            "original_case": case,
                            "original_line": line_num
                        }
                    }
                    
                    converted_data.append(converted_item)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    skipped_count += 1
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    skipped_count += 1
                    continue
        
        # 写入输出文件
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for item in converted_data:
                output_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Successfully processed {len(converted_data)} items")
        if skipped_count > 0:
            print(f"⚠️  Skipped {skipped_count} items due to errors")
        print(f"📁 Output saved to: {output_file_path}")
        
        return converted_data
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file_path}' not found")
        return None
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

def preview_conversion(input_file_path, num_lines=3):
    """
    预览转换结果，显示前几行的转换效果
    
    Args:
        input_file_path: 输入文件路径
        num_lines: 预览的行数
    """
    print(f"🔍 Previewing first {num_lines} conversions from {input_file_path}:")
    print("=" * 80)
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    
                    # 提取信息
                    uuid = item.get('meta', {}).get('uuid', '')
                    case = item.get('meta', {}).get('case', '')
                    messages = item.get('data', {}).get('messages', [])
                    question = extract_question_from_messages(messages)
                    gold_answer = item.get('data', {}).get('gold_answer', '')
                    
                    print(f"\n📋 Line {i+1}:")
                    print(f"   UUID: {uuid}")
                    print(f"   Case: {case}")
                    print(f"   Question: {question}")
                    print(f"   Answer: {gold_answer}")
                    
                    # 显示转换后的格式
                    converted = {
                        "id": f"preview_{i}",
                        "question": question,
                        "answers": [gold_answer] if gold_answer else [],
                        "metadata": {
                            "dataset": "preview",
                            "index": i,
                            "original_uuid": uuid,
                            "original_case": case
                        }
                    }
                    
                    print(f"   Converted: {json.dumps(converted, ensure_ascii=False)}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing line {i+1}: {e}")
    
    except FileNotFoundError:
        print(f"❌ File not found: {input_file_path}")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='Convert JSONL format for Q&A dataset')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('-o', '--output', help='Output JSONL file path (default: input_converted.jsonl)')
    parser.add_argument('-d', '--dataset', default='converted_dataset', help='Dataset name (default: converted_dataset)')
    parser.add_argument('-p', '--preview', action='store_true', help='Preview conversion without saving')
    parser.add_argument('-n', '--preview-lines', type=int, default=3, help='Number of lines to preview (default: 3)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' does not exist")
        return
    
    # 如果只是预览
    if args.preview:
        preview_conversion(args.input_file, args.preview_lines)
        return
    
    # 设置输出文件路径
    if args.output:
        output_file = args.output
    else:
        # 默认输出文件名
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_converted.jsonl"
    
    # 执行转换
    print(f"🚀 Starting conversion...")
    print(f"📥 Input: {args.input_file}")
    print(f"📤 Output: {output_file}")
    print(f"🏷️  Dataset: {args.dataset}")
    print("-" * 50)
    
    result = convert_jsonl_format(args.input_file, output_file, args.dataset)
    
    if result:
        print(f"\n🎉 Conversion completed successfully!")

if __name__ == "__main__":
    # 如果直接运行脚本，使用命令行参数
    main()
