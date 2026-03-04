#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色内容显示脚本
用于清晰地显示JSON中每个role和对应的content
"""

import json
import sys
import argparse
from typing import List, Dict, Any


def display_roles(data: List[Dict[str, Any]], show_line_numbers: bool = False):
    """
    显示每个role和对应的content
    
    Args:
        data: JSON数据列表
        show_line_numbers: 是否显示行号
    """
    for i, item in enumerate(data, 1):
        role = item.get('role', 'unknown')
        content = item.get('content', '')
        
        print(f"{'='*60}")
        print(f"角色 {i}: {role.upper()}")
        print(f"{'='*60}")
        
        if show_line_numbers:
            # 显示行号
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                print(f"{line_num:3d}| {line}")
        else:
            print(content)
        
        print()  # 空行分隔


def display_roles_from_file(input_file: str, show_line_numbers: bool = False):
    """
    从文件读取JSON并显示角色内容
    
    Args:
        input_file: 输入文件路径
        show_line_numbers: 是否显示行号
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        data = json.loads(content)
        
        if isinstance(data, list):
            display_roles(data, show_line_numbers)
        elif isinstance(data, dict) and 'messages' in data:
            # 如果是包含messages字段的对象
            display_roles(data['messages'], show_line_numbers)
        else:
            print("错误: JSON格式不支持，需要是包含role和content的数组")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"处理文件时出错: {e}")


def display_roles_from_string(json_str: str, show_line_numbers: bool = False):
    """
    从字符串解析JSON并显示角色内容
    
    Args:
        json_str: JSON字符串
        show_line_numbers: 是否显示行号
    """
    try:
        data = json.loads(json_str)
        
        if isinstance(data, list):
            display_roles(data, show_line_numbers)
        elif isinstance(data, dict) and 'messages' in data:
            display_roles(data['messages'], show_line_numbers)
        else:
            print("错误: JSON格式不支持，需要是包含role和content的数组")
            
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"处理时出错: {e}")


def interactive_mode():
    """交互模式"""
    print("角色内容显示工具 - 交互模式")
    print("请输入JSON字符串 (输入 'quit' 退出):")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input.strip():
                continue
                
            display_roles_from_string(user_input)
            
        except KeyboardInterrupt:
            print("\n\n退出程序")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='角色内容显示工具')
    parser.add_argument('input', default='/Users/yang/Desktop/S-Bench/input_data.json', nargs='?', help='输入文件路径或JSON字符串')
    parser.add_argument('-n', '--line-numbers', action='store_true', help='显示行号')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    if not args.input:
        # 如果没有输入参数，尝试从标准输入读取
        try:
            input_data = sys.stdin.read()
            if input_data.strip():
                display_roles_from_string(input_data, args.line_numbers)
            else:
                print("请提供JSON数据")
        except KeyboardInterrupt:
            print("\n程序被中断")
    else:
        # 检查输入是否为文件路径
        import os
        if os.path.isfile(args.input):
            display_roles_from_file(args.input, args.line_numbers)
        else:
            # 作为JSON字符串处理
            display_roles_from_string(args.input, args.line_numbers)


if __name__ == "__main__":
    main()
