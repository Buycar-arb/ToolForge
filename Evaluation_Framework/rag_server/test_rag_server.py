#!/usr/bin/env python3
"""
RAG Server 测试脚本
用于测试 RAG 服务器是否成功启动并正常工作
"""

import sys
import time
import json
import requests
import argparse
from typing import List, Dict, Any


def test_server_health(base_url: str) -> bool:
    """
    测试服务器是否正常响应
    """
    try:
        # 尝试访问根路径
        response = requests.get(base_url, timeout=5)
        print(f"✓ 服务器响应状态码: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"✗ 无法连接到服务器 {base_url}")
        print("  请确认服务器已启动")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ 连接超时")
        return False
    except Exception as e:
        print(f"✗ 连接错误: {e}")
        return False


def test_retrieve_endpoint(base_url: str, queries: List[str], topk: int = 3) -> bool:
    """
    测试 /retrieve 端点
    """
    endpoint = f"{base_url}/retrieve"
    
    # 构建请求数据
    request_data = {
        "queries": queries,
        "topk": topk,
        "return_scores": True
    }
    
    print(f"\n测试检索端点: {endpoint}")
    print(f"查询内容: {queries}")
    print(f"Top-K: {topk}")
    
    try:
        start_time = time.time()
        response = requests.post(
            endpoint,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✓ 检索成功 (耗时: {elapsed_time:.2f}秒)")
            
            # 解析响应
            result = response.json()
            
            # 验证响应格式
            if "result" in result:
                results_list = result["result"]
                print(f"✓ 返回了 {len(results_list)} 个查询的结果")
                
                # 显示第一个查询的结果
                if results_list and len(results_list) > 0:
                    first_query_results = results_list[0]
                    print(f"\n第一个查询的检索结果 (共 {len(first_query_results)} 条):")
                    
                    for i, item in enumerate(first_query_results[:2], 1):  # 只显示前2条
                        if isinstance(item, dict) and "document" in item:
                            doc = item["document"]
                            score = item.get("score", "N/A")
                            
                            # 截取文本预览
                            text_preview = doc.get("text", doc.get("contents", ""))[:100]
                            if len(text_preview) == 100:
                                text_preview += "..."
                            
                            print(f"\n  结果 {i}:")
                            print(f"    标题: {doc.get('title', 'N/A')}")
                            print(f"    分数: {score}")
                            print(f"    文本预览: {text_preview}")
                
                return True
            else:
                print(f"✗ 响应格式错误: 缺少 'result' 字段")
                print(f"  实际响应: {response.text[:200]}")
                return False
                
        else:
            print(f"✗ 请求失败，状态码: {response.status_code}")
            print(f"  响应内容: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"✗ 请求超时 (超过30秒)")
        print("  可能原因: 模型或索引加载缓慢，请稍后重试")
        return False
    except requests.exceptions.ConnectionError:
        print(f"✗ 无法连接到端点 {endpoint}")
        print("  请确认服务器已完全启动")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ 无法解析响应JSON: {e}")
        return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False


def test_batch_retrieve(base_url: str) -> bool:
    """
    测试批量查询功能
    """
    endpoint = f"{base_url}/retrieve"
    
    # 多个查询
    queries = [
        "What is machine learning?",
        "How does neural network work?",
        "What is deep learning?"
    ]
    
    request_data = {
        "queries": queries,
        "topk": 2,
        "return_scores": False
    }
    
    print(f"\n测试批量查询...")
    print(f"查询数量: {len(queries)}")
    
    try:
        response = requests.post(
            endpoint,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result and len(result["result"]) == len(queries):
                print(f"✓ 批量查询成功，返回 {len(result['result'])} 组结果")
                return True
            else:
                print(f"✗ 返回结果数量不匹配")
                return False
        else:
            print(f"✗ 批量查询失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ 批量查询出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试 RAG 服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=5003, help="服务器端口")
    parser.add_argument("--query", type=str, default="What is Python?", help="测试查询")
    parser.add_argument("--topk", type=int, default=3, help="返回结果数量")
    parser.add_argument("--full", action="store_true", help="运行完整测试套件")
    
    args = parser.parse_args()
    
    # 构建基础URL
    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 60)
    print("RAG 服务器测试工具")
    print("=" * 60)
    print(f"目标服务器: {base_url}")
    print()
    
    # 测试计数器
    total_tests = 0
    passed_tests = 0
    
    # 1. 测试服务器健康状态
    print("1. 检查服务器连接...")
    total_tests += 1
    if test_server_health(base_url):
        passed_tests += 1
        
        # 2. 测试检索端点
        print("\n2. 测试检索功能...")
        total_tests += 1
        if test_retrieve_endpoint(base_url, [args.query], args.topk):
            passed_tests += 1
        
        # 3. 完整测试套件
        if args.full:
            print("\n3. 运行完整测试套件...")
            
            # 测试批量查询
            print("\n3.1 测试批量查询...")
            total_tests += 1
            if test_batch_retrieve(base_url):
                passed_tests += 1
            
            # 测试不同的topk值
            print("\n3.2 测试不同的 Top-K 值...")
            for k in [1, 5, 10]:
                total_tests += 1
                print(f"\n  测试 topk={k}")
                if test_retrieve_endpoint(base_url, ["test query"], k):
                    passed_tests += 1
    
    # 测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"总测试数: {total_tests}")
    print(f"通过数: {passed_tests}")
    print(f"失败数: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n✅ 所有测试通过！RAG 服务器运行正常。")
        return 0
    else:
        print(f"\n⚠️  部分测试失败 ({total_tests - passed_tests}/{total_tests})")
        return 1


if __name__ == "__main__":
    sys.exit(main())