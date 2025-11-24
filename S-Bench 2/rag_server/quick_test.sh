#!/bin/bash

# RAG 服务器快速测试脚本

echo "======================================"
echo "RAG 服务器快速测试"
echo "======================================"

# 默认配置
HOST="0.0.0.0"
PORT=5003
BASE_URL="http://${HOST}:${PORT}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n目标服务器: ${YELLOW}${BASE_URL}${NC}"

# 1. 检查服务器是否在运行
echo -e "\n1. 检查服务器进程..."
if pgrep -f "retrieval_server.py" > /dev/null; then
    echo -e "${GREEN}✓${NC} 找到 retrieval_server.py 进程"
    PID=$(pgrep -f "retrieval_server.py")
    echo "  进程 ID: $PID"
else
    echo -e "${RED}✗${NC} 未找到 retrieval_server.py 进程"
    echo "  请先启动服务器: bash rag_server/launch.sh"
    exit 1
fi

# 2. 检查端口是否监听
echo -e "\n2. 检查端口 ${PORT}..."
if netstat -an | grep -q ":${PORT}.*LISTEN" || lsof -i:${PORT} > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} 端口 ${PORT} 正在监听"
else
    echo -e "${YELLOW}⚠${NC} 端口 ${PORT} 未监听，服务器可能还在启动中..."
fi

# 3. 测试 HTTP 连接
echo -e "\n3. 测试 HTTP 连接..."
if curl -s -o /dev/null -w "%{http_code}" ${BASE_URL} > /dev/null 2>&1; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${BASE_URL})
    echo -e "${GREEN}✓${NC} 服务器响应，状态码: $HTTP_CODE"
else
    echo -e "${RED}✗${NC} 无法连接到服务器"
    echo "  服务器可能还在加载模型，请稍后再试"
    exit 1
fi

# 4. 测试检索功能
echo -e "\n4. 测试检索端点..."
echo "  发送测试查询: 'What is artificial intelligence?'"

# 构建 JSON 请求
REQUEST_JSON='{
    "queries": ["What is artificial intelligence?"],
    "topk": 3,
    "return_scores": true
}'

# 发送请求并保存响应
RESPONSE=$(curl -s -X POST ${BASE_URL}/retrieve \
    -H "Content-Type: application/json" \
    -d "$REQUEST_JSON" 2>/dev/null)

if [ $? -eq 0 ] && [ ! -z "$RESPONSE" ]; then
    # 检查响应是否包含 result 字段
    if echo "$RESPONSE" | grep -q '"result"'; then
        echo -e "${GREEN}✓${NC} 检索成功！"
        
        # 提取并显示第一个结果的标题（如果存在）
        FIRST_TITLE=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'result' in data and data['result'] and data['result'][0]:
        first_result = data['result'][0][0]
        if 'document' in first_result:
            title = first_result['document'].get('title', 'No title')
            print(title[:80])
except:
    pass
" 2>/dev/null)
        
        if [ ! -z "$FIRST_TITLE" ]; then
            echo "  第一个结果标题: $FIRST_TITLE"
        fi
    else
        echo -e "${RED}✗${NC} 响应格式错误"
        echo "  响应预览: ${RESPONSE:0:100}"
    fi
else
    echo -e "${RED}✗${NC} 检索请求失败"
fi

# 5. 显示日志位置提示
echo -e "\n5. 日志信息"
echo "  如需查看详细日志，请查看服务器输出"
echo "  或使用: tail -f nohup.out (如果使用 nohup 启动)"

# 6. 总结
echo -e "\n======================================"
if curl -s -o /dev/null -w "%{http_code}" ${BASE_URL}/retrieve -X POST \
    -H "Content-Type: application/json" \
    -d "$REQUEST_JSON" | grep -q "200"; then
    echo -e "${GREEN}✅ RAG 服务器运行正常！${NC}"
    echo -e "\n使用 Python 测试脚本进行更详细的测试:"
    echo "  python3 rag_server/test_rag_server.py"
    echo "  python3 rag_server/test_rag_server.py --full  # 完整测试"
else
    echo -e "${YELLOW}⚠️  服务器可能还在初始化${NC}"
    echo "  请等待几秒后重试"
fi
echo "======================================"