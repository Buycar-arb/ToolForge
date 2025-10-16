# ToolForge 快速开始指南

5分钟快速上手 ToolForge！

## 🚀 最快的开始方式

### 1. 安装 (1分钟)

```bash
cd ToolForge
pip install -r gradio_webui/requirements.txt
```

### 2. 配置API密钥 (2分钟)

**选择一种方式：**

**方式A - 环境变量（推荐）：**
```bash
export API_KEYS="your-key-1,your-key-2"
export API_BASE_URL="https://api.openai.com/v1"
```

**方式B - 配置文件：**
```bash
# 编辑这个文件
nano stage_2_generate/config/api_keys.py

# 添加你的API密钥
API_KEYS = [
    "your-api-key-1",
    "your-api-key-2",
]
```

### 3. 启动界面 (30秒)

```bash
cd gradio_webui
python quick_fast.py
```

打开浏览器访问: `http://localhost:7860`

### 4. 开始使用 (1分钟)

1. 选择一个功能标签页
2. 配置参数
3. 点击开始按钮
4. 查看结果

## 🎯 三个主要功能

### 功能一：工具变体生成
输入一个工具定义 → 生成多个语义相似的变体

**用途：** 扩充工具库，提升数据多样性

### 功能二：工具标注
输入问题数据 → 自动标注合适的工具和执行路径

**用途：** 数据预处理，准备生成训练数据

### 功能三：数据生成与验证
输入标注数据 → 生成对话 → 自动验证质量

**用途：** 生成高质量的训练数据

## 📝 第一次运行示例

### 示例1：生成工具变体

1. 进入"工具变体生成"标签
2. 使用默认的示例工具JSON
3. 设置输出路径：`./output/tool_variants.jsonl`
4. 目标数量：`10`
5. 点击"开始生成"
6. 在文件查看器中查看结果

### 示例2：标注数据

1. 准备一个JSONL格式的输入文件
2. 进入"工具标注"标签
3. 设置输入文件路径
4. 设置输出文件路径
5. 处理行数：`10`（测试用）
6. 并发数：`2`（测试用）
7. 点击"开始处理"
8. 观察实时LLM输出

## 💡 快速提示

### API密钥
- 支持多个API密钥自动轮换
- 建议至少配置3个密钥以提高处理速度
- 密钥会自动循环使用，避免单个密钥限流

### 文件格式
- 输入：JSONL格式（每行一个JSON对象）
- 输出：JSONL格式（保持与输入相同）
- 编码：UTF-8

### 并发设置
- 开始时使用较小的并发数（2-5）
- 根据API限制和性能逐步调整
- 建议并发数：10-20

### 错误处理
- 系统自动重试失败的请求（最多5次）
- 处理错误的数据会被标记但不会中断流程
- 检查输出文件中的 `processing_error` 字段

## 🔧 常见问题快速解决

### Q: "No API keys configured" 错误
**A:** 设置环境变量或编辑配置文件添加API密钥

### Q: API调用一直失败
**A:** 检查：
1. API密钥是否有效
2. BASE_URL是否正确
3. 是否有足够的API额度

### Q: 端口被占用
**A:** 修改 `gradio_webui/quick_fast.py` 中的端口号：
```python
demo.launch(server_port=7861)  # 改成其他端口
```

### Q: 处理速度很慢
**A:** 
1. 增加并发数
2. 添加更多API密钥
3. 检查网络连接

## 📚 下一步

- 阅读 [完整文档](README.md)
- 查看 [详细设置指南](SETUP_GUIDE.md)
- 了解 [Gradio界面功能](gradio_webui/README.en.md)
- 学习 [添加自定义功能](gradio_webui/HOW_TO_ADD_FEATURE.md)

## 🎓 学习资源

1. **基础使用**
   - README.md - 项目概述
   - SETUP_GUIDE.md - 详细配置

2. **高级功能**
   - ARCHITECTURE.md - 系统架构
   - CONTRIBUTING.md - 贡献代码

3. **问题排查**
   - 查看控制台错误信息
   - 检查日志文件
   - 搜索GitHub Issues

## ✅ 检查清单

上手前确认：
- [ ] Python 3.8+ 已安装
- [ ] 依赖已安装
- [ ] API密钥已配置
- [ ] 网络连接正常
- [ ] 有测试数据（可选）

开始使用！
- [ ] 启动Gradio界面
- [ ] 选择一个功能
- [ ] 配置基本参数
- [ ] 运行测试任务
- [ ] 查看结果

---

**需要帮助？** 查看 [完整文档](README.md) 或提交 [Issue](https://github.com/yourusername/ToolForge/issues)

祝使用愉快！🎉

