import gradio as gr
import json
import sys
import os
import asyncio
from datetime import datetime
from pathlib import Path
from collections import deque

# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to Python search path
sys.path.append(os.path.dirname(current_dir))

from stage_1_label.code.llm_generate_label import LLMGenerateLabel, process_single_line
from feature_generate_judge import FeatureGenerateJudgeProcessor
from feature_tool_variant_generator import ToolVariantGenerator
from feature_tool_list_manager import tool_list_manager

class GradioLabelGenerator:
    def __init__(self):
        self.is_running = False
        self.current_task = None
        self.llm_outputs = deque(maxlen=100)  # Save latest 100 LLM outputs
        self.current_input_file = ""
        self.current_output_file = ""
    
    async def process_data(
        self,
        input_file: str,
        output_file: str,
        residue_file: str,
        max_lines: int,
        concurrency: int,
        model: str,
        temperature: float,
        max_tokens: int,
        progress=gr.Progress()
    ):
        """Main data processing function"""
        try:
            self.is_running = True
            self.llm_outputs.clear()
            self.current_input_file = input_file
            self.current_output_file = output_file
            
            # Validate input file
            if not os.path.exists(input_file):
                return f"❌ Error: Input file does not exist - {input_file}", ""
            
            # Create output directory
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            if residue_file:
                os.makedirs(os.path.dirname(residue_file) if os.path.dirname(residue_file) else ".", exist_ok=True)
            
            # Initialize generator
            generator_label = LLMGenerateLabel(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Read data
            processed_data = []
            residue_data = []
            generate_count = 1
            
            progress(0, desc=f"📚 Reading data...")
            
            with open(input_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        line_data = json.loads(line)
                        
                        if generate_count <= max_lines:
                            processed_data.append((line_data, generate_count))
                        else:
                            residue_data.append(line_data)
                        
                        generate_count += 1
                    except json.JSONDecodeError as e:
                        print(f"Line {generate_count} JSON parsing failed: {e}")
                        generate_count += 1
                        continue
            
            total_count = generate_count - 1
            process_count = len(processed_data)
            residue_count = len(residue_data)
            
            status_msg = f"""📊 Data reading completed:
   - Total data: {total_count} items
   - For processing: {process_count} items
   - Remaining data: {residue_count} items
   - Concurrency: {concurrency}
"""
            print(status_msg)
            
            # Save remaining data
            if residue_data and residue_file:
                progress(0.1, desc=f"💾 Saving remaining data...")
                with open(residue_file, "w", encoding='utf-8') as rf:
                    for residue_item in residue_data:
                        rf.write(json.dumps(residue_item, ensure_ascii=False) + '\n')
                print(f"✅ Remaining {len(residue_data)} data items saved to: {residue_file}")
            
            if not processed_data:
                return "⚠️  No data to process, program ends", ""
            
            # Define content callback function
            async def content_callback(line_num, content, question):
                # Save to deque
                self.llm_outputs.append({
                    "line": line_num,
                    "question": question,
                    "content": content,
                    "time": datetime.now().strftime('%H:%M:%S')
                })
            
            # Create semaphore to control concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def process_with_semaphore(data_item):
                async with semaphore:
                    line_data, count = data_item
                    return await process_single_line(generator_label, line_data, count, content_callback)
            
            # Concurrent data processing
            progress(0.2, desc=f"🚀 Starting concurrent processing of {process_count} data items...")
            
            tasks = [process_with_semaphore(data_item) for data_item in processed_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Save results
            progress(0.8, desc=f"💾 Saving processing results...")
            
            success_count = 0
            error_count = 0
            
            with open(output_file, "w", encoding='utf-8') as w:
                for i, result in enumerate(results):
                    original_count = processed_data[i][1]
                    
                    if isinstance(result, Exception):
                        print(f"❌ Data item {original_count} processing exception: {result}")
                        line_data = processed_data[i][0]
                        line_data["processing_error"] = f"Processing exception: {str(result)}"
                        line_data["reasoning"] = "Processing exception"
                        line_data["tool_select"] = "Processing exception"
                        line_data["route_select"] = "Processing exception"
                        processed_line = line_data
                        error_count += 1
                    else:
                        processed_line = result
                        success_count += 1
                    
                    w.write(json.dumps(processed_line, ensure_ascii=False) + '\n')
                    
                    # Update progress
                    if (i + 1) % 10 == 0:
                        progress_percent = 0.8 + 0.2 * (i + 1) / len(results)
                        progress(progress_percent, desc=f"💾 Saved {i + 1}/{len(results)} items")
            
            progress(1.0, desc="✅ Processing completed!")
            
            final_msg = f"""
✅ Processing completed!

📊 Statistics:
   - Processed data: {process_count} items
   - Success: {success_count} items
   - Failed: {error_count} items
   - Remaining data: {residue_count} items
   - Concurrency: {concurrency}

📁 Output files:
   - Processing results: {output_file}
   - Remaining data: {residue_file if residue_file else 'Not set'}

⏰ Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Generate LLM output summary
            llm_summary = self._generate_llm_summary()
            
            return final_msg, llm_summary
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error occurred during processing:\n{str(e)}\n\nDetailed information:\n{traceback.format_exc()}"
            return error_msg, ""
        finally:
            self.is_running = False
    
    def _generate_llm_summary(self):
        """Generate LLM output summary"""
        if not self.llm_outputs:
            return "No LLM output records available"
        
        summary = f"📝 Total {len(self.llm_outputs)} LLM outputs recorded\n\n"
        summary += "="*80 + "\n\n"
        
        for output in self.llm_outputs:
            summary += f"🔹 Item {output['line']} | Time: {output['time']}\n"
            summary += f"❓ Question: {output['question'][:100]}...\n" if len(output['question']) > 100 else f"❓ Question: {output['question']}\n"
            summary += f"💬 LLM Response:\n{output['content']}\n"
            summary += "="*80 + "\n\n"
        
        return summary
    
    def run_process(
        self,
        input_file,
        output_file,
        residue_file,
        max_lines,
        concurrency,
        model,
        temperature,
        max_tokens
    ):
        """Synchronous wrapper"""
        return asyncio.run(self.process_data(
            input_file,
            output_file,
            residue_file,
            max_lines,
            concurrency,
            model,
            temperature,
            max_tokens
        ))
    
    def load_jsonl_file(self, file_path):
        """Load JSONL file"""
        if not file_path or not os.path.exists(file_path):
            return []
        
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    try:
                        data.append((i, json.loads(line)))
                    except json.JSONDecodeError:
                        data.append((i, {"error": "JSON parsing failed"}))
        except Exception as e:
            return [(0, {"error": f"File reading failed: {str(e)}"})]
        
        return data
    
    def get_line_content(self, file_path, line_number):
        """Get content of specified line"""
        data = self.load_jsonl_file(file_path)
        
        if not data:
            return "File is empty or does not exist"
        
        if line_number < 1 or line_number > len(data):
            return f"Line number out of range (1-{len(data)})"
        
        line_num, content = data[line_number - 1]
        return json.dumps(content, ensure_ascii=False, indent=2)
    
    def get_file_info(self, file_path):
        """Get file information"""
        if not file_path or not os.path.exists(file_path):
            return "File does not exist", 0
        
        data = self.load_jsonl_file(file_path)
        total_lines = len(data)
        
        info = f"""
📄 File path: {file_path}
📊 Total lines: {total_lines}
📅 Modified time: {datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')}
💾 File size: {os.path.getsize(file_path) / 1024:.2f} KB
"""
        return info, total_lines


# Create global instances
generator_feature1 = ToolVariantGenerator()
generator_feature2 = GradioLabelGenerator()
generator_gen_judge = FeatureGenerateJudgeProcessor()

# ==================== Folder and file path processing functions ====================
def scan_folder_for_files(folder_path, extension=".jsonl"):
    """
    Scan folder and return file list with specified extension
    
    Args:
        folder_path: Folder path
        extension: File extension (default .jsonl)
    
    Returns:
        list: File name list
    """
    if not folder_path or not os.path.exists(folder_path):
        return []
    
    try:
        files = []
        for filename in os.listdir(folder_path):
            if filename.endswith(extension):
                files.append(filename)
        return sorted(files)
    except Exception as e:
        print(f"Folder scanning failed: {e}")
        return []

def get_full_path(folder_path, filename):
    """
    Concatenate complete file path
    
    Args:
        folder_path: Folder path
        filename: File name
    
    Returns:
        str: Complete path
    """
    if not folder_path or not filename:
        return ""
    return os.path.join(folder_path, filename)

# ==================== File viewer related functions ====================
def view_file_content(processor, file_type, custom_path, line_number):
    """View file content - general function"""
    # Determine file path to view
    if file_type == "Input file":
        file_path = processor.current_input_file
    elif file_type == "Output file":
        file_path = processor.current_output_file
    else:  # Custom path
        file_path = custom_path
    
    if not file_path:
        return "Please run processing task first or enter custom file path", ""
    
    # Get file information
    info, total_lines = processor.get_file_info(file_path)
    
    # Get specified line content
    if line_number and line_number > 0:
        content = processor.get_line_content(file_path, int(line_number))
    else:
        content = "Please enter line number to view"
    
    return info, content

def update_line_slider(processor, file_type, custom_path):
    """Update line number slider maximum value - general function"""
    if file_type == "Input file":
        file_path = processor.current_input_file
    elif file_type == "Output file":
        file_path = processor.current_output_file
    else:
        file_path = custom_path
    
    if not file_path or not os.path.exists(file_path):
        return gr.update(maximum=1, value=1)
    
    _, total_lines = processor.get_file_info(file_path)
    return gr.update(maximum=max(1, total_lines), value=1)

# ==================== UI component building functions ====================
def create_file_viewer(processor, tab_name):
    """Create file viewer component - reusable"""
    gr.Markdown(f"""
    ### 📖 {tab_name} - File Content Viewer
    
    View each data item in input or output files.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_type = gr.Radio(
                label="Select file type",
                choices=["Input file", "Output file", "Custom path"],
                value="Input file"
            )
            
            custom_file_path = gr.Textbox(
                label="Custom file path (used when selecting custom path)",
                placeholder="/path/to/your/file.jsonl",
                visible=False
            )
            
            line_number = gr.Slider(
                label="Select line number",
                minimum=1,
                maximum=100,
                value=1,
                step=1,
                interactive=True
            )
            
            view_btn = gr.Button("🔍 View Content", variant="primary")
        
        with gr.Column(scale=2):
            file_info = gr.Textbox(
                label="File information",
                lines=5,
                interactive=False
            )
            
            line_content = gr.Textbox(
                label="Line content (JSON format)",
                lines=20,
                interactive=False
            )
    
    # Bind events
    def on_file_type_change(file_type):
        if file_type == "Custom path":
            return gr.update(visible=True), gr.update(maximum=1, value=1)
        else:
            slider_update = update_line_slider(processor, file_type, "")
            return gr.update(visible=False), slider_update
    
    file_type.change(
        fn=on_file_type_change,
        inputs=[file_type],
        outputs=[custom_file_path, line_number]
    )
    
    custom_file_path.change(
        fn=lambda ft, cp: update_line_slider(processor, ft, cp),
        inputs=[file_type, custom_file_path],
        outputs=[line_number]
    )
    
    view_btn.click(
        fn=lambda ft, cp, ln: view_file_content(processor, ft, cp, ln),
        inputs=[file_type, custom_file_path, line_number],
        outputs=[file_info, line_content]
    )
    
    line_number.change(
        fn=lambda ft, cp, ln: view_file_content(processor, ft, cp, ln),
        inputs=[file_type, custom_file_path, line_number],
        outputs=[file_info, line_content]
    )
    
    return file_type, custom_file_path, line_number, view_btn, file_info, line_content

# ==================== Main Interface ====================
def create_ui():
    with gr.Blocks(title="LLM Data Labeling Tool", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏷️ LLM Data Processing Tool Suite
        
        Multi-functional data processing platform supporting various data processing tasks and real-time monitoring.
        """)
        
        # ========== Top-level Tabs: Distinguish different functions ==========
        with gr.Tabs() as main_tabs:
            
            
            
            # ==================== 功能一：工具变体生成 ====================
            with gr.Tab("🔧 功能一：工具变体生成"):
                with gr.Tabs():
                    # 功能一 - 工具变体生成
                    with gr.Tab("📝 工具变体生成"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### 📁 工具配置")
                                
                                f2_original_tool = gr.Code(
                                    label="原始工具JSON",
                                    language="json",
                                    value="""{
  "name": "example_search",
  "description": "示例搜索工具，用于查找相关信息",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "搜索查询内容"
      },
      "category": {
        "type": "string",
        "description": "搜索类别"
      }
    },
    "required": ["query"]
  }
}""",
                                    lines=15
                                )
                                
                                f2_output_file = gr.Textbox(
                                    label="输出文件路径",
                                    placeholder="/path/to/output.jsonl",
                                    value="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/chenhao/ToolForge_github/code_perfect/data/original_data/tool_variants.jsonl"
                                )
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### ⚙️ 生成参数配置")
                                
                                f2_target_count = gr.Number(
                                    label="目标生成数量",
                                    value=20,
                                    precision=0,
                                    minimum=1,
                                    maximum=100
                                )
                                
                                f2_cos_th = gr.Slider(
                                    label="余弦相似度阈值",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.05
                                )
                                
                                f2_bm25_th = gr.Slider(
                                    label="BM25相似度阈值",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.6,
                                    step=0.05
                                )
                                
                                f2_model = gr.Dropdown(
                                    label="模型选择",
                                    choices=[
                                        "anthropic.claude-sonnet-4",
                                        "anthropic.claude-3-5-sonnet",
                                        "anthropic.claude-3-opus",
                                        "gpt-4.1"
                                    ],
                                    value="gpt-4.1"
                                )
                                
                                f2_temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1
                                )
                                
                                f2_max_tokens = gr.Number(
                                    label="Max Tokens",
                                    value=8192,
                                    precision=0,
                                    minimum=1024,
                                    maximum=16384
                                )
                        
                        with gr.Row():
                            f2_start_btn = gr.Button("🚀 开始生成工具变体", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                f2_output_status = gr.Textbox(
                                    label="生成状态",
                                    placeholder="等待开始...",
                                    lines=15,
                                    max_lines=20,
                                    interactive=False
                                )
                            
                            with gr.Column(scale=1):
                                f2_generation_log = gr.Textbox(
                                    label="📋 生成日志",
                                    placeholder="生成日志将在这里显示...",
                                    lines=15,
                                    max_lines=20,
                                    interactive=False
                                )
                        
                        # 绑定功能一的处理事件
                        f2_start_btn.click(
                            fn=generator_feature1.process_data_tool_variant,
                            inputs=[
                                f2_original_tool,
                                f2_output_file,
                                f2_target_count,
                                f2_cos_th,
                                f2_bm25_th,
                                f2_model,
                                f2_temperature,
                                f2_max_tokens
                            ],
                            outputs=[f2_output_status, f2_generation_log]
                        )
                        
                        gr.Markdown("""
                        ---
                        ### 📖 功能说明
                        
                        **工具变体生成功能：**
                        
                        1. **输入原始工具**：在左侧JSON编辑器中输入要生成变体的原始工具定义
                        2. **配置生成参数**：
                           - 目标生成数量：要生成多少个工具变体
                           - 余弦相似度阈值：控制语义相似度（越高越相似）
                           - BM25相似度阈值：控制关键词相似度（越低越相似）
                           - 模型和生成参数：控制LLM生成质量
                        3. **自动生成**：系统会自动生成符合要求的工具变体
                        4. **相似度检查**：使用向量相似度和BM25算法确保变体质量
                        
                        **输出格式：**
                        - 每个工具变体保存为一行JSON
                        - 保持原始工具的核心功能不变
                        - 在名称、描述、参数等方面进行同义替换
                        """)
                    
                    # 功能一 - 文件查看器
                    with gr.Tab("📂 文件查看器"):
                        create_file_viewer(generator_feature1, "工具变体生成")
                        
                        gr.Markdown("""
                        ---
                        ### 📖 文件查看器说明
                        
                        - **输出文件**：查看生成的工具变体文件
                        - **自定义路径**：查看任意工具变体文件
                        - 使用滑块快速浏览不同的工具变体
                        - JSON格式化显示，便于查看工具结构
                        """)
                    
                    # 功能一 - 工具列表管理
                    with gr.Tab("🔧 工具列表管理"):
                        gr.Markdown("""
                        ### 🔧 TOOL_LIST 管理
                        
                        管理 `stage_1_label/code/tool_prompts.py` 中的 TOOL_LIST 配置。
                        从工具库中选择需要的工具，保存后会自动更新配置文件。
                        """)
                        
                        with gr.Row():
                            # 左侧：可用工具列表
                            with gr.Column(scale=2):
                                gr.Markdown("### 📦 可用工具库")
                                
                                tool_refresh_btn = gr.Button("🔄 刷新工具列表", size="sm")
                                
                                tool_statistics = gr.Textbox(
                                    label="统计信息",
                                    lines=8,
                                    interactive=False
                                )
                                
                                available_tools_checkbox = gr.CheckboxGroup(
                                    label="选择要添加的工具（可多选）",
                                    choices=[],
                                    value=[],
                                    interactive=True
                                )
                                
                                gr.Markdown("""
                                💡 **提示**：
                                - 勾选工具后，点击"添加到TOOL_LIST"按钮
                                - 带有 ✓ 标记的工具已在当前TOOL_LIST中
                                """)
                            
                            # 中间：操作按钮
                            with gr.Column(scale=1):
                                gr.Markdown("### 🔄 操作")
                                
                                add_tools_btn = gr.Button(
                                    "➡️ 添加到TOOL_LIST",
                                    variant="primary",
                                    size="lg"
                                )
                                
                                gr.Markdown("---")
                                
                                remove_tools_btn = gr.Button(
                                    "⬅️ 从TOOL_LIST移除",
                                    variant="secondary",
                                    size="lg"
                                )
                                
                                gr.Markdown("---")
                                
                                save_tool_list_btn = gr.Button(
                                    "💾 保存到tool_prompts.py",
                                    variant="primary",
                                    size="lg"
                                )
                            
                            # 右侧：当前TOOL_LIST
                            with gr.Column(scale=2):
                                gr.Markdown("### 📋 当前 TOOL_LIST")
                                
                                current_tools_checkbox = gr.CheckboxGroup(
                                    label="选择要移除的工具（可多选）",
                                    choices=[],
                                    value=[],
                                    interactive=True
                                )
                                
                                current_tools_display = gr.Textbox(
                                    label="工具详情",
                                    lines=15,
                                    interactive=False,
                                    placeholder="当前TOOL_LIST中的工具将在这里显示..."
                                )
                        
                        # 状态提示
                        with gr.Row():
                            operation_status = gr.Textbox(
                                label="操作状态",
                                placeholder="等待操作...",
                                lines=3,
                                interactive=False
                            )
                        
                        # 定义回调函数
                        def refresh_tools():
                            """刷新工具列表"""
                            # 扫描工具库
                            tool_list_manager.scan_tool_bank()
                            # 加载当前TOOL_LIST
                            tool_list_manager.load_current_tool_list()
                            
                            # 获取可用工具选项
                            available_choices = []
                            for tool_name in sorted(tool_list_manager.available_tools.keys()):
                                if tool_name in tool_list_manager.current_tool_list:
                                    available_choices.append(f"{tool_name} ✓")
                                else:
                                    available_choices.append(tool_name)
                            
                            # 获取当前TOOL_LIST选项
                            current_choices = tool_list_manager.current_tool_list.copy()
                            
                            # 获取统计信息
                            stats = tool_list_manager.get_statistics()
                            
                            # 获取当前工具详情
                            current_display = tool_list_manager.get_tool_info_text(
                                tool_list_manager.current_tool_list
                            )
                            
                            return (
                                gr.update(choices=available_choices, value=[]),
                                gr.update(choices=current_choices, value=[]),
                                current_display,
                                stats,
                                "✅ 工具列表已刷新"
                            )
                        
                        def add_tools(selected_tools):
                            """添加工具到TOOL_LIST"""
                            if not selected_tools:
                                return (
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    "⚠️ 请先选择要添加的工具"
                                )
                            
                            # 清理工具名称（去掉 ✓ 标记）
                            clean_tools = []
                            for tool in selected_tools:
                                clean_name = tool.replace(" ✓", "").strip()
                                clean_tools.append(clean_name)
                            
                            # 添加到当前列表（去重）
                            for tool in clean_tools:
                                if tool not in tool_list_manager.current_tool_list:
                                    tool_list_manager.current_tool_list.append(tool)
                            
                            # 更新显示
                            current_choices = tool_list_manager.current_tool_list.copy()
                            current_display = tool_list_manager.get_tool_info_text(
                                tool_list_manager.current_tool_list
                            )
                            
                            # 更新可用工具列表（添加 ✓ 标记）
                            available_choices = []
                            for tool_name in sorted(tool_list_manager.available_tools.keys()):
                                if tool_name in tool_list_manager.current_tool_list:
                                    available_choices.append(f"{tool_name} ✓")
                                else:
                                    available_choices.append(tool_name)
                            
                            return (
                                gr.update(choices=current_choices, value=[]),
                                current_display,
                                gr.update(choices=available_choices, value=[]),
                                f"✅ 已添加 {len(clean_tools)} 个工具（未保存，请点击'保存'按钮）"
                            )
                        
                        def remove_tools(selected_tools):
                            """从TOOL_LIST移除工具"""
                            if not selected_tools:
                                return (
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    "⚠️ 请先选择要移除的工具"
                                )
                            
                            # 从当前列表移除
                            for tool in selected_tools:
                                if tool in tool_list_manager.current_tool_list:
                                    tool_list_manager.current_tool_list.remove(tool)
                            
                            # 更新显示
                            current_choices = tool_list_manager.current_tool_list.copy()
                            current_display = tool_list_manager.get_tool_info_text(
                                tool_list_manager.current_tool_list
                            )
                            
                            # 更新可用工具列表（移除 ✓ 标记）
                            available_choices = []
                            for tool_name in sorted(tool_list_manager.available_tools.keys()):
                                if tool_name in tool_list_manager.current_tool_list:
                                    available_choices.append(f"{tool_name} ✓")
                                else:
                                    available_choices.append(tool_name)
                            
                            return (
                                gr.update(choices=current_choices, value=[]),
                                current_display,
                                gr.update(choices=available_choices, value=[]),
                                f"✅ 已移除 {len(selected_tools)} 个工具（未保存，请点击'保存'按钮）"
                            )
                        
                        def save_tool_list():
                            """保存TOOL_LIST到文件"""
                            success, message = tool_list_manager.save_tool_list(
                                tool_list_manager.current_tool_list
                            )
                            
                            if success:
                                return f"{message}\n\n⏰ 保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n💡 配置已更新，可能需要重启应用才能生效"
                            else:
                                return message
                        
                        # 绑定事件
                        tool_refresh_btn.click(
                            fn=refresh_tools,
                            outputs=[
                                available_tools_checkbox,
                                current_tools_checkbox,
                                current_tools_display,
                                tool_statistics,
                                operation_status
                            ]
                        )
                        
                        add_tools_btn.click(
                            fn=add_tools,
                            inputs=[available_tools_checkbox],
                            outputs=[
                                current_tools_checkbox,
                                current_tools_display,
                                available_tools_checkbox,
                                operation_status
                            ]
                        )
                        
                        remove_tools_btn.click(
                            fn=remove_tools,
                            inputs=[current_tools_checkbox],
                            outputs=[
                                current_tools_checkbox,
                                current_tools_display,
                                available_tools_checkbox,
                                operation_status
                            ]
                        )
                        
                        save_tool_list_btn.click(
                            fn=save_tool_list,
                            outputs=[operation_status]
                        )
                        
                        gr.Markdown("""
                        ---
                        ### 📖 使用说明
                        
                        1. **刷新工具列表**：点击"🔄 刷新工具列表"按钮，扫描工具库
                        2. **添加工具**：
                           - 在左侧勾选要添加的工具
                           - 点击"➡️ 添加到TOOL_LIST"按钮
                        3. **移除工具**：
                           - 在右侧勾选要移除的工具
                           - 点击"⬅️ 从TOOL_LIST移除"按钮
                        4. **保存配置**：点击"💾 保存到tool_prompts.py"按钮
                        5. **注意**：保存后可能需要重启应用才能在其他功能中生效
                        
                        **工具库路径**：`stage_2_generate/tool_bank/tools/`  
                        **配置文件路径**：`stage_1_label/code/tool_prompts.py`
                        """)
            
# ==================== 功能二：工具标注 ====================
            with gr.Tab("📝 功能二：工具标注"):
                with gr.Tabs():
                    # 功能二 - 数据处理
                    with gr.Tab("📝 数据处理"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### 📁 文件路径配置")
                                
                                f1_input_file = gr.Textbox(
                                    label="输入文件路径",
                                    placeholder="/path/to/input.jsonl",
                                    value="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/chenhao/ToolForge_github/code_perfect/data/original_data/test_5.jsonl"
                                )
                                
                                f1_output_file = gr.Textbox(
                                    label="输出文件路径",
                                    placeholder="/path/to/output.jsonl",
                                    value="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/chenhao/ToolForge_github/code_perfect/data/original_data/test_1.jsonl"
                                )
                                
                                f1_residue_file = gr.Textbox(
                                    label="剩余数据文件路径（可选）",
                                    placeholder="/path/to/residue.jsonl",
                                    value="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/chenhao/ToolForge_github/code_perfect/data/original_data/test_2.jsonl"
                                )
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### ⚙️ 处理参数配置")
                                
                                f1_max_lines = gr.Number(
                                    label="处理行数 (MAX_LINES)",
                                    value=5000,
                                    precision=0,
                                    minimum=1
                                )
                                
                                f1_concurrency = gr.Number(
                                    label="并发数量",
                                    value=10,
                                    precision=0,
                                    minimum=1,
                                    maximum=50
                                )
                                
                                f1_model = gr.Dropdown(
                                    label="模型选择",
                                    choices=[
                                        "anthropic.claude-sonnet-4",
                                        "anthropic.claude-3-5-sonnet",
                                        "anthropic.claude-3-opus",
                                        "gpt-4.1"
                                    ],
                                    value="gpt-4.1"
                                )
                                
                                f1_temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.0,
                                    step=0.1
                                )
                                
                                f1_max_tokens = gr.Number(
                                    label="Max Tokens",
                                    value=8192,
                                    precision=0,
                                    minimum=1024,
                                    maximum=16384
                                )
                        
                        with gr.Row():
                            f1_start_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                f1_output_status = gr.Textbox(
                                    label="处理状态",
                                    placeholder="等待开始...",
                                    lines=15,
                                    max_lines=20,
                                    interactive=False
                                )
                            
                            with gr.Column(scale=1):
                                f1_llm_output = gr.Textbox(
                                    label="🤖 LLM实时输出",
                                    placeholder="LLM返回的内容将在这里显示...",
                                    lines=15,
                                    max_lines=20,
                                    interactive=False
                                )
                        
                        # 绑定功能二的处理事件
                        f1_start_btn.click(
                            fn=generator_feature2.run_process,
                            inputs=[
                                f1_input_file,
                                f1_output_file,
                                f1_residue_file,
                                f1_max_lines,
                                f1_concurrency,
                                f1_model,
                                f1_temperature,
                                f1_max_tokens
                            ],
                            outputs=[f1_output_status, f1_llm_output]
                        )
                        
                        gr.Markdown("""
                        ---
                        ### 📖 功能说明
                        
                        此功能用于对多跳问题进行工具标注，包括：
                        - 分析问题类型
                        - 选择合适的工具
                        - 确定执行路径（case1/case2/case3/case4）
                        """)
                    
                    # 功能二 - 文件查看器
                    with gr.Tab("📂 文件查看器"):
                        create_file_viewer(generator_feature2, "功能二")
                        
                        gr.Markdown("""
                        ---
                        ### 📖 文件查看器说明
                        
                        - **输入文件**: 查看原始问题数据
                        - **输出文件**: 查看标注后的数据（包含reasoning、tool_select、route_select）
                        - **自定义路径**: 查看任意JSONL文件
                        """)
            
            # ==================== 功能三：数据生成与校验 ====================
            with gr.Tab("🎲 功能三：数据生成与校验"):
                with gr.Tabs():
                    # 功能三 - 数据处理
                    with gr.Tab("📝 数据处理"):
                        gr.Markdown("""
                        ### 🎲 数据生成与校验
                        
                        整合 stage_2_generate 和 stage_3_judge 的功能，自动生成数据并进行校验。
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### 📁 文件路径配置")
                                
                                # 输入文件配置（从功能二自动获取）
                                gen_input_folder = gr.Textbox(
                                    label="输入文件夹路径（自动从功能二获取）",
                                    placeholder="/path/to/folder/",
                                    value=""
                                )
                                
                                with gr.Row():
                                    gen_input_file_dropdown = gr.Dropdown(
                                        label="选择输入文件",
                                        choices=[],
                                        value=None,
                                        interactive=True
                                    )
                                    
                                    gen_refresh_input_btn = gr.Button("🔄 刷新", size="sm", scale=0)
                                
                                gen_input_file = gr.Textbox(
                                    label="完整输入路径",
                                    interactive=False,
                                    placeholder="自动生成..."
                                )
                                
                                gen_system_prompt = gr.Textbox(
                                    label="System Prompt文件路径",
                                    value=os.path.join(os.path.dirname(current_dir), "stage_2_generate", "prompts", "opt_sys_prompt.txt")
                                )
                                
                                gen_user_prompt = gr.Textbox(
                                    label="User Prompt文件路径",
                                    value=os.path.join(os.path.dirname(current_dir), "stage_2_generate", "prompts", "opt_user_prompt.txt")
                                )
                                
                                gen_tool_bank = gr.Textbox(
                                    label="Tool Bank路径",
                                    value=os.path.join(os.path.dirname(current_dir), "stage_2_generate", "tool_bank", "tools")
                                )
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### ⚙️ 处理参数配置")
                                
                                gen_model = gr.Dropdown(
                                    label="模型选择",
                                    choices=[
                                        "anthropic.claude-sonnet-4",
                                        "anthropic.claude-3-5-sonnet",
                                        "anthropic.claude-3-opus",
                                        "gpt-4.1"
                                    ],
                                    value="gpt-4.1"
                                )
                                
                                gen_temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.0,
                                    step=0.1
                                )
                                
                                gen_max_tokens = gr.Number(
                                    label="Max Tokens",
                                    value=8192,
                                    precision=0,
                                    minimum=1024,
                                    maximum=16384
                                )
                                
                                gen_virtual_min = gr.Number(
                                    label="虚拟工具最小数量",
                                    value=3,
                                    precision=0,
                                    minimum=0
                                )
                                
                                gen_virtual_max = gr.Number(
                                    label="虚拟工具最大数量",
                                    value=8,
                                    precision=0,
                                    minimum=1
                                )
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 📋 Cases配置（JSON格式）")
                                
                                gen_cases_config = gr.Code(
                                    label="Cases配置",
                                    language="json",
                                    value="""{
    "case_C1": {
        "target_count": 10,
        "data_output": "/path/to/output/case_C1_data.jsonl",
        "score_output": "/path/to/output/case_C1_score.jsonl"
    },
    "case_C2": {
        "target_count": 10,
        "data_output": "/path/to/output/case_C2_data.jsonl",
        "score_output": "/path/to/output/case_C2_score.jsonl"
    }
}""",
                                    lines=15
                                )
                                
                                gr.Markdown("""
                                **配置说明：**
                                - `target_count`: 目标生成数量
                                - `data_output`: 生成数据保存路径
                                - `score_output`: 打分结果保存路径
                                - 可添加任意多个case配置
                                """)
                        
                        with gr.Row():
                            gen_start_btn = gr.Button("🚀 开始生成与校验", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gen_output_status = gr.Textbox(
                                    label="处理状态",
                                    placeholder="等待开始...",
                                    lines=20,
                                    max_lines=25,
                                    interactive=False
                                )
                            
                            with gr.Column(scale=1):
                                gen_process_log = gr.Textbox(
                                    label="📋 处理日志",
                                    placeholder="处理日志将在这里显示...",
                                    lines=20,
                                    max_lines=25,
                                    interactive=False
                                )
                        
                        # 定义功能三的辅助函数
                        def update_gen_input_files_from_folder(folder_path):
                            """扫描文件夹并更新输入文件下拉框"""
                            if not folder_path or not os.path.exists(folder_path):
                                return gr.update(choices=[], value=None)
                            
                            files = scan_folder_for_files(folder_path, ".jsonl")
                            return gr.update(choices=files, value=files[0] if files else None)
                        
                        def update_gen_input_full_path_from_selection(folder, filename):
                            """根据选择的文件更新完整路径"""
                            return get_full_path(folder, filename) if filename else ""
                        
                        # 手动刷新按钮
                        gen_refresh_input_btn.click(
                            fn=update_gen_input_files_from_folder,
                            inputs=[gen_input_folder],
                            outputs=[gen_input_file_dropdown]
                        )
                        
                        # 文件夹路径改变时，自动扫描
                        gen_input_folder.change(
                            fn=update_gen_input_files_from_folder,
                            inputs=[gen_input_folder],
                            outputs=[gen_input_file_dropdown]
                        )
                        
                        # 文件选择改变时，更新完整路径
                        gen_input_file_dropdown.change(
                            fn=update_gen_input_full_path_from_selection,
                            inputs=[gen_input_folder, gen_input_file_dropdown],
                            outputs=[gen_input_file]
                                )
                        
                        # 绑定功能三的处理事件
                        gen_start_btn.click(
                            fn=generator_gen_judge.process_data_generate_judge,
                            inputs=[
                                gen_input_file,
                                gen_model,
                                gen_temperature,
                                gen_max_tokens,
                                gen_system_prompt,
                                gen_user_prompt,
                                gen_tool_bank,
                                gen_virtual_min,
                                gen_virtual_max,
                                gen_cases_config
                            ],
                            outputs=[gen_output_status, gen_process_log]
                        )
                        
                        gr.Markdown("""
                        ---
                        ### 📖 功能说明
                        
                        此功能整合了数据生成和校验流程：
                        
                        **生成阶段：**
                        1. 读取预处理的标注数据
                        2. 根据配置生成多轮对话数据
                        3. 使用虚拟工具库增强数据多样性
                        
                        **校验阶段：**
                        1. 规则校验：检查格式、工具调用、参数等
                        2. LLM校验：使用GPT评估对话质量
                        3. 打分：rule_score (0/1) + gpt_score (0/1) = total_score (0-2)
                        
                        **输出结果：**
                        - 生成数据文件：包含完整的对话数据
                        - 打分文件：包含每条数据的评分和原因
                        
                        **评分标准：**
                        - total_score = 2：通过所有校验，保存到生成数据文件
                        - total_score < 2：校验失败，仅记录打分信息
                        """)
                    
                    # 功能三 - 文件查看器（增强版：支持多文件）
                    with gr.Tab("📂 文件查看器"):
                        gr.Markdown("""
                        ### 📖 多文件查看器
                        
                        查看输入文件和所有生成的输出文件（包括数据文件和打分文件）。
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gen_file_selector = gr.Dropdown(
                                    label="选择文件",
                                    choices=["输入文件"],
                                    value="输入文件",
                                    interactive=True
                                )
                                
                                gen_refresh_btn = gr.Button("🔄 刷新文件列表", size="sm")
                                
                                gen_custom_path = gr.Textbox(
                                    label="或输入自定义路径",
                                    placeholder="/path/to/your/file.jsonl",
                                    visible=True
                                )
                                
                                gen_line_number = gr.Slider(
                                    label="选择行号",
                                    minimum=1,
                                    maximum=100,
                                    value=1,
                                    step=1,
                                    interactive=True
                                )
                                
                                gen_view_btn = gr.Button("🔍 查看内容", variant="primary")
                            
                            with gr.Column(scale=2):
                                gen_file_info = gr.Textbox(
                                    label="文件信息",
                                    lines=5,
                                    interactive=False
                                )
                                
                                gen_line_content = gr.Textbox(
                                    label="行内容（JSON格式）",
                                    lines=20,
                                    interactive=False
                                )
                        
                        # 刷新文件列表
                        def refresh_file_list():
                            files = generator_gen_judge.get_all_output_files()
                            if not files:
                                return gr.update(choices=["输入文件", "自定义路径"], value="输入文件")
                            
                            file_choices = [name for name, _ in files]
                            file_choices.append("自定义路径")
                            return gr.update(choices=file_choices, value=file_choices[0] if file_choices else "输入文件")
                        
                        # 查看文件内容
                        def view_gen_file_content(file_selector, custom_path, line_number):
                            # 确定文件路径
                            if file_selector == "自定义路径":
                                file_path = custom_path
                            elif file_selector == "输入文件":
                                file_path = generator_gen_judge.current_input_file
                            else:
                                # 从输出文件中查找
                                files = generator_gen_judge.get_all_output_files()
                                file_path = None
                                for name, path in files:
                                    if name == file_selector:
                                        file_path = path
                                        break
                                
                                if not file_path:
                                    return "文件未找到", ""
                            
                            if not file_path:
                                return "请选择文件或输入路径", ""
                            
                            # 获取文件信息和内容
                            info, total_lines = generator_gen_judge.get_file_info(file_path)
                            
                            if line_number and line_number > 0:
                                content = generator_gen_judge.get_line_content(file_path, int(line_number))
                            else:
                                content = "请输入要查看的行号"
                            
                            return info, content
                        
                        # 更新行号滑块最大值
                        def update_gen_slider(file_selector, custom_path):
                            if file_selector == "自定义路径":
                                file_path = custom_path
                            elif file_selector == "输入文件":
                                file_path = generator_gen_judge.current_input_file
                            else:
                                files = generator_gen_judge.get_all_output_files()
                                file_path = None
                                for name, path in files:
                                    if name == file_selector:
                                        file_path = path
                                        break
                            
                            if not file_path or not os.path.exists(file_path):
                                return gr.update(maximum=1, value=1)
                            
                            _, total_lines = generator_gen_judge.get_file_info(file_path)
                            return gr.update(maximum=max(1, total_lines), value=1)
                        
                        # 绑定事件
                        gen_refresh_btn.click(
                            fn=refresh_file_list,
                            outputs=[gen_file_selector]
                        )
                        
                        gen_file_selector.change(
                            fn=update_gen_slider,
                            inputs=[gen_file_selector, gen_custom_path],
                            outputs=[gen_line_number]
                        )
                        
                        gen_custom_path.change(
                            fn=update_gen_slider,
                            inputs=[gen_file_selector, gen_custom_path],
                            outputs=[gen_line_number]
                        )
                        
                        gen_view_btn.click(
                            fn=view_gen_file_content,
                            inputs=[gen_file_selector, gen_custom_path, gen_line_number],
                            outputs=[gen_file_info, gen_line_content]
                        )
                        
                        gen_line_number.change(
                            fn=view_gen_file_content,
                            inputs=[gen_file_selector, gen_custom_path, gen_line_number],
                            outputs=[gen_file_info, gen_line_content]
                        )
                        
                        gr.Markdown("""
                        ---
                        ### 📖 文件查看器说明
                        
                        - **输入文件**：查看原始输入数据
                        - **case_XX-数据**：查看该case生成的对话数据
                        - **case_XX-打分**：查看该case的评分结果
                        - **自定义路径**：查看任意JSONL文件
                        - 使用"🔄 刷新文件列表"按钮更新可用文件列表
                        """)
        
        # ==================== 跨功能的事件绑定 ====================
        # 需要在这里定义跨Tab的事件绑定（所有组件都已创建完成）
        
        # 定义辅助函数
        def extract_folder_from_f2_output(f2_output_path):
            """
            从功能二的输出路径中提取文件夹路径
            
            智能判断：
            1. 如果输入是文件路径（.jsonl结尾）→ 提取文件夹
            2. 如果输入本身就是文件夹 → 直接使用
            3. 如果路径不存在 → 尝试提取文件夹部分
            """
            if not f2_output_path:
                return "", gr.update(choices=[], value=None), ""
            
            # 判断输入路径的类型
            if os.path.exists(f2_output_path):
                # 路径存在，判断是文件还是文件夹
                if os.path.isdir(f2_output_path):
                    # 本身就是文件夹
                    folder_path = f2_output_path
                else:
                    # 是文件，提取文件夹
                    folder_path = os.path.dirname(f2_output_path)
            else:
                # 路径不存在，根据扩展名判断
                if f2_output_path.endswith('.jsonl'):
                    # 看起来是文件路径，提取文件夹
                    folder_path = os.path.dirname(f2_output_path)
                else:
                    # 可能本身就是文件夹路径
                    folder_path = f2_output_path
            
            # 扫描文件
            files = scan_folder_for_files(folder_path, ".jsonl") if folder_path and os.path.exists(folder_path) else []
            
            return (
                folder_path,
                gr.update(choices=files, value=files[0] if files else None),
                get_full_path(folder_path, files[0]) if files else ""
            )
        
        # 功能二输出路径改变时，自动更新功能三输入
        f1_output_file.change(
            fn=extract_folder_from_f2_output,
            inputs=[f1_output_file],
            outputs=[gen_input_folder, gen_input_file_dropdown, gen_input_file]
        )
        
        # 底部说明
        gr.Markdown("""
        ---
        ### 🎯 使用指南
        
        #### 📚 功能架构
        - **功能一：工具变体生成** - 基于原始工具自动生成语义相似的变体工具
        - **功能二：工具标注** - 使用LLM对多跳问题进行工具选择和路径规划
        - **功能三：数据生成与校验** - 整合数据生成和质量校验流程
        
        #### 🔧 如何添加新功能
        1. 创建新的处理类（参考 `feature_template.py`）
        2. 实现处理逻辑和文件操作方法
        3. 在主界面添加新的Tab
        4. 使用 `create_file_viewer()` 快速创建文件查看器
        
        #### ⚙️ 每个功能包含
        - **数据处理页面**：配置参数并执行处理任务
        - **文件查看器页面**：查看输入/输出文件内容
        - **实时反馈**：处理状态和结果实时展示
        """)
    
    return demo

# 启动应用
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
