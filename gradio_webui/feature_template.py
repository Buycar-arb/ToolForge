"""
New Feature Template
Copy this template and modify to create new features
"""

import json
import os
from datetime import datetime
from collections import deque

class FeatureTemplateProcessor:
    """
    Feature template processing class
    
    Usage steps:
    1. Copy this class and rename (e.g., FeatureDataCleanProcessor)
    2. Modify state variables in __init__
    3. Implement process_data_xxx method
    4. Import and create instance in quick_fast.py
    5. Add corresponding Tab in UI
    """
    
    def __init__(self):
        # Basic state
        self.is_running = False
        self.current_input_file = ""
        self.current_output_file = ""
        
        # Custom state (add as needed)
        self.process_logs = deque(maxlen=100)  # Processing logs
        self.results = []  # Processing results
        # self.custom_state = ...  # Add more states
    
    def process_data_template(
        self,
        input_file: str,
        output_file: str,
        param1: str,           # Modify parameters as needed
        param2: int,           # Modify parameters as needed
        progress=None          # Gradio Progress object
    ):
        """
        Main processing function
        
        Args:
            input_file: Input file path
            output_file: Output file path
            param1: Custom parameter 1
            param2: Custom parameter 2
            progress: Gradio progress object
        
        Returns:
            tuple: (status message, log message)
        """
        try:
            self.is_running = True
            self.current_input_file = input_file
            self.current_output_file = output_file
            
            # ========== Step 1: Validate input ==========
            if progress:
                progress(0, desc="🔍 Validating input...")
            
            if not os.path.exists(input_file):
                return f"❌ Error: Input file does not exist - {input_file}", ""
            
            # Create output directory
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            
            # ========== Step 2: Read data ==========
            if progress:
                progress(0.1, desc="📚 Reading data...")
            
            input_data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        input_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Line {i} JSON parsing failed: {e}")
                        continue
            
            total_count = len(input_data)
            print(f"Read {total_count} data items")
            
            # ========== Step 3: Process data ==========
            if progress:
                progress(0.3, desc="🔄 Processing data...")
            
            processed_data = []
            success_count = 0
            error_count = 0
            
            for i, item in enumerate(input_data):
                try:
                    # TODO: Implement your processing logic here
                    # Example:
                    processed_item = self._process_single_item(item, param1, param2)
                    processed_data.append(processed_item)
                    success_count += 1
                    
                    # Update progress
                    if progress and (i + 1) % 10 == 0:
                        progress_value = 0.3 + 0.5 * (i + 1) / total_count
                        progress(progress_value, desc=f"🔄 Processing {i + 1}/{total_count}")
                    
                except Exception as e:
                    print(f"Processing item {i+1} failed: {e}")
                    error_count += 1
                    # Can choose to keep original data or mark error
                    item['processing_error'] = str(e)
                    processed_data.append(item)
            
            # ========== Step 4: Save results ==========
            if progress:
                progress(0.8, desc="💾 Saving results...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # ========== Step 5: Generate report ==========
            if progress:
                progress(1.0, desc="✅ Completed!")
            
            # Status message
            result_msg = f"""
✅ Processing completed!

📊 Statistics:
   - Total data: {total_count} items
   - Successfully processed: {success_count} items
   - Failed: {error_count} items
   - Success rate: {success_count/total_count*100:.2f}%

⚙️ Processing parameters:
   - Parameter 1: {param1}
   - Parameter 2: {param2}

📁 Output file:
   - {output_file}

⏰ Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Log message
            log_msg = self._generate_log()
            
            return result_msg, log_msg
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error occurred during processing:\n{str(e)}\n\nDetailed information:\n{traceback.format_exc()}"
            return error_msg, ""
        
        finally:
            self.is_running = False
    
    def _process_single_item(self, item, param1, param2):
        """
        Process single data item
        
        TODO: Implement your core processing logic here
        
        Args:
            item: Single data item (dict)
            param1: Parameter 1
            param2: Parameter 2
        
        Returns:
            dict: Processed data
        """
        # Example processing logic
        processed_item = item.copy()
        
        # TODO: Add your processing code
        # For example:
        # processed_item['new_field'] = some_processing(item, param1, param2)
        
        # Record log
        log_entry = f"Processing item: {item.get('id', 'unknown')} - Parameters: {param1}, {param2}"
        self.process_logs.append(log_entry)
        
        return processed_item
    
    def _generate_log(self):
        """Generate processing log"""
        if not self.process_logs:
            return "No processing logs available"
        
        log_text = f"📋 Processing logs (latest {len(self.process_logs)} items)\n\n"
        log_text += "="*80 + "\n\n"
        
        for log in self.process_logs:
            log_text += f"• {log}\n"
        
        return log_text
    
    # ========== File operation methods (standard implementation, no modification needed) ==========
    
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


# ==================== UI Integration Code Template ====================
"""
Add the following code in quick_fast.py:

# 1. Import processing class
from feature_template import FeatureTemplateProcessor

# 2. Create global instance
generator_feature_template = FeatureTemplateProcessor()

# 3. Add Tab in create_ui() function

with gr.Tab("🎯 Feature Name"):
    with gr.Tabs():
        # Data processing
        with gr.Tab("📝 Data Processing"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📁 File Path Configuration")
                    
                    ft_input_file = gr.Textbox(
                        label="Input file path",
                        placeholder="/path/to/input.jsonl"
                    )
                    
                    ft_output_file = gr.Textbox(
                        label="Output file path",
                        placeholder="/path/to/output.jsonl"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Processing Parameter Configuration")
                    
                    ft_param1 = gr.Textbox(
                        label="Parameter 1",
                        value=""
                    )
                    
                    ft_param2 = gr.Number(
                        label="Parameter 2",
                        value=10,
                        precision=0
                    )
            
            with gr.Row():
                ft_start_btn = gr.Button("🚀 Start Processing", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=1):
                    ft_output_status = gr.Textbox(
                        label="Processing Status",
                        lines=15,
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    ft_log = gr.Textbox(
                        label="📋 Processing Log",
                        lines=15,
                        interactive=False
                    )
            
            # Bind events
            ft_start_btn.click(
                fn=generator_feature_template.process_data_template,
                inputs=[
                    ft_input_file,
                    ft_output_file,
                    ft_param1,
                    ft_param2
                ],
                outputs=[ft_output_status, ft_log]
            )
        
        # File viewer
        with gr.Tab("📂 File Viewer"):
            create_file_viewer(generator_feature_template, "Feature Name")
"""


# ==================== Usage Example ====================
if __name__ == "__main__":
    # Test processor
    processor = FeatureTemplateProcessor()
    
    # Create test data
    test_input = "test_input.jsonl"
    test_output = "test_output.jsonl"
    
    with open(test_input, 'w', encoding='utf-8') as f:
        f.write(json.dumps({"id": 1, "text": "Test data 1"}, ensure_ascii=False) + '\n')
        f.write(json.dumps({"id": 2, "text": "Test data 2"}, ensure_ascii=False) + '\n')
    
    # Run processing
    result, log = processor.process_data_template(
        test_input,
        test_output,
        param1="Test parameter",
        param2=100
    )
    
    print(result)
    print("\n" + "="*80 + "\n")
    print(log)

