"""
Tool List Management Feature
Used to manage TOOL_LIST in tool_prompts.py
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

class ToolListManager:
    """Tool list manager"""
    
    def __init__(self):
        # Get project root directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.current_dir)
        
        # Tool bank path
        self.tool_bank_dir = os.path.join(self.project_root, "stage_2_generate", "tool_bank", "tools")
        
        # tool_prompts.py path
        self.tool_prompts_path = os.path.join(self.project_root, "stage_1_label", "code", "tool_prompts.py")
        
        # Cache
        self.available_tools = {}  # {tool_name: description}
        self.current_tool_list = []  # Current tool names in TOOL_LIST
    
    def scan_tool_bank(self):
        """
        Scan tool_bank directory to get all available tools
        
        Returns:
            dict: {tool_name: description}
        """
        tools = {}
        
        try:
            if not os.path.exists(self.tool_bank_dir):
                print(f"⚠️ Tool bank directory does not exist: {self.tool_bank_dir}")
                return tools
            
            # Traverse all .jsonl files
            for filename in os.listdir(self.tool_bank_dir):
                if not filename.endswith('.jsonl'):
                    continue
                
                # Tool name = filename (remove .jsonl)
                tool_name = filename[:-6]  # Remove .jsonl
                
                # Read first line to get description
                file_path = os.path.join(self.tool_bank_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            tool_data = json.loads(first_line)
                            description = tool_data.get('description', 'Description missing')
                        else:
                            description = 'Description missing (file is empty)'
                except json.JSONDecodeError:
                    description = 'Description missing (JSON format error)'
                except Exception as e:
                    description = f'Description missing (read error: {str(e)})'
                
                tools[tool_name] = description
            
            print(f"✅ Scanned {len(tools)} tools")
            
        except Exception as e:
            print(f"❌ Tool bank scanning failed: {e}")
        
        self.available_tools = tools
        return tools
    
    def load_current_tool_list(self):
        """
        Load current TOOL_LIST from tool_prompts.py
        
        Returns:
            list: Tool name list
        """
        tools = []
        
        try:
            if not os.path.exists(self.tool_prompts_path):
                print(f"⚠️ tool_prompts.py does not exist: {self.tool_prompts_path}")
                return tools
            
            # Read file content
            with open(self.tool_prompts_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use regex to extract TOOL_LIST content
            # Match: TOOL_LIST = """..."""
            pattern = r'TOOL_LIST\s*=\s*"""(.*?)"""'
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                tool_list_content = match.group(1).strip()
                
                # Parse each line
                for line in tool_list_content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Format: tool_name：description
                    if '：' in line:
                        tool_name = line.split('：')[0].strip()
                        tools.append(tool_name)
                
                print(f"✅ Loaded {len(tools)} tools from TOOL_LIST")
            else:
                print("⚠️ TOOL_LIST definition not found")
        
        except Exception as e:
            print(f"❌ Loading TOOL_LIST failed: {e}")
        
        self.current_tool_list = tools
        return tools
    
    def save_tool_list(self, selected_tools):
        """
        Save tool list to tool_prompts.py
        
        Args:
            selected_tools: list of tool names
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            if not os.path.exists(self.tool_prompts_path):
                return False, f"❌ tool_prompts.py does not exist: {self.tool_prompts_path}"
            
            # Read original file content
            with open(self.tool_prompts_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Build new TOOL_LIST content
            tool_list_lines = []
            for tool_name in selected_tools:
                description = self.available_tools.get(tool_name, 'Description missing')
                tool_list_lines.append(f"{tool_name}：{description}")
            
            new_tool_list = '\n'.join(tool_list_lines)
            new_tool_list_block = f'TOOL_LIST = """\n{new_tool_list}\n"""'
            
            # Replace TOOL_LIST section
            pattern = r'TOOL_LIST\s*=\s*""".*?"""'
            new_content = re.sub(pattern, new_tool_list_block, content, flags=re.DOTALL)
            
            # Write back to file
            with open(self.tool_prompts_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.current_tool_list = selected_tools
            
            return True, f"✅ Successfully saved {len(selected_tools)} tools to TOOL_LIST"
        
        except Exception as e:
            import traceback
            return False, f"❌ Save failed: {str(e)}\n\n{traceback.format_exc()}"
    
    def get_tool_info_text(self, tool_names):
        """
        Get tool information text display
        
        Args:
            tool_names: list of tool names
        
        Returns:
            str: Formatted tool information text
        """
        if not tool_names:
            return "No tools available"
        
        lines = []
        for i, tool_name in enumerate(tool_names, 1):
            description = self.available_tools.get(tool_name, 'Description missing')
            lines.append(f"{i}. {tool_name}")
            lines.append(f"   {description}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def get_available_tools_choices(self):
        """
        Get available tools choice list (for Gradio CheckboxGroup)
        
        Returns:
            list: [(tool_name, is_in_current_list), ...]
        """
        choices = []
        for tool_name in sorted(self.available_tools.keys()):
            label = f"{tool_name}"
            if tool_name in self.current_tool_list:
                label += " ✓"
            choices.append(label)
        
        return choices
    
    def get_statistics(self):
        """
        Get statistics information
        
        Returns:
            str: Statistics information text
        """
        total_available = len(self.available_tools)
        total_selected = len(self.current_tool_list)
        
        info = f"""
📊 Tool Statistics

- Total available tools: {total_available}
- Selected tools count: {total_selected}
- Unselected tools count: {total_available - total_selected}

📁 Tool bank path: {self.tool_bank_dir}
📄 Configuration file path: {self.tool_prompts_path}

⏰ Update time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return info


# Create global instance
tool_list_manager = ToolListManager()

