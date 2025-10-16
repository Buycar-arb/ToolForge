# 文件处理相关工具
import json
import random
from pathlib import Path
from stage_2_generate.config.settings import TARGET_GENERAL_FILE
class FileProcessor:
    @staticmethod
    def load_tool_definitions(tools_dir):
        """
        Load all tool definitions from JSONL files in the specified tools directory.
        
        Returns:
            dict: A mapping of {tool_name: tool_definition}.
        """
        tool_definitions = {}
        
        tools_path = Path(tools_dir)
        if not tools_path.exists():
            print(f"Tool directory not found: {tools_dir}")
            return tool_definitions
        
        # Iterate through all JSONL files in the directory
        for jsonl_file in tools_path.glob("*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            tool_data = json.loads(line)
                            if 'name' in tool_data:
                                tool_name = tool_data['name']
                                tool_definitions[tool_name] = tool_data
                            else:
                                print(f"Warning: Missing 'name' field in {jsonl_file.name}, line {line_num}")
                        
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON in {jsonl_file.name}, line {line_num}: {e}")
                            continue
            
            except Exception as e:
                print(f"Error reading file {jsonl_file}: {e}")
        
        # print(f"Total tools loaded: {len(tool_definitions)}")
        return tool_definitions


    @staticmethod
    def _extract_random_tool_from_file(file_path):
        """Randomly extract a single tool definition from a JSONL file."""
        try:
            tools_in_file = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            tool = json.loads(line)
                            tools_in_file.append(tool)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON in {file_path}, line {line_num}: {e}")
                            continue
            
            if tools_in_file:
                selected_tool = random.choice(tools_in_file)
                
                # print(f"Selected tool from {file_path}: {selected_tool.get('name', 'Unknown')}")
                return selected_tool
            else:
                print(f"Warning: {file_path} is empty or contains no valid tools.")
                return None
                
        except Exception as e:
            print(f"Error: Failed to read {file_path}: {e}")
            return None

    @staticmethod
    def load_random_tools_excluding_good_tools(tool_bank_dir, good_tool_content):
        """
        Randomly select tools from JSONL files in the tool_bank directory,
        excluding the files specified in good_tool_content.
        Additionally, randomly select tools from the good_tool_content files.

        Args:
            tool_bank_dir (str): Directory containing JSONL tool definition files.
            good_tool_content (list): List of tool names to exclude, e.g.,
                ["transportation_system_search", "science_tech_search"].

        Returns:
            tuple: (
                random_tools,
                good_tools,
                available_general_status,
                good_general_status,
                general_tool_name,
                good_tool_mapping,
                good_tool_full_data
            )
                - random_tools: List of randomly selected tools from non-excluded files.
                - good_tools: List of randomly selected tools from good_tool_content files.
                - available_general_status: Status of the general file in available_files,
                either "exist general_file" or "without general_file".
                - good_general_status: Status of the general file in good_tool_files,
                either "exist general_file" or "without general_file".
                - general_tool_name: Extracted tool name from the general file (or False if not found).
                - good_tool_mapping: List of mappings in the format
                [{"original_tool": <filename>, "diversity": <tool_name>}, ...].
                - good_tool_full_data: List of complete tool definitions for good tools.
        """
        # Target general tool file path
        target_general_file = TARGET_GENERAL_FILE
        
        # Collect all JSONL files
        all_jsonl_files = []
        for file_path in Path(tool_bank_dir).glob("*.jsonl"):
            if file_path.is_file():
                all_jsonl_files.append(file_path)
        
        if not all_jsonl_files:
            print(f"Warning: No JSONL files found in {tool_bank_dir}")
            return [], [], "without general_file", "without general_file", False, [], []
        
        # Classify files into good_tool files and others
        available_files = []      # Files not in good_tool_content
        good_tool_files = []      # Files in good_tool_content
        excluded_files = []       # Record of excluded file names
        missing_good_files = []   # Record of missing good_tool files
        
        # Map filenames (without extension) to their paths
        file_map = {file_path.stem: file_path for file_path in all_jsonl_files}
        
        # Check if the general file exists
        general_file_exists = False
        for file_path in all_jsonl_files:
            if str(file_path.absolute()) == target_general_file:
                general_file_exists = True
                break
        
        for file_path in all_jsonl_files:
            file_stem = file_path.stem
            if file_stem in good_tool_content:
                good_tool_files.append(file_path)
                excluded_files.append(file_path.name)
            else:
                available_files.append(file_path)
        
        # Check if any good_tool files are missing
        for good_tool_name in good_tool_content:
            if good_tool_name not in file_map:
                missing_good_files.append(f"{good_tool_name}.jsonl")
        
        if missing_good_files:
            print(f"Warning: Missing good_tool files: {missing_good_files}")
        
        # Initialize status variables
        available_general_status = "without general_file"
        good_general_status = "without general_file"
        general_tool_name_and_content = False
        
        # Randomly select one tool from each non-excluded file
        random_tools = []
        if available_files:
            for file_path in available_files:
                tool = FileProcessor._extract_random_tool_from_file(file_path)
                if tool:
                    random_tools.append(tool)
                    # Check if this file is the general tool file
                    if str(file_path.absolute()) == target_general_file:
                        available_general_status = "exist general_file"
                        if "name" in tool:
                            general_tool_name_and_content = tool
                        else:
                            print("Warning: Tool in general file has no 'name' field.")
        else:
            print("Warning: No available JSONL files found outside good_tool_content.")
        
        # Randomly select tools from good_tool files
        good_tools_list = []
        good_tool_mapping = []      # List of mappings between original and selected tools
        good_tool_full_data = []    # List of full tool data
        
        if good_tool_files:
            for file_path in good_tool_files:
                tool = FileProcessor._extract_random_tool_from_file(file_path)
                if tool:
                    good_tools_list.append(tool)
                    
                    # Create mapping dictionary and append
                    file_stem = file_path.stem
                    tool_name = tool.get("name", "Unknown")
                    mapping_dict = {
                        "original_tool": file_stem,
                        "diversity": tool_name
                    }
                    good_tool_mapping.append(mapping_dict)
                    
                    # Check if this is the general file
                    if str(file_path.absolute()) == target_general_file:
                        good_general_status = "exist general_file"
                        if "name" in tool:
                            general_tool_name_and_content = tool
                        else:
                            print("Warning: Tool in general file has no 'name' field.")
        else:
            print("Warning: No good_tool JSONL files found.")
        
        return (
            random_tools,
            good_tools_list,
            available_general_status,
            good_general_status,
            general_tool_name_and_content,
            good_tool_mapping,
            good_tool_full_data
        )
