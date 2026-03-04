import asyncio
import re
import textwrap
from copy import deepcopy
from typing import Dict, List, Optional

import json
import torch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from swift.llm.template import register_template
from swift.llm.template.template.qwen import QwenTemplateMeta
from swift.llm.template.base import Template
from swift.utils import get_logger

from collections import Counter
import numpy as np

logger = get_logger()

class ThinkingTemplate2(Template):
    
    def _swift_prepare_messages(self, messages):
        super()._swift_prepare_messages(messages)
        
        
register_template(
    QwenTemplateMeta(
        "mymymy",   # 模板名称，可以自定义，但是别用qwen3，qwq等等这些已有的模板名称
        default_system=None, 
        template_cls=ThinkingTemplate2
    )
)