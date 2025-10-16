from stage_2_generate.processors.case_processors import CaseA1Processor, CaseA2Processor, CaseA3Processor, CaseA4Processor
from stage_2_generate.processors.case_processors import CaseB1Processor, CaseB2Processor, CaseB3Processor, CaseB4Processor, CaseB5Processor, CaseB6Processor 
from stage_2_generate.processors.case_processors import CaseC1Processor, CaseC3Processor, CaseC4Processor, CaseC5Processor, CaseC6Processor, CaseC7Processor, CaseC8Processor, CaseC9Processor, CaseC10Processor
from stage_2_generate.processors.case_processors import CaseD1Processor, CaseD2Processor, CaseD3Processor, CaseD4Processor, CaseD5Processor, CaseD6Processor, CaseD7Processor, CaseD8Processor, CaseD9Processor, CaseD10Processor
class ConversationGenerator:
    def __init__(self, **kwargs):
        self.processors = {
            'case_A1': CaseA1Processor(**kwargs),
            'case_A2': CaseA2Processor(**kwargs),
            'case_A3': CaseA3Processor(**kwargs),
            'case_A4': CaseA4Processor(**kwargs),
            'case_B1': CaseB1Processor(**kwargs),
            'case_B2': CaseB2Processor(**kwargs),
            'case_B3': CaseB3Processor(**kwargs),
            'case_B4': CaseB4Processor(**kwargs),
            'case_B5': CaseB5Processor(**kwargs),
            'case_B6': CaseB6Processor(**kwargs),
            'case_C1': CaseC1Processor(**kwargs),
            'case_C3': CaseC3Processor(**kwargs),
            'case_C4': CaseC4Processor(**kwargs),
            'case_C5': CaseC5Processor(**kwargs),
            'case_C6': CaseC6Processor(**kwargs),
            'case_C7': CaseC7Processor(**kwargs),
            'case_C8': CaseC8Processor(**kwargs),
            'case_C9': CaseC9Processor(**kwargs),
            'case_C10': CaseC10Processor(**kwargs),
            'case_D1': CaseD1Processor(**kwargs),
            'case_D2': CaseD2Processor(**kwargs),
            'case_D3': CaseD3Processor(**kwargs),
            'case_D4': CaseD4Processor(**kwargs),
            'case_D5': CaseD5Processor(**kwargs),
            'case_D6': CaseD6Processor(**kwargs),
            'case_D7': CaseD7Processor(**kwargs),
            'case_D8': CaseD8Processor(**kwargs),
            'case_D9': CaseD9Processor(**kwargs),
            'case_D10': CaseD10Processor(**kwargs),
        }
    
    async def process(self, processor_case):
        if processor_case not in self.processors:
            raise ValueError(f"Unsupported processing type: {processor_case}")
        return await self.processors[processor_case].process()