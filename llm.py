
class LLM:
    def __init__(self, name, prompt_template):
        self.name = name
        self.prompt_template = prompt_template
        
    def create_full_prompt(self, actual_prompt):
        parts = self.prompt_template.split('\n')
        parts[1] = actual_prompt
        full_prompt = '\n'.join(parts)
        return full_prompt