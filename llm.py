
class LLM:
    def __init__(self, name, prompt_template):
        self.name = name
        self.prompt_template = prompt_template
        
    def create_full_prompt(self, actual_prompt, mistral_instruct=False):
        delimeter = ' ' if mistral_instruct else '\n' 
        parts = self.prompt_template.split(delimeter)
        parts[1] = actual_prompt
        full_prompt = delimeter.join(parts)
        return full_prompt