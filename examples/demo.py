# Constants for caption generation, copied from the original JoyCaption GGUF node
CAPTION_TYPE_MAP = {
}

CAPTION_LENGTH_CHOICES = (["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)])

def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
	if caption_type not in CAPTION_TYPE_MAP:
		print(f"Warning: Unknown caption_type '{caption_type}'. Using default.")
		default_template_key = list(CAPTION_TYPE_MAP.keys())[0] 
		prompt_templates = CAPTION_TYPE_MAP.get(caption_type, CAPTION_TYPE_MAP[default_template_key])
	else:
		prompt_templates = CAPTION_TYPE_MAP[caption_type]

	if caption_length == "any": map_idx = 0
	elif isinstance(caption_length, str) and caption_length.isdigit(): map_idx = 1
	else: map_idx = 2
	
	if map_idx >= len(prompt_templates): map_idx = 0 

	prompt = prompt_templates[map_idx]
	if extra_options: prompt += " " + " ".join(extra_options)
	
	try:
		return prompt.format(name=name_input or "{NAME}", length=caption_length, word_count=caption_length)
	except KeyError as e:
		print(f"Warning: Prompt template formatting error for caption_type '{caption_type}', map_idx {map_idx}. Missing key: {e}")
		return prompt + f" (Formatting error: missing key {e})"



class JoyCaptionOllamaPrompter:
    CATEGORY = 'JoyCaption/Ollama'
    FUNCTION = "generate_prompts"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "user_prompt")

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "caption_type": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive (Casual)"}),
            "caption_length": (CAPTION_LENGTH_CHOICES, {"default": "medium-length"}),
        }
        opt = {
            "extra_options_input": ("JJC_OLLAMA_EXTRA_OPTION",)
        }
        return {"required": req, "optional": opt}

    def generate_prompts(self, caption_type, caption_length, extra_options_input=None):
        extras_list = []
        char_name = ""
        if extra_options_input:
            if isinstance(extra_options_input, tuple) and len(extra_options_input) == 2:
                extras_list, char_name = extra_options_input
                if not isinstance(extras_list, list): extras_list = []
                if not isinstance(char_name, str): char_name = ""
            else:
                print(f"JoyCaption (Ollama Prompter) Warning: extra_options_input is not in the expected format. Received: {type(extra_options_input)}")
        
        user_prompt = build_prompt(caption_type, caption_length, extras_list, char_name)
        
        system_prompt =""

        # Add length enforcement to system prompt
        if isinstance(caption_length, (int, str)) and str(caption_length).isdigit():
            system_prompt += f"\nIMPORTANT: Your response MUST NOT exceed {caption_length} words."
        elif caption_length in ["very short", "short", "medium-length", "long", "very long"]:
            length_guides = {
                "very short": "25", "short": "50", "medium-length": "100",
                "long": "150", "very long": "200"
            }
            system_prompt += f"\nIMPORTANT: Keep your response approximately {length_guides[caption_length]} words."
            
        return (system_prompt.strip(), user_prompt.strip())

 