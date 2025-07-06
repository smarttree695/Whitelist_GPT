import numpy as np
import tkinter as tk
from tkinter import ttk
from string import ascii_lowercase, punctuation
import time
from llama_cpp import Llama

class MistralWhitelistGenerator:
    def __init__(self, model_path):
        """Initialize with a path to a Mistral GGUF model."""
        # Initialize model
        self.llm = Llama(
            model_path=model_path,
            use_mlock=False,
            use_mmap=True,
            gpu_layers=40  # Adjust as needed based on GPU memory
        )
        
        # Initialize token cache
        self.token_first_chars = {}
        self.token_second_chars = {}
        self.approved_chars = set()
        
        # Pre-compute token information
        self._compute_token_first_chars()
    
    def _compute_token_first_chars(self):
        """Pre-compute the first and second characters of each token."""
        vocab_size = self.llm.n_vocab()
        
        for token_id in range(vocab_size):
            try:
                # Get token text from token ID
                token_bytes = self.llm.detokenize([token_id])
                token_text = token_bytes.decode('utf-8', errors='replace')
                
                if token_text:
                    # Store first character (lowercase for case-insensitive comparison)
                    first_char = token_text[0].lower()
                    self.token_first_chars[token_id] = first_char
                    
                    # Store second character if token starts with space and has length > 1
                    if first_char == ' ' and len(token_text) > 1:
                        self.token_second_chars[token_id] = token_text[1].lower()
            except Exception as e:
                # Skip any problematic tokens
                pass
    
    def create_whitelist_processor(self, approved_chars):
        """Create a logits processor function that allows only tokens starting with approved chars."""
        self.approved_chars = set(char.lower() for char in approved_chars)
        allowed_token_ids = set()
        
        # Build a set of allowed token IDs
        for token_id, first_char in self.token_first_chars.items():
            allow_token = False
            
            if first_char == ' ' and ' ' in self.approved_chars:
                # For space-prefixed tokens, check second character
                second_char = self.token_second_chars.get(token_id)
                if second_char and second_char in self.approved_chars:
                    allow_token = True
            elif first_char in self.approved_chars:
                # For non-space tokens, just check first character
                allow_token = True
                
            if allow_token:
                allowed_token_ids.add(token_id)
        
        # Create the logits processor function
        def whitelist_logits_processor(input_ids, scores):
            """
            Mask logits to only allow tokens starting with approved characters.
            
            Args:
                input_ids: Token IDs of the prompt (not used in this processor)
                scores: Logits for each token in the vocabulary
                
            Returns:
                Modified logits with very negative values for disallowed tokens
            """
            # Create a copy to avoid modifying the original scores
            modified_scores = np.copy(scores)
            
            # Set very negative values for all tokens not in our allowed set
            # This effectively prevents them from being sampled
            for token_id in range(len(modified_scores)):
                if token_id not in allowed_token_ids:
                    modified_scores[token_id] = -1e10
                    
            return modified_scores
            
        # Return the processor and the count of allowed tokens
        return whitelist_logits_processor, len(allowed_token_ids)
    
    def generate_response(self, prompt, max_tokens=50, temperature=0.7):
        """Generate response using only allowed tokens."""
        if not hasattr(self, 'whitelist_processor'):
            raise ValueError("Please create whitelist first with create_whitelist!")
        
        try:
            # Generate with the constraint processor
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=1.1,
                logits_processor=[self.whitelist_processor]
            )
            
            # Return the generated text
            return response['choices'][0]['text']
        except Exception as e:
            return f"Error during generation: {str(e)}"

class MistralInterface:
    def __init__(self, model_path):
        self.root = tk.Tk()
        self.root.title("Mistral Character Whitelist Generator")
        
        # Initialize generator
        self.status_var = tk.StringVar()
        self.status_var.set("Loading model...")
        self.root.update()
        
        self.generator = MistralWhitelistGenerator(model_path)
        self.status_var.set("Model loaded")
        
        # Track selected letters
        self.letter_vars = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI components."""
        # Create frames for organization
        self.letters_frame = ttk.LabelFrame(self.root, text="Allowed Starting Characters")
        self.letters_frame.pack(padx=10, pady=5, fill="x")
        
        # Create letter checkboxes
        self._create_letter_checkboxes()
        
        # Token list creation
        self.token_frame = ttk.LabelFrame(self.root, text="Token Management")
        self.token_frame.pack(padx=10, pady=5, fill="x")
        
        self.token_info_label = ttk.Label(self.token_frame, 
                                        text="Allowed tokens: Not computed")
        self.token_info_label.pack(pady=5)
        
        self.create_list_btn = ttk.Button(self.token_frame, 
                                        text="Create Approved Token List",
                                        command=self._create_token_list)
        self.create_list_btn.pack(pady=5)
        
        # Generation controls
        self.gen_frame = ttk.LabelFrame(self.root, text="Text Generation")
        self.gen_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        ttk.Label(self.gen_frame, text="Enter prompt:").pack(pady=2)
        self.prompt_entry = ttk.Entry(self.gen_frame, width=50)
        self.prompt_entry.pack(pady=2)
        
        self.generate_btn = ttk.Button(self.gen_frame, 
                                     text="Generate Response",
                                     command=self._generate_response)
        self.generate_btn.pack(pady=5)
        
        # Output area
        self.output_text = tk.Text(self.gen_frame, height=10, width=50)
        self.output_text.pack(pady=5, padx=5, fill="both", expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief="sunken")
        self.status_bar.pack(fill="x", padx=5, pady=5)

    def _create_letter_checkboxes(self):
        """Create checkboxes for letters, space, and punctuation."""
        # Create three frames for organization
        frames = []
        for i in range(3):
            frame = ttk.Frame(self.letters_frame)
            frame.pack(side="top", pady=2)
            frames.append(frame)
        
        # Add letter checkboxes (A-Z) but store as lowercase
        for i, letter in enumerate(ascii_lowercase):
            frame_idx = i // 9
            var = tk.BooleanVar(value=True)
            # Store lowercase in vars dict but display uppercase
            self.letter_vars[letter] = var
            
            cb = ttk.Checkbutton(frames[frame_idx], text=letter.upper(), 
                               variable=var)
            cb.pack(side="left", padx=2)
        
        # Add space and basic punctuation in the last row
        special_frame = ttk.Frame(self.letters_frame)
        special_frame.pack(side="top", pady=5)
        
        # Add space
        var = tk.BooleanVar(value=True)
        self.letter_vars[' '] = var
        cb = ttk.Checkbutton(special_frame, text="SPACE", variable=var)
        cb.pack(side="left", padx=5)
        
        # Add basic punctuation
        basic_punct = ",.!?-"
        for punct in basic_punct:
            var = tk.BooleanVar(value=True)
            self.letter_vars[punct] = var
            cb = ttk.Checkbutton(special_frame, text=punct, variable=var)
            cb.pack(side="left", padx=5)

    def _get_selected_letters(self):
        """Get list of currently selected letters."""
        return [letter for letter, var in self.letter_vars.items() 
                if var.get()]

    def _create_token_list(self):
        """Create the allowed token list based on selected letters."""
        selected_letters = self._get_selected_letters()
        if not selected_letters:
            self.status_var.set("Error: Please select at least one letter!")
            return
            
        self.status_var.set("Creating whitelist...")
        self.root.update()
        
        # Create the processor and get allowed count
        processor, allowed_count = self.generator.create_whitelist_processor(selected_letters)
        
        # Store the processor in the generator
        self.generator.whitelist_processor = processor
        
        self.token_info_label.config(
            text=f"Allowed tokens: {allowed_count:,} tokens\n"
                f"Using letters: {', '.join(letter.upper() for letter in sorted(selected_letters))}"
        )
        self.status_var.set("Whitelist created!")

    def _generate_response(self):
        """Generate response with current constraints."""
        if not hasattr(self.generator, 'whitelist_processor'):
            self.status_var.set("Error: Please create whitelist first!")
            return
            
        prompt = self.prompt_entry.get()
        if not prompt:
            self.status_var.set("Error: Please enter a prompt!")
            return
            
        self.status_var.set("Generating response...")
        self.root.update()
        
        try:
            start_time = time.time()
            response = self.generator.generate_response(prompt)
            elapsed = time.time() - start_time
            
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, response)
            
            self.status_var.set(f"Generation completed in {elapsed:.2f} seconds")
        except Exception as e:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Error: {str(e)}")
            self.status_var.set("Error during generation!")

    def run(self):
        """Start the application."""
        self.root.mainloop()

if __name__ == "__main__":
    # Path to your local GGUF model
    model_path = r"C:\Users\abrax\.cache\lm-studio\models\second-state\Mistral-Nemo-Instruct-2407-GGUF\Mistral-Nemo-Instruct-2407-Q5_0.gguf"
    
    app = MistralInterface(model_path)
    app.run()