import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tkinter as tk
from tkinter import ttk
from string import ascii_lowercase, punctuation
import time

class GPT2WhitelistGenerator:
    def __init__(self, model_name="gpt2"):
        # Initialize model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer settings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize token cache
        self.token_first_chars = None
        self.allowed_tokens_mask = None
        self.approved_chars = set()
        
        # Pre-compute token information
        self._compute_token_first_chars()

    def _compute_token_first_chars(self):
        """Pre-compute the first and second characters of each token."""
        vocab_size = len(self.tokenizer)
        self.token_first_chars = {}
        self.token_second_chars = {}
        
        for token_id in range(vocab_size):
            token_text = self.tokenizer.decode([token_id])
            if token_text:
                # Store all characters as lowercase for case-insensitive comparison
                first_char = token_text[0].lower()
                self.token_first_chars[token_id] = first_char
                
                # Store second character if token starts with space and has length > 1
                if first_char == ' ' and len(token_text) > 1:
                    self.token_second_chars[token_id] = token_text[1].lower()

    def create_allowed_tokens_mask(self, approved_chars):
        """Create a mask for tokens that start with approved characters."""
        self.approved_chars = set(char.lower() for char in approved_chars)
        vocab_size = len(self.tokenizer)
        
        # Create the mask on device
        mask = torch.full((1, vocab_size), float('-inf'), device=self.device)
        
        # Count of allowed tokens
        allowed_count = 0
        
        # Check each token
        for token_id in range(vocab_size):
            # Get first character
            first_char = self.token_first_chars.get(token_id)
            if first_char is None:
                continue
                
            # Check if token should be allowed
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
                mask[0, token_id] = 0.0
                allowed_count += 1
        
        self.allowed_tokens_mask = mask
        return allowed_count

    @torch.no_grad()
    def generate_response(self, prompt, max_length=50, temperature=0.7):
        """Generate response using only allowed tokens."""
        if self.allowed_tokens_mask is None:
            raise ValueError("Please create allowed token list first!")
            
        # Encode input
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        def custom_logits_processor(input_ids, scores):
            scores = scores + self.allowed_tokens_mask
            return torch.nan_to_num(scores, nan=float('-inf'), 
                                  posinf=0, neginf=float('-inf'))

        # Generate with custom processor
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            logits_processor=[custom_logits_processor],
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class GPT2Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPT-2 Character Whitelist Generator")
        
        # Initialize generator
        self.generator = GPT2WhitelistGenerator()
        
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
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
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
            
        self.status_var.set("Creating token list...")
        self.root.update()
        
        # Create the mask
        allowed_count = self.generator.create_allowed_tokens_mask(selected_letters)
        
        self.token_info_label.config(
            text=f"Allowed tokens: {allowed_count:,} tokens\n"
                f"Using letters: {', '.join(letter.upper() for letter in sorted(selected_letters))}"
        )
        self.status_var.set("Token list created!")

    def _generate_response(self):
        """Generate response with current constraints."""
        if self.generator.allowed_tokens_mask is None:
            self.status_var.set("Error: Please create token list first!")
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
    app = GPT2Interface()
    app.run()