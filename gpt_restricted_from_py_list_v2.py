import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tkinter as tk
from tkinter import ttk, filedialog
import time
import importlib.util
import os

class GPT2WordlistGenerator:
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
            
        # Initialize word list and mask
        self.allowed_tokens_mask = None
        self.approved_words = []
        
    def load_wordlist_from_file(self, filepath):
        """Load approved words from a Python file containing without_prefix list."""
        try:
            # Get the directory and filename
            directory = os.path.dirname(os.path.abspath(filepath))
            filename = os.path.basename(filepath)
            module_name = os.path.splitext(filename)[0]
            
            # Setup the module spec
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the without_prefix list
            if hasattr(module, 'without_prefix'):
                self.approved_words = [word.lower() for word in module.without_prefix]
                return True
            else:
                raise AttributeError("File does not contain 'without_prefix' list")
        except Exception as e:
            raise Exception(f"Error loading word list: {str(e)}")

    def create_allowed_tokens_mask(self):
        """Create a mask for tokens that exactly match approved words, plus always allow certain tokens."""
        vocab_size = len(self.tokenizer)
        mask = torch.full((1, vocab_size), float('-inf'), device=self.device)
        allowed_count = 0

        # Special tokens to always allow
        always_allowed_tokens = {" ", ".", "?"}

        # For debugging
        debug_allowed = []

        # Process each token
        for token_id in range(vocab_size):
            token_text = self.tokenizer.decode([token_id])
            if not token_text:
                continue

            # Convert to lowercase for case-insensitive comparison
            token_text = token_text.lower()

            # Remove leading space
            stripped_text = token_text.lstrip()
            if not stripped_text:  # Skip if token is only whitespace
                continue

            # Check if the token is in the always-allowed list or matches exactly an approved word
            if stripped_text in always_allowed_tokens or stripped_text in self.approved_words:
                mask[0, token_id] = 0.0
                allowed_count += 1
                debug_allowed.append((token_id, token_text))

        # Print debug information
        print(f"\nAllowed Tokens Examples (first 20):")
        for token_id, token_text in debug_allowed[:20]:
            print(f"Token {token_id}: '{token_text}'")

        self.allowed_tokens_mask = mask
        return allowed_count

    @torch.no_grad()
    def generate_response(self, prompt, max_length=200, temperature=0.7):
        """Generate response using only allowed tokens."""
        if self.allowed_tokens_mask is None:
            raise ValueError("Please load word list first!")
            
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
        self.root.title("GPT-2 Word List Generator")
        
        # Initialize generator
        self.generator = GPT2WordlistGenerator()
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI components."""
        # File selection
        file_frame = ttk.LabelFrame(self.root, text="Word List File")
        file_frame.pack(padx=10, pady=5, fill="x")
        
        self.file_path_var = tk.StringVar()
        self.file_path_label = ttk.Label(file_frame, 
                                       textvariable=self.file_path_var,
                                       wraplength=400)
        self.file_path_label.pack(pady=5, padx=5, fill="x")
        
        self.select_file_btn = ttk.Button(file_frame, 
                                        text="Select Python File",
                                        command=self._select_file)
        self.select_file_btn.pack(pady=5)
        
        # Token list info
        self.token_frame = ttk.LabelFrame(self.root, text="Token Information")
        self.token_frame.pack(padx=10, pady=5, fill="x")
        
        self.token_info_label = ttk.Label(self.token_frame, 
                                        text="Allowed tokens: Not computed")
        self.token_info_label.pack(pady=5)
        
        self.create_list_btn = ttk.Button(self.token_frame, 
                                        text="Create Token List",
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
        self.status_var.set("Ready - Please select a Python file")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief="sunken")
        self.status_bar.pack(fill="x", padx=5, pady=5)
    
    def _select_file(self):
        """Open file dialog to select Python file."""
        filepath = filedialog.askopenfilename(
            title="Select Python File",
            filetypes=[("Python files", "*.py")]
        )
        if filepath:
            try:
                self.generator.load_wordlist_from_file(filepath)
                self.file_path_var.set(filepath)
                self.status_var.set("Word list loaded successfully")
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                self.file_path_var.set("")

    def _create_token_list(self):
        """Create the allowed token list based on loaded words."""
        if not self.generator.approved_words:
            self.status_var.set("Error: Please load word list first!")
            return
            
        self.status_var.set("Creating token list...")
        self.root.update()
        
        try:
            allowed_count = self.generator.create_allowed_tokens_mask()
            self.token_info_label.config(
                text=f"Allowed tokens: {allowed_count:,}\n"
                     f"Using {len(self.generator.approved_words)} approved words"
            )
            self.status_var.set("Token list created!")
        except Exception as e:
            self.status_var.set(f"Error creating token list: {str(e)}")

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