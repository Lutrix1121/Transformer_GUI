import tkinter as tk
from tkinter import messagebox, ttk
import os
import threading
import TGUI_globals as globals
from TGUI_tooltip import ToolTip
from TGUI_constraints_window import ConstraintsWindow
from TGUI_transformer_backend import TransformerBackend


class TransformerGenerateSamplesWindow:
    """Window for generating samples using transformer model"""
    
    def __init__(self, parent):
        self.parent = parent
        self.current_theme = globals.CURRENT_THEME
        self.entries = {}
        self.constraints = {}
        self.backend = TransformerBackend()
        self.column_names = []
        self.categorical_columns = []
        self.progress_label = None
        self.generation_thread = None
        
        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title("Generate Samples - Transformer Model")
        self.window.geometry("550x900")
        self.window.configure(bg=self.current_theme['bg'])
        self.window.resizable(width=False, height=False)
        
        self.window.transient(parent)
        self.window.grab_set()
        self.window.focus_set()
        
        self.setup_ui()
        
        # Initialize the backend
        self.initialize_backend()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.create_data_settings()
        self.create_generation_settings()
        self.create_model_parameters()
        self.create_constraint_section()
        self.create_control_buttons()
        self.create_progress_section()
    
    def create_data_settings(self):
        """Create data and target settings"""
        # Data settings frame
        data_frame = tk.LabelFrame(
            self.window,
            text="Data Settings",
            font=("Arial", 11, 'bold'),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        data_frame.columnconfigure(0, weight=1)
        data_frame.columnconfigure(1, weight=1)
        
        # Target class column input
        class_entry = self.create_labeled_entry(
            data_frame, "class_column",
            "Target class column (optional)",
            split = 1,
            font_size=10,
            tooltip="Name of the class column for targeted generation"
        )
        
        # Specific class value input
        target_entry = self.create_labeled_entry(
            data_frame, "target_class",
            "Specific class value (optional)",
            split = 2,
            font_size=10,
            tooltip="Specific class value to generate. Leave empty for mixed classes"
        )
    
    def create_generation_settings(self):
        """Create generation settings"""
        # Generation settings frame
        gen_frame = tk.LabelFrame(
            self.window,
            text="Generation Settings",
            font=("Arial", 11, 'bold'),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        gen_frame.pack(fill=tk.X, padx=10, pady=5)
        gen_frame.columnconfigure(0, weight=1)
        gen_frame.columnconfigure(1, weight=1)
        
        # Sample size input
        self.create_labeled_entry(
            gen_frame, "sample_size",
            "Number of samples to generate",
            split = 0,
            font_size=10,
            default_value="1000",
            tooltip="Number of synthetic samples to generate"
        )
        
        # Noise factor
        self.create_labeled_entry(
            gen_frame, "noise_factor",
            "Generation noise factor",
            split = 1,
            font_size=10,
            default_value="0.01",
            tooltip="Noise factor for generation (0.001-0.1). Higher = more variation"
        )
        
        tk.Label(
            gen_frame,
            text="Categorical sampling method:",
            font=("Arial", 10),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        ).grid(row=4, column=0, sticky="w", padx=5, pady=2)
        
        self.sampling_method = tk.StringVar(value="nearest")
        method_combo = ttk.Combobox(
            gen_frame,
            textvariable=self.sampling_method,
            values=["nearest", "probabilistic"],
            state="readonly",
            font=("Arial", 10),
            width=25
        )
        method_combo.grid(row=5, column=0, sticky="w", padx=5, pady=2)
        ToolTip(method_combo, "Method for categorical value selection: 'nearest' (deterministic) or 'probabilistic' (random based on distances)")
    
    def create_model_parameters(self):
        """Create model parameter settings"""
        # Model parameters frame
        model_frame = tk.LabelFrame(
            self.window,
            text="Model Parameters",
            font=("Arial", 11, 'bold'),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        model_frame.columnconfigure(0, weight=1)
        model_frame.columnconfigure(1, weight=1)
        
        # Training parameters
        training_params = [
            ("epochs", "Training epochs", "50", "Number of training epochs"),
            ("batch_size", "Batch size", "32", "Batch size for training"),
            ("learning_rate", "Learning rate", "0.001", "Learning rate for optimizer"),
            ("seq_length", "Sequence length", "30", "Sequence length for transformer")
        ]
        
        # Architecture parameters
        arch_params = [
            ("d_model", "Model dimension", "64", "Transformer model dimension"),
            ("nhead", "Number of heads", "4", "Number of attention heads"),
            ("num_layers", "Number of layers", "2", "Number of transformer layers"),
            ("dim_feedforward", "Feedforward dimension", "128", "Dimension of feedforward network")
        ]
        all_params = training_params + arch_params

        for i, (param_key, label_text, default_val, tooltip_text) in enumerate(all_params):
            entry = self.create_labeled_entry(
                model_frame, param_key, label_text, split = i,
                font_size=9, default_value=default_val
            )
            ToolTip(entry, tooltip_text)
    
    def create_constraint_section(self):
        """Create constraints section"""
        # Constraints frame
        constraints_frame = tk.LabelFrame(
            self.window,
            text="Value Constraints",
            font=("Arial", 11, 'bold'),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        constraints_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Info label
        info_label = tk.Label(
            constraints_frame,
            text="Set constraints to control the ranges/values of generated data",
            font=("Arial", 9),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        info_label.pack(pady=5)
        
        # Constraints button
        self.constraints_button = tk.Button(
            constraints_frame,
            text="Set Constraints",
            command=self.open_constraints_window,
            font=("Arial", 10),
            width=20,
            bg=self.current_theme['button_bg'],
            state=tk.DISABLED  # Initially disabled until data is loaded
        )
        self.constraints_button.pack(pady=5)
        
        # Constraints status
        self.constraints_status = tk.Label(
            constraints_frame,
            text="No constraints set",
            font=("Arial", 9),
            bg=self.current_theme['bg'],
            fg='gray'
        )
        self.constraints_status.pack(pady=2)
    
    def create_control_buttons(self):
        """Create control buttons"""
        button_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        button_frame.pack(pady=10)
        
        # Train new model checkbox
        self.train_new_model = tk.BooleanVar(value=True)
        train_checkbox = tk.Checkbutton(
            button_frame,
            text="Train new model (uncheck to use existing)",
            variable=self.train_new_model,
            font=("Arial", 10),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text'],
            selectcolor=self.current_theme['bg']
        )
        train_checkbox.pack(pady=5)
        
        # Buttons row
        buttons_row = tk.Frame(button_frame, bg=self.current_theme['bg'])
        buttons_row.pack(pady=10)
        
        generate_button = tk.Button(
            buttons_row,
            text="Generate Samples",
            command=self.start_generation,
            font=("Arial", 12),
            width=20,
            height=2,
            bg=self.current_theme['generate_bg']
        )
        generate_button.pack(side='left', padx=10)
        
        cancel_button = tk.Button(
            buttons_row,
            text="Cancel",
            command=self.cancel_generation,
            font=("Arial", 12),
            width=15,
            height=2,
            bg='red'
        )
        cancel_button.pack(side='left', padx=10)
    
    def create_progress_section(self):
        """Create progress display section"""
        progress_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        progress_frame.pack(fill=tk.X, padx=50, pady=10)
        
        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to generate samples",
            font=("Arial", 10),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        self.progress_label.pack()
    
    def create_labeled_entry(self, parent, key, label_text, split, font_size=10, default_value="", tooltip=""):
        """Create a labeled entry widget"""
        col = split%2
        row = split//2
        
        label = tk.Label(
            parent,
            text=label_text,
            font=("Arial", font_size),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        label.grid(row=row*2, column=col, sticky="w", padx=5, pady=2)
        
        entry = tk.Entry(parent, font=("Arial", font_size), width=35)
        if default_value:
            entry.insert(0, default_value)
        entry.grid(row=row*2+1, column=col, sticky="w", padx=5, pady=2)
        
        self.entries[key] = entry
        
        if tooltip:
            ToolTip(entry, tooltip)
        
        return entry
    
    def initialize_backend(self):
        """Initialize the transformer backend with current data"""
        if not globals.FILENAME:
            self.update_progress("No data file selected", "red")
            return
        
        self.update_progress("Initializing transformer backend...", "blue")        

        categorical_columns = globals.CATEGORICAL_COLUMNS
        
        # Create basic model config for initialization
        model_config = {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 2,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'max_seq_length': 30,
            'batch_size': 32,
            'learning_rate': 0.001,
            'generation_noise_factor': 0.01,
            'categorical_sampling_method': 'nearest'
        }
        
        try:
            self.column_names, success = self.backend.initialize_generator(
                globals.FILENAME,
                categorical_columns,
                model_config
            )
            
            if success:
                self.categorical_columns = self.backend.categorical_columns
                self.constraints_button.config(state=tk.NORMAL)
                self.update_progress("Backend initialized successfully", "green")
            else:
                self.update_progress("Failed to initialize backend", "red")
                
        except Exception as e:
            self.update_progress(f"Error initializing backend: {e}", "red")
    
    def open_constraints_window(self):
        """Open the constraints configuration window"""
        if not self.column_names:
            messagebox.showwarning("Warning", "No data loaded. Please check your data file.")
            return
        
        constraints_window = ConstraintsWindow(
            self.window,
            self.column_names,
            self.categorical_columns
        )
        
        # Wait for window to close and get constraints
        self.window.wait_window(constraints_window.window)
        
        # Update constraints
        self.constraints = constraints_window.get_constraints()
        
        # Update status
        if self.constraints:
            constraint_count = len(self.constraints)
            self.constraints_status.config(
                text=f"Constraints set for {constraint_count} column(s)",
                fg='green'
            )
        else:
            self.constraints_status.config(
                text="No constraints set",
                fg='gray'
            )
    
    def get_model_config(self):
        """Get model configuration from UI inputs"""
        try:
            config = {
                'd_model': int(self.entries['d_model'].get() or 64),
                'nhead': int(self.entries['nhead'].get() or 4),
                'num_encoder_layers': int(self.entries['num_layers'].get() or 2),
                'dim_feedforward': int(self.entries['dim_feedforward'].get() or 128),
                'dropout': 0.1,
                'max_seq_length': int(self.entries['seq_length'].get() or 30),
                'batch_size': int(self.entries['batch_size'].get() or 32),
                'learning_rate': float(self.entries['learning_rate'].get() or 0.001),
                'epochs': int(self.entries['epochs'].get() or 50),
                'generation_noise_factor': float(self.entries['noise_factor'].get() or 0.01),
                'categorical_sampling_method': self.sampling_method.get()
            }
            return config
        except ValueError as e:
            raise ValueError(f"Invalid parameter value: {e}")
    
    def update_progress(self, message, color="black"):
        """Update progress message"""
        if self.progress_label:
            self.progress_label.config(text=message, fg=color)
            self.window.update()
    
    def start_generation(self):
        """Start the sample generation process"""
        if self.generation_thread and self.generation_thread.is_alive():
            messagebox.showwarning("Warning", "Generation already in progress!")
            return
        
        # Start generation in separate thread
        self.generation_thread = threading.Thread(target=self.generate_samples_worker)
        self.generation_thread.daemon = True
        self.generation_thread.start()
    
    def generate_samples_worker(self):
        """Worker method for sample generation (runs in separate thread)"""
        try:
            # Validate inputs
            if not globals.FILENAME:
                self.update_progress("Error: No data file selected", "red")
                return
            
            if not globals.SAVEPATH:
                self.update_progress("Error: No save path selected", "red")
                return
            
            # Get parameters
            sample_size = int(self.entries['sample_size'].get())
            if sample_size <= 0:
                self.update_progress("Error: Sample size must be positive", "red")
                return
            
            # Get categorical columns
            categorical_columns = globals.CATEGORICAL_COLUMNS
            
            # Get model configuration
            model_config = self.get_model_config()
            
            # Initialize or reinitialize backend if needed
            self.update_progress("Setting up transformer model...", "blue")
            
            self.column_names, success = self.backend.initialize_generator(
                globals.FILENAME,
                categorical_columns,
                model_config
            )
            
            if not success:
                self.update_progress("Error: Failed to initialize generator", "red")
                return
            
            # Model path
            model_filename = "tabular_transformer_model.pth"
            model_path = os.path.join(globals.SAVEPATH, model_filename)
            
            # Train or load model
            if self.train_new_model.get() or not os.path.exists(model_path):
                self.update_progress("Training transformer model...", "blue")
                
                success = self.backend.train_model(
                    epochs=model_config['epochs'],
                    batch_size=model_config['batch_size'],
                    save_path=model_path
                )
                
                if not success:
                    self.update_progress("Error: Failed to train model", "red")
                    return
                
                self.update_progress("Model trained successfully", "green")
            else:
                self.update_progress("Loading existing model...", "blue")
                
                success = self.backend.load_existing_model(model_path)
                
                if not success:
                    self.update_progress("Error: Failed to load model", "red")
                    return
                
                self.update_progress("Model loaded successfully", "green")
            
            # Generate samples
            self.update_progress("Generating synthetic samples...", "blue")
            
            class_column = self.entries['class_column'].get().strip()
            target_class = self.entries['target_class'].get().strip()
            
            if class_column and target_class:
                # Generate with specific class
                generated_samples = self.backend.generate_samples(
                    num_samples=sample_size,
                    target_class=target_class,
                    target_class_column=class_column,
                    value_constraints=self.constraints,
                    generation_noise_factor=model_config['generation_noise_factor'],
                    categorical_sampling_method=model_config['categorical_sampling_method']
                )
            else:
                # Generate mixed samples
                generated_samples = self.backend.generate_samples(
                    num_samples=sample_size,
                    value_constraints=self.constraints,
                    generation_noise_factor=model_config['generation_noise_factor'],
                    categorical_sampling_method=model_config['categorical_sampling_method']
                )
            
            # Save samples
            self.update_progress("Saving generated samples...", "blue")
            
            output_filename = f"Generated_Transformer_Samples_{sample_size}.csv"
            output_path = os.path.join(globals.SAVEPATH, output_filename)
            
            success = self.backend.save_samples(generated_samples, output_path)
            
            if success:
                self.update_progress(f"Generation completed! Saved {len(generated_samples)} samples", "green")
                
                # Show success message in main thread
                self.window.after(0, lambda: messagebox.showinfo(
                    "Success", 
                    f"Generated {len(generated_samples)} samples successfully!\nSaved to: {output_path}"
                ))
            else:
                self.update_progress("Error: Failed to save samples", "red")
        
        except ValueError as e:
            self.update_progress(f"Error: Invalid input - {e}", "red")
        except Exception as e:
            self.update_progress(f"Error: {e}", "red")
    
    def cancel_generation(self):
        """Cancel generation and close window"""
        # Note: Due to threading limitations, we can't easily stop the generation mid-process
        # This mainly serves to close the window
        if self.generation_thread and self.generation_thread.is_alive():
            response = messagebox.askyesno(
                "Cancel Generation", 
                "Generation is in progress. Closing this window won't stop the process.\nDo you want to close anyway?"
            )
            if not response:
                return
        
        self.window.destroy()


# Legacy function for backward compatibility
def generateSamples(gui):
    """Legacy function for backward compatibility"""
    TransformerGenerateSamplesWindow(gui)
            