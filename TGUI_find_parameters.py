import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import TGUI_globals as globals
from TGUI_tooltip import ToolTip
from tuning import tune_hyperparameters


class FindParametersWindow:

    def __init__(self, parent):
        self.parent = parent
        self.current_theme = globals.CURRENT_THEME
        self.entries = {}
        self.search_thread = None
        self.progress_label = None
        self.progress_var = None
        
        # Initialize the backend object immediately
        self.backend = type('Backend', (), {'stop_search': False})()

        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title("Search Parameters")
        self.window.geometry("600x500")
        self.window.configure(bg=self.current_theme['bg'])
        self.window.resizable(width=False, height=False)

        self.window.transient(parent)
        self.window.grab_set()
        self.window.focus_set()
        
        # Handle window close event
        self.window.protocol("WM_DELETE_WINDOW", self.exit_window)

        self.setup_ui()

    # ------------------------------
    # UI Setup
    # ------------------------------
    def setup_ui(self):
        self.create_search_settings()
        self.create_parameter_ranges()
        self.create_control_buttons()
        self.create_progress_section()

    def create_labeled_entry(self, parent, key, label_text, split, font_size=10, default_value="", tooltip=""):
        """Create a labeled entry widget in a 2-column grid"""
        col = split % 2
        row = split // 2

        label = tk.Label(
            parent,
            text=label_text,
            font=("Arial", font_size),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        label.grid(row=row*2, column=col, sticky="w", padx=5, pady=2)

        entry = tk.Entry(parent, font=("Arial", font_size), width=30)
        if default_value:
            entry.insert(0, default_value)
        entry.grid(row=row*2 + 1, column=col, sticky="w", padx=5, pady=2)

        self.entries[key] = entry

        if tooltip:
            ToolTip(entry, tooltip)

        return entry

    def create_search_settings(self):
        """Create the search settings section"""
        frame = tk.LabelFrame(
            self.window, text="Search Settings",
            font=("Arial", 11, 'bold'),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        frame.pack(fill=tk.X, padx=10, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.create_labeled_entry(frame, "epochs", "Epoch count", split=0,
                                  font_size=10, default_value="50",
                                  tooltip="Number of training epochs for each parameter combination")

    def create_parameter_ranges(self):
        """Create the parameter ranges section"""
        frame = tk.LabelFrame(
            self.window, text="Parameter Ranges",
            font=("Arial", 11, 'bold'),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        frame.pack(fill=tk.X, padx=10, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        params = [
            ("d_model", "Model dimensions (comma-separated)", "64,128",
             "Transformer model dimension values for grid search"),
            ("nhead", "Number of heads (comma-separated)", "2,4,8",
             "Number of attention heads"),
            ("num_encoder_layers", "Encoder layers (comma-separated)", "2,4",
             "Number of encoder layers"),
            ("dim_feedforward", "Feedforward dimensions (comma-separated)", "128,256",
             "Dimensions of the feedforward network"),
            ("dropout", "Dropout values (comma-separated)", "0.1,0.2",
             "Dropout rates to test"),
            ("batch_size", "Batch sizes (comma-separated)", "16,32",
             "Training batch sizes"),
            ("learning_rate", "Learning rates (comma-separated)", "0.001,0.005",
             "Learning rates for optimizer"),
            ("max_seq_length", "Max sequence lengths (comma-separated)", "30,50",
             "Maximum sequence lengths for training")
        ]

        for i, (key, label, default, tip) in enumerate(params):
            self.create_labeled_entry(frame, key, label, split=i, font_size=9,
                                      default_value=default, tooltip=tip)

    def create_control_buttons(self):
        """Create control buttons"""
        button_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        button_frame.pack(pady=20)

        self.search_button = tk.Button(
            button_frame,
            text="Start the search",
            command=self.start_search,
            font=("Arial", 12),
            width=20,
            height=2,
            bg=self.current_theme['generate_bg']
        )
        self.search_button.pack(side='left', padx=10)

        self.cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=self.stop_search,
            font=("Arial", 12),
            width=15,
            height=2,
            bg='red'
        )
        self.cancel_button.pack(side='left', padx=10)

    def create_progress_section(self):
        """Create progress bar and label"""
        progress_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        progress_frame.pack(fill=tk.X, padx=10, pady=10)

        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to search parameters",
            font=("Arial", 10),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        self.progress_label.pack()

        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=300,
            mode='determinate'
        )
        progress_bar.pack(pady=5)

    # ------------------------------
    # Functionality
    # ------------------------------
    def update_progress(self, current, total):
        """Update progress bar and label"""
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.progress_label.config(
            text=f"Progress: {current}/{total} trials completed ({progress:.1f}%)"
        )
        self.window.update()

    def parse_parameter_lists(self):
        """Parse and validate all parameter lists"""
        def parse_list(input_str, param_name, value_type):
            if not input_str.strip():
                raise ValueError(f"{param_name} cannot be empty!")
            values = []
            for item in input_str.split(','):
                item = item.strip()
                if item:
                    try:
                        values.append(value_type(item))
                    except ValueError:
                        raise ValueError(f"Invalid {value_type.__name__} value '{item}' in {param_name}")
            return values

        return {
            'd_model': parse_list(self.entries['d_model'].get(), "d_model", int),
            'nhead': parse_list(self.entries['nhead'].get(), "nhead", int),
            'num_encoder_layers': parse_list(self.entries['num_encoder_layers'].get(), "num_encoder_layers", int),
            'dim_feedforward': parse_list(self.entries['dim_feedforward'].get(), "dim_feedforward", int),
            'dropout': parse_list(self.entries['dropout'].get(), "dropout", float),
            'batch_size': parse_list(self.entries['batch_size'].get(), "batch_size", int),
            'learning_rate': parse_list(self.entries['learning_rate'].get(), "learning_rate", float),
            'max_seq_length': parse_list(self.entries['max_seq_length'].get(), "max_seq_length", int),
        }

    def start_search(self):
        """Start parameter search in a thread"""
        if self.search_thread and self.search_thread.is_alive():
            messagebox.showwarning("Warning", "Search already in progress!")
            return

        # Reset the stop flag
        self.backend.stop_search = False
        
        # Update UI state
        self.search_button.config(state='disabled', text="Searching...")
        self.cancel_button.config(text="Stop Search", bg='orange')

        self.search_thread = threading.Thread(target=self.search_parameters_worker)
        self.search_thread.daemon = True
        self.search_thread.start()

    def stop_search(self):
        """Stop the current search"""
        if self.search_thread and self.search_thread.is_alive():
            # Set stop flag
            self.backend.stop_search = True
            self.update_progress_message("Stopping search, please wait...", "orange")
            
            # Start a separate thread to handle the stopping process
            stop_thread = threading.Thread(target=self._handle_stop_process)
            stop_thread.daemon = True
            stop_thread.start()
        else:
            # No search running, just exit
            self.exit_window()

    def _handle_stop_process(self):
        """Handle the stopping process in a separate thread"""
        try:
            # Wait for the search thread to finish (with longer timeout)
            self.search_thread.join(timeout=10.0)
            
            # Update UI on main thread
            self.window.after(0, self._reset_ui_after_stop)
            
        except Exception as e:
            print(f"Error during stop process: {e}")
            self.window.after(0, self._reset_ui_after_stop)

    def _reset_ui_after_stop(self):
        """Reset UI state after stopping (runs on main thread)"""
        self.search_button.config(state='normal', text="Start the search")
        self.cancel_button.config(text="Cancel", bg='red')
        
        if self.backend.stop_search:
            self.update_progress_message("Search stopped by user", "red")
        
        # Reset progress bar
        self.progress_var.set(0)

    def search_parameters_worker(self):
        """Worker function for parameter search"""
        try:
            # Reset stop flag at the beginning
            self.backend.stop_search = False
        
            try:
                if not globals.FILENAME:
                    self.update_progress_message("Error: No data file selected", "red")
                    return
                if not globals.SAVEPATH:
                    self.update_progress_message("Error: No save path selected", "red")
                    return

                epoch_str = self.entries['epochs'].get().strip()
                if not epoch_str:
                    self.update_progress_message("Error: Please enter epoch count", "red")
                    return
                try:
                    tuning_epochs = int(epoch_str)
                except ValueError:
                    self.update_progress_message("Error: Invalid epoch count", "red")
                    return

                params = self.parse_parameter_lists()

                self.update_progress_message("Starting parameter search...", "blue")

                results_csv_path = os.path.join(globals.SAVEPATH, "tuning_results.csv")

                best_params, best_history = tune_hyperparameters(
                    csv_path=globals.FILENAME,
                    param_grid=params,
                    seq_length=None,
                    tuning_epochs=tuning_epochs,
                    categorical_columns=globals.CATEGORICAL_COLUMNS,
                    results_csv=results_csv_path,
                    progress_callback=self.update_progress,
                    stop_flag=lambda: self.backend.stop_search
                )

                if self.backend.stop_search:
                    self.update_progress_message("Search cancelled by user", "red")
                    return

                self.update_progress_message("Search completed!", "green")
                messagebox.showinfo(
                    "Success",
                    f"Best parameters: {best_params}\n"
                    f"Results saved to: {results_csv_path}"
                )
                self.window.destroy()

            except ValueError as e:
                if not self.backend.stop_search:
                    self.update_progress_message(f"Error: {e}", "red")
            except Exception as e:
                if not self.backend.stop_search:
                    self.update_progress_message(f"Error: {e}", "red")
                    
        except Exception as e:
            if not self.backend.stop_search:
                self.update_progress_message(f"Error: {e}", "red")
        finally:
            # Reset UI state when worker finishes
            self.window.after(0, self._reset_ui_after_stop)

    def update_progress_message(self, message, color="black"):
        """Update only the text message above progress bar"""
        def update_ui():
            if self.progress_label and self.progress_label.winfo_exists():
                self.progress_label.config(text=message, fg=color)
                self.window.update()
        
        # Ensure UI update happens on main thread
        if threading.current_thread() == threading.main_thread():
            update_ui()
        else:
            self.window.after(0, update_ui)
    
    def exit_window(self):
        """Handle window exit with running thread cleanup"""
        if self.search_thread and self.search_thread.is_alive():
            response = messagebox.askyesno(
                "Exit",
                "Parameter search is still running.\nDo you want to stop the search and exit?",
                parent=self.window
            )
            if response:
                # Set stop flag
                self.backend.stop_search = True
                self.update_progress_message("Stopping search...", "red")
                
                # Wait for thread to finish with longer timeout
                try:
                    self.search_thread.join(timeout=5.0)
                except:
                    pass
                
                self.window.destroy()
        else:
            response = messagebox.askyesno(
                "Exit",
                "Are you sure you want to exit?",
                parent=self.window
            )
            if response:
                self.window.destroy()


# Legacy function
def findParameters(gui):
    FindParametersWindow(gui)