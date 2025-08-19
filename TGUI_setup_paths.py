import tkinter as tk
from tkinter import filedialog, messagebox
import os
import TGUI_globals as globals


class SetupPathsWindow:
    """Window for setting up file paths and configurations"""
    
    def __init__(self, parent, file_label, path_label):
        self.parent = parent
        self.file_label = file_label
        self.path_label = path_label
        self.current_theme = globals.CURRENT_THEME
        
        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title("Setup File and Save Path")
        self.window.geometry("1300x700")
        self.window.configure(bg=self.current_theme['bg'])
        self.window.resizable(width=False, height=False)
        
        self.window.transient(parent)
        self.window.grab_set()
        self.window.focus_set()
        
        # Initialize UI elements
        self.file_status_label = None
        self.path_status_label = None
        self.integer_entry = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(
            self.window, 
            text="Configure Data File and Save Location",
            font=("Arial", 16, "bold"), 
            bg=self.current_theme['bg'], 
            fg=self.current_theme['text']
        )
        title_label.pack(pady=20)
        globals.WIDGETS.append((title_label, 'label'))
        
        self.create_file_section()
        self.create_categorical_columns_section()
        self.create_save_path_section()
        self.create_control_buttons()
    
    def create_file_section(self):
        file_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        file_frame.pack(pady=20, padx=20, fill='x')
        
        file_section_label = tk.Label(
            file_frame, 
            text="1. Select Data File",
            font=("Arial", 12, "bold"), 
            bg=self.current_theme['bg'], 
            fg=self.current_theme['text']
        )
        file_section_label.pack(anchor='w', pady=(0, 10))
        
        file_button = tk.Button(
            file_frame, 
            text="Browse for Data File",
            command=self.select_file,
            font=("Arial", 12), 
            width=20, 
            height=2, 
            bg=self.current_theme['button_bg']
        )
        file_button.pack(pady=5)
        
        self.file_status_label = tk.Label(
            file_frame, 
            text="No file selected",
            font=("Arial", 10), 
            bg=self.current_theme['bg'], 
            fg='red'
        )
        self.file_status_label.pack(pady=5)
    
    def create_categorical_columns_section(self):
        categorical_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        categorical_frame.pack(pady=20, padx=20, fill='x')
        
        categorical_section_label = tk.Label(
            categorical_frame, 
            text="2. Specify Categorical Columns (Optional)",
            font=("Arial", 12, "bold"), 
            bg=self.current_theme['bg'], 
            fg=self.current_theme['text']
        )
        categorical_section_label.pack(anchor='w', pady=(0, 10))
        
        categorical_info_label = tk.Label(
            categorical_frame,
            text="Enter column names that should be treated as categorical, separated by commas:",
            font=("Arial", 10), 
            bg=self.current_theme['bg'], 
            fg=self.current_theme['text']
        )
        categorical_info_label.pack(anchor='w', pady=(0, 5))
        
        self.categorical_entry = tk.Entry(categorical_frame, font=("Arial", 10), width=60)
        self.categorical_entry.pack(pady=5, fill='x')
    
    def create_save_path_section(self):
        path_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        path_frame.pack(pady=20, padx=20, fill='x')
        
        path_section_label = tk.Label(
            path_frame, 
            text="3. Select Save Location",
            font=("Arial", 12, "bold"), 
            bg=self.current_theme['bg'], 
            fg=self.current_theme['text']
        )
        path_section_label.pack(anchor='w', pady=(0, 10))
        
        path_button = tk.Button(
            path_frame, 
            text="Browse for Save Location",
            command=self.select_save_path,
            font=("Arial", 12), 
            width=20, 
            height=2, 
            bg=self.current_theme['generate_bg']
        )
        path_button.pack(pady=5)
        
        self.path_status_label = tk.Label(
            path_frame, 
            text="No save path selected",
            font=("Arial", 10), 
            bg=self.current_theme['bg'], 
            fg='red'
        )
        self.path_status_label.pack(pady=5)
    
    def create_control_buttons(self):
        button_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        button_frame.pack(pady=30)
        
        done_button = tk.Button(
            button_frame, 
            text="Done",
            command=self.close_setup,
            font=("Arial", 12), 
            width=15, 
            height=2, 
            bg=self.current_theme['generate_bg']
        )
        done_button.pack(side='left', padx=10)
        
        cancel_button = tk.Button(
            button_frame, 
            text="Cancel",
            command=self.window.destroy,
            font=("Arial", 12), 
            width=15, 
            height=2, 
            bg='red'
        )
        cancel_button.pack(side='left', padx=10)
    
    def select_file(self):
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            globals.FILENAME = filename
            self.file_status_label.config(
                text=f"Selected: {os.path.basename(globals.FILENAME)}", 
                fg='green'
            )
            self.file_label.config(text=f"Selected: {os.path.basename(globals.FILENAME)}")
    
    def select_save_path(self):
        savepath = filedialog.askdirectory(title="Select Save Location")
        if savepath:
            globals.SAVEPATH = os.path.join(savepath + "/tuning_results")
            if not os.path.exists(globals.SAVEPATH):
                os.makedirs(globals.SAVEPATH)
            self.path_status_label.config(text=f"Save to: {globals.SAVEPATH}", fg='green')
            self.path_label.config(text=f"Save to: {globals.SAVEPATH}")
    
    def close_setup(self):
        """Close setup window and save configuration"""
        # Process integer columns input
        categorical_input = self.categorical_entry.get().strip()
        if categorical_input:
            globals.CATEGORICAL_COLUMNS = [col.strip() for col in categorical_input.split(',') if col.strip()]
        else:
            globals.CATEGORICAL_COLUMNS = []
        
        if globals.FILENAME and globals.SAVEPATH:
            messagebox.showinfo(
                "Setup Complete",
                f"Configuration saved!\n\n"
                f"Data file: {os.path.basename(globals.FILENAME)}\n"
                f"Save location: {globals.SAVEPATH}\n"
                f"Integer columns: {', '.join(globals.CATEGORICAL_COLUMNS) if globals.CATEGORICAL_COLUMNS else 'None'}"
            )
            self.window.destroy()
        else:
            response = messagebox.askyesno(
                "Incomplete Setup",
                "You haven't selected both file and save path. Close anyway?"
            )
            if response:
                self.window.destroy()


# Legacy functions for backward compatibility
def setupPaths(gui, file_label, path_label):
    """Legacy function for backward compatibility"""
    SetupPathsWindow(gui, file_label, path_label)

def selectFile(status_label, file_label):
    """Legacy function for backward compatibility"""
    filename = filedialog.askopenfilename(
        title="Select Data File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if filename:
        globals.FILENAME = filename
        status_label.config(text=f"Selected: {os.path.basename(globals.FILENAME)}", fg='green')
        file_label.config(text=f"Selected: {os.path.basename(globals.FILENAME)}")

def selectSavePath(status_label, path_label):
    """Legacy function for backward compatibility"""
    savepath = filedialog.askdirectory(title="Select Save Location")
    if savepath:
        globals.SAVEPATH = savepath
        status_label.config(text=f"Save to: {globals.SAVEPATH}", fg='lightgreen')
        path_label.config(text=f"Save to: {globals.SAVEPATH}")

def closeSetup(window, integer_entry):
    """Legacy function for backward compatibility"""
    # Process integer columns input
    categorical_input = integer_entry.get().strip()
    if categorical_input:
        globals.CATEGORICAL_COLUMNS = [col.strip() for col in categorical_input.split(',') if col.strip()]
    else:
        globals.CATEGORICAL_COLUMNS = []
    
    if globals.FILENAME and globals.SAVEPATH:
        integer_info = f" (Integer columns: {len(globals.CATEGORICAL_COLUMNS)})" if globals.CATEGORICAL_COLUMNS else " (No integer columns specified)"
        messagebox.showinfo("Setup Complete", 
                           f"Configuration saved!\n\n"
                           f"Data file: {os.path.basename(globals.FILENAME)}\n"
                           f"Save location: {globals.SAVEPATH}\n"
                           f"Integer columns: {', '.join(globals.CATEGORICAL_COLUMNS) if globals.CATEGORICAL_COLUMNS else 'None'}")
        window.destroy()
    else:
        response = messagebox.askyesno("Incomplete Setup", 
                                     "You haven't selected both file and save path. Close anyway?")
        if response:
            window.destroy()