import tkinter as tk
from tkinter import messagebox, ttk
import TGUI_globals as globals
from TGUI_tooltip import ToolTip
from typing import Dict, Any, List, Union, Tuple


class ConstraintsWindow:
    """Window for setting value constraints on columns"""
    
    def __init__(self, parent, column_names: List[str], categorical_columns: List[str]):
        self.parent = parent
        self.column_names = column_names
        self.categorical_columns = categorical_columns
        self.current_theme = globals.CURRENT_THEME
        self.constraints = {}
        self.constraint_widgets = {}
        
        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title("Set Value Constraints")
        self.window.geometry("600x700")
        self.window.configure(bg=self.current_theme['bg'])
        self.window.resizable(width=True, height=True)
        
        self.window.transient(parent)
        self.window.grab_set()
        self.window.focus_set()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(
            self.window,
            text="Set Value Constraints for Columns",
            font=("Arial", 14, 'bold'),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        title_label.pack(pady=10)
        
        # Info label
        info_label = tk.Label(
            self.window,
            text="Set constraints for columns to control generated data ranges",
            font=("Arial", 10),
            bg=self.current_theme['bg'],
            fg=self.current_theme['text']
        )
        info_label.pack(pady=(0, 10))
        
        # Create scrollable frame
        self.create_scrollable_frame()
        
        # Create control buttons
        self.create_control_buttons()
    
    def create_scrollable_frame(self):
        """Create a scrollable frame for constraints"""
        # Main frame with scrollbar
        main_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas and scrollbar
        canvas = tk.Canvas(main_frame, bg=self.current_theme['bg'])
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg=self.current_theme['bg'])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Populate constraints
        self.populate_constraints()
    
    def populate_constraints(self):
        """Populate constraint widgets for each column"""
        for i, col_name in enumerate(self.column_names):
            # Column frame
            col_frame = tk.Frame(self.scrollable_frame, 
                               bg=self.current_theme['bg'], 
                               relief=tk.RIDGE, 
                               bd=1)
            col_frame.pack(fill=tk.X, padx=5, pady=3)
            
            # Column name and type
            col_info = f"{col_name} ({'Categorical' if col_name in self.categorical_columns else 'Numerical'})"
            col_label = tk.Label(col_frame, 
                               text=col_info,
                               font=("Arial", 10, 'bold'),
                               bg=self.current_theme['bg'],
                               fg=self.current_theme['text'])
            col_label.pack(anchor=tk.W, padx=5, pady=2)
            
            # Constraint inputs based on column type
            if col_name in self.categorical_columns:
                self.create_categorical_constraint(col_frame, col_name)
            else:
                self.create_numerical_constraint(col_frame, col_name)
    
    def create_numerical_constraint(self, parent_frame: tk.Frame, col_name: str):
        """Create numerical constraint inputs (min, max)"""
        input_frame = tk.Frame(parent_frame, bg=self.current_theme['bg'])
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Min value
        tk.Label(input_frame, 
                text="Min:", 
                font=("Arial", 9),
                bg=self.current_theme['bg'],
                fg=self.current_theme['text']).pack(side=tk.LEFT)
        
        min_entry = tk.Entry(input_frame, width=12, font=("Arial", 9))
        min_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        # Max value
        tk.Label(input_frame, 
                text="Max:", 
                font=("Arial", 9),
                bg=self.current_theme['bg'],
                fg=self.current_theme['text']).pack(side=tk.LEFT)
        
        max_entry = tk.Entry(input_frame, width=12, font=("Arial", 9))
        max_entry.pack(side=tk.LEFT, padx=5)
        
        # Store widgets
        self.constraint_widgets[col_name] = {
            'type': 'numerical',
            'min_entry': min_entry,
            'max_entry': max_entry
        }
        
        # Tooltips
        ToolTip(min_entry, f"Minimum value for {col_name} (leave empty for no constraint)")
        ToolTip(max_entry, f"Maximum value for {col_name} (leave empty for no constraint)")
    
    def create_categorical_constraint(self, parent_frame: tk.Frame, col_name: str):
        """Create categorical constraint inputs (allowed values)"""
        input_frame = tk.Frame(parent_frame, bg=self.current_theme['bg'])
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(input_frame, 
                text="Allowed values (comma-separated):", 
                font=("Arial", 9),
                bg=self.current_theme['bg'],
                fg=self.current_theme['text']).pack(anchor=tk.W)
        
        values_entry = tk.Entry(input_frame, width=50, font=("Arial", 9))
        values_entry.pack(fill=tk.X, pady=2)
        
        # Store widgets
        self.constraint_widgets[col_name] = {
            'type': 'categorical',
            'values_entry': values_entry
        }
        
        # Tooltip
        ToolTip(values_entry, f"Comma-separated list of allowed values for {col_name} (e.g., '1,2,3' or 'A,B,C'). Leave empty for no constraint.")
    
    def create_control_buttons(self):
        """Create control buttons"""
        button_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
        button_frame.pack(pady=20)
        
        # Clear all button
        clear_button = tk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_all_constraints,
            font=("Arial", 10),
            width=12,
            bg='orange'
        )
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Apply button
        apply_button = tk.Button(
            button_frame,
            text="Apply Constraints",
            command=self.apply_constraints,
            font=("Arial", 12),
            width=15,
            height=2,
            bg=self.current_theme['generate_bg']
        )
        apply_button.pack(side=tk.LEFT, padx=10)
        
        # Cancel button
        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=self.window.destroy,
            font=("Arial", 12),
            width=12,
            height=2,
            bg='red'
        )
        cancel_button.pack(side=tk.LEFT, padx=10)
    
    def clear_all_constraints(self):
        """Clear all constraint inputs"""
        for col_name, widgets in self.constraint_widgets.items():
            if widgets['type'] == 'numerical':
                widgets['min_entry'].delete(0, tk.END)
                widgets['max_entry'].delete(0, tk.END)
            elif widgets['type'] == 'categorical':
                widgets['values_entry'].delete(0, tk.END)
    
    def apply_constraints(self):
        """Parse and apply constraints"""
        try:
            constraints = {}
            
            for col_name, widgets in self.constraint_widgets.items():
                if widgets['type'] == 'numerical':
                    min_val = widgets['min_entry'].get().strip()
                    max_val = widgets['max_entry'].get().strip()
                    
                    # Only add constraint if both values are provided
                    if min_val and max_val:
                        try:
                            min_float = float(min_val)
                            max_float = float(max_val)
                            
                            if min_float >= max_float:
                                messagebox.showerror("Error", 
                                                   f"Invalid range for {col_name}: min ({min_float}) must be less than max ({max_float})")
                                return
                            
                            constraints[col_name] = (min_float, max_float)
                        except ValueError:
                            messagebox.showerror("Error", 
                                               f"Invalid numerical values for {col_name}")
                            return
                
                elif widgets['type'] == 'categorical':
                    values_str = widgets['values_entry'].get().strip()
                    
                    if values_str:
                        # Parse comma-separated values
                        try:
                            values = [val.strip() for val in values_str.split(',') if val.strip()]
                            if values:
                                constraints[col_name] = values
                        except Exception as e:
                            messagebox.showerror("Error", 
                                               f"Error parsing values for {col_name}: {e}")
                            return
            
            # Store constraints and close window
            self.constraints = constraints
            
            if constraints:
                constraint_count = len(constraints)
                messagebox.showinfo("Constraints Applied", 
                                  f"Applied constraints to {constraint_count} column(s)")
            else:
                messagebox.showinfo("No Constraints", "No constraints were set")
            
            self.window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error applying constraints: {e}")
    
    def get_constraints(self) -> Dict[str, Union[Tuple[float, float], List[str]]]:
        """Get the applied constraints"""
        return self.constraints
