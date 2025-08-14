import tkinter as tk
from tkinter import messagebox
import logging
from TGUI_tooltip import ToolTip
from TGUI_setup_paths import SetupPathsWindow
from TGUI_generate_samples import TransformerGenerateSamplesWindow
from TGUI_find_parameters import FindParametersWindow
import TGUI_globals as globals
from TGUI_theme import toggle_theme

# Configure basic logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseWindow:
    """Base class for all GUI windows with common functionality"""
    
    def __init__(self, parent=None, title="Window", geometry="600x400"):
        self.parent = parent
        self.window = None
        
        try:
            self.current_theme = globals.CURRENT_THEME
        except (AttributeError, NameError):
            # Fallback theme if globals not available
            self.current_theme = {'bg': '#ffffff', 'text': '#000000', 'button_bg': '#e0e0e0'}
            logging.warning("Could not load theme from globals, using fallback")
        
        self._create_window(parent, title, geometry)
    
    def _create_window(self, parent, title, geometry):
        try:
            if parent:
                self.window = tk.Toplevel(parent)
                self.window.transient(parent)
                self.window.grab_set()
                self.window.focus_set()
            else:
                self.window = tk.Tk()
                
            self.window.title(title)
            self.window.geometry(geometry)
            self.window.configure(bg=self.current_theme['bg'])
            self.window.resizable(width=False, height=False)
            
        except tk.TclError as e:
            logging.error(f"Failed to create window: {e}")
            if hasattr(self, 'window') and self.window:
                self.window.destroy()
            raise
    
    def center_window(self):
        try:
            self.window.update_idletasks()
            x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
            y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
            self.window.geometry(f"+{x}+{y}")
        except tk.TclError as e:
            logging.error(f"Failed to center window: {e}")
            # Window will just stay at default position
    
    def create_button_frame(self, buttons_config):
        try:
            button_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
            button_frame.pack(pady=20)
            
            for config in buttons_config:
                if not isinstance(config, dict) or 'text' not in config or 'command' not in config:
                    logging.warning(f"Invalid button configuration: {config}")
                    continue
                    
                button = tk.Button(
                    button_frame,
                    text=config['text'],
                    command=config['command'],
                    font=config.get('font', ("Arial", 12)),
                    width=config.get('width', 15),
                    height=config.get('height', 2),
                    bg=config.get('bg', self.current_theme['button_bg'])
                )
                button.pack(side='left', padx=10)
            
            return button_frame
            
        except tk.TclError as e:
            logging.error(f"Failed to create button frame: {e}")
            return None


class MainGUI(BaseWindow):   
    def __init__(self):
        try:
            super().__init__(None, "Transformer synthetic data generator", "800x600")
            self.file_label = None
            self.path_label = None
            self.theme_button = None
            self.setup_ui()
        except Exception as e:
            logging.error(f"Failed to initialize MainGUI: {e}")
            messagebox.showerror("Initialization Error", 
                               f"Failed to start the application:\n{str(e)}")
            raise
    
    def setup_ui(self):
        try:
            self.create_theme_button()
            self.create_title()
            self.create_setup_section()
            self.create_action_buttons()
            self.create_terminate_button()
        except Exception as e:
            logging.error(f"Failed to setup UI: {e}")
            messagebox.showerror("UI Setup Error", 
                               f"Failed to create user interface:\n{str(e)}")
    
    def create_theme_button(self):
        try:
            # Check if globals has the required theme data
            light_mode = getattr(globals, 'LIGHT_MODE', {'button_bg': '#e0e0e0', 'text': '#000000'})
            
            self.theme_button = tk.Button(
                self.window,
                text='🌙',
                font=("Arial", 12),
                width=3,
                bg=light_mode['button_bg'],
                fg=light_mode['text'],
                border=0,
                command=self.toggle_theme
            )
            self.theme_button.place(relx=0.97, rely=0.02, anchor='ne')
            
            # Only add tooltip if ToolTip is available
            try:
                ToolTip(self.theme_button, "Toggle Light/Dark mode")
            except:
                logging.warning("ToolTip not available for theme button")
                
        except Exception as e:
            logging.error(f"Failed to create theme button: {e}")
    
    def create_title(self):
        try:
            title_label = tk.Label(
                self.window, 
                text="Transformer Synthetic Data Generator",
                font=("Arial", 24), 
                bg=self.current_theme['bg'], 
                fg=self.current_theme['text']
            )
            title_label.pack(pady=30)
            
            # Only add to globals if it exists
            if hasattr(globals, 'WIDGETS'):
                globals.WIDGETS.append((title_label, 'label'))
                
        except Exception as e:
            logging.error(f"Failed to create title: {e}")
    
    def create_setup_section(self):
        try:
            setup_frame = tk.Frame(self.window, bg=self.current_theme['bg'])
            setup_frame.pack(pady=25)
            
            if hasattr(globals, 'WIDGETS'):
                globals.WIDGETS.append((setup_frame, 'frame'))
            
            # Setup button
            setup_button = tk.Button(
                setup_frame, 
                text="Setup Paths",
                command=self.open_setup_paths,
                font=("Arial", 16), 
                width=20, 
                height=2, 
                bg=self.current_theme['button_bg']
            )
            setup_button.pack(pady=5)
            
            try:
                ToolTip(setup_button, "Define data file path and save path")
            except:
                logging.warning("ToolTip not available for setup button")
                
            if hasattr(globals, 'WIDGETS'):
                globals.WIDGETS.append((setup_button, 'setup_button'))
            
            # Path and file labels
            self.path_label = tk.Label(
                setup_frame, 
                text="No save path selected",
                font=("Arial", 10), 
                bg=self.current_theme['bg'],
                fg=self.current_theme['text']
            )
            self.path_label.pack(pady=5)
            
            self.file_label = tk.Label(
                setup_frame, 
                text="No file selected",
                font=("Arial", 10), 
                bg=self.current_theme['bg'],
                fg=self.current_theme['text']
            )
            self.file_label.pack(pady=5)
            
            if hasattr(globals, 'WIDGETS'):
                globals.WIDGETS.extend([(self.path_label, 'label'), (self.file_label, 'label')])
                
        except Exception as e:
            logging.error(f"Failed to create setup section: {e}")
    
    def create_action_buttons(self):
        try:
            # Generate samples button
            generate_button = tk.Button(
                self.window, 
                text="Generate Samples",
                command=self.open_generate_samples,
                font=("Arial", 16), 
                width=20, 
                height=2, 
                bg=self.current_theme.get('generate_bg', self.current_theme['button_bg'])
            )
            generate_button.pack(pady=10)
            
            try:
                ToolTip(generate_button, "Run the GAN to generate synthetic samples based on the selected data file and parameters")
            except:
                logging.warning("ToolTip not available for generate button")
            
            # Find parameters button
            find_params_button = tk.Button(
                self.window, 
                text="Find Parameters",
                command=self.open_find_parameters,
                font=("Arial", 16), 
                width=20, 
                height=2, 
                bg=self.current_theme.get('generate_bg', self.current_theme['button_bg'])
            )
            find_params_button.pack(pady=10)
            
            try:
                ToolTip(find_params_button, "Do the grid or random search for GAN parameters to optimize the model performance based on the lowest discriminator or generator loss saved in the results directory")
            except:
                logging.warning("ToolTip not available for find parameters button")
            
            if hasattr(globals, 'WIDGETS'):
                globals.WIDGETS.extend([(generate_button, 'generate_button'), (find_params_button, 'generate_button')])
                
        except Exception as e:
            logging.error(f"Failed to create action buttons: {e}")
    
    def create_terminate_button(self):
        try:
            terminate_button = tk.Button(
                self.window, 
                text="End Program",
                command=self.safe_destroy,
                font=("Arial", 16), 
                width=20, 
                height=2, 
                bg='red'
            )
            terminate_button.pack(pady=10)
        except Exception as e:
            logging.error(f"Failed to create terminate button: {e}")
    
    def safe_destroy(self):
        try:
            self.window.destroy()
        except:
            # Force exit if normal destroy fails
            import sys
            sys.exit(0)
    
    def toggle_theme(self):
        try:
            widgets = getattr(globals, 'WIDGETS', [])
            toggle_theme(self.window, self.theme_button, widgets=widgets)
            self.current_theme = getattr(globals, 'CURRENT_THEME', self.current_theme)
        except Exception as e:
            logging.error(f"Failed to toggle theme: {e}")
            messagebox.showwarning("Theme Error", "Could not change theme")
    
    def open_setup_paths(self):
        try:
            SetupPathsWindow(self.window, self.file_label, self.path_label)
        except Exception as e:
            logging.error(f"Failed to open setup paths window: {e}")
            messagebox.showerror("Window Error", "Could not open setup paths window")
    
    def open_generate_samples(self):
        try:
            TransformerGenerateSamplesWindow(self.window)
        except Exception as e:
            logging.error(f"Failed to open generate samples window: {e}")
            messagebox.showerror("Window Error", "Could not open generate samples window")
    
    def open_find_parameters(self):
        try:
            FindParametersWindow(self.window)
        except Exception as e:
            logging.error(f"Failed to open find parameters window: {e}")
            messagebox.showerror("Window Error", "Could not open find parameters window")
    
    def run(self):
        try:
            self.window.mainloop()
        except KeyboardInterrupt:
            logging.info("Application interrupted by user")
            self.safe_destroy()
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            messagebox.showerror("Application Error", f"An unexpected error occurred:\n{str(e)}")


def create_main_gui():
    try:
        app = MainGUI()
        app.run()
    except Exception as e:
        logging.critical(f"Failed to create main GUI: {e}")
        try:
            messagebox.showerror("Critical Error", 
                               f"Failed to start the application:\n{str(e)}\n\nCheck the logs for more details.")
        except:
            print(f"Critical Error: {e}")


if __name__ == "__main__":
    create_main_gui()