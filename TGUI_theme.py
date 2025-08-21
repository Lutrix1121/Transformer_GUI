import tkinter as tk
import TGUI_globals as globals

def toggle_theme(gui, theme_button, widgets):
    """Toggle between light and dark mode"""
    current_bg = gui.cget('bg')
    new_theme = globals.DARK_MODE if current_bg == globals.LIGHT_MODE['bg'] else globals.LIGHT_MODE
    
    globals.CURRENT_THEME = new_theme
    # Update main window
    gui.configure(bg=new_theme['bg'])
    
    # Update theme button text
    theme_button.configure(
        text='     ☀️' if current_bg == globals.LIGHT_MODE['bg'] else '🌙',
        bg=new_theme['button_bg'],
        fg=new_theme['text']
    )

    # Update all widgets
    for widget, widget_type in widgets:
        try:
            if widget.winfo_exists():
                if widget_type == 'label':
                    widget.configure(bg=new_theme['bg'], fg=new_theme['text'])
                elif widget_type == 'frame':
                    widget.configure(bg=new_theme['bg'])
                elif widget_type == 'setup_button':
                    widget.configure(bg=new_theme['button_bg'])
                elif widget_type == 'generate_button':
                    widget.configure(bg=new_theme['generate_bg'])
        except tk.TclError:
            globals.WIDGETS.remove((widget, widget_type))
