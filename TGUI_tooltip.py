import tkinter as tk

class ToolTip:
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<Motion>", self.on_motion)

    def on_enter(self, event=None):
        self.schedule_tooltip()

    def on_leave(self, event=None):
        self.cancel_tooltip()
        self.hide_tooltip()

    def on_motion(self, event=None):
        self.x, self.y = event.x, event.y

    def schedule_tooltip(self):
        self.cancel_tooltip()
        self.id = self.widget.after(500, self.show_tooltip)  # 500ms delay

    def cancel_tooltip(self):
        if self.id:
            self.widget.after_cancel(self.id)
        self.id = None

    def show_tooltip(self):
        if self.tipwindow or not self.text:
            return
        
        # Calculate position relative to widget
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        
        # Make tooltip non-interactive
        tw.wm_attributes("-topmost", True)
        if hasattr(tw.wm_attributes, "-disabled"):
            tw.wm_attributes("-disabled", True)
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("Arial", "9", "normal"), padx=4, pady=2)
        label.pack()

    def hide_tooltip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            try:
                tw.destroy()
            except:
                pass