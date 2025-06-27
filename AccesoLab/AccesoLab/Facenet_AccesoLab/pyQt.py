# Import the required libraries
from tkinter import *
from tkinter import ttk

# Create an instance of tkinter frame or window
win=Tk()

# Set the size of the window
win.geometry("700x350")

# Remove the Title bar of the window
win.overrideredirect(True)

# Define a function for resizing the window
def moveMouseButton(e):
   x1=winfo_pointerx()
   y1=winfo_pointery()
   x0=winfo_rootx()
   y0=winfo_rooty()

   win.geometry("%s x %s" % ((x1-x0),(y1-y0)))

# Add a Label widget
# label=Label(win,text="Grab the lower-right corner to resize the window")
# label.pack(side="top", fill="both", expand=True)

# Add the gripper for resizing the window
grip=ttk.Sizegrip()
grip.place(relx=1.0, rely=1.0, anchor="se")
# grip.lift(label)
grip.bind("<B1-Motion>", moveMouseButton)

win.mainloop()