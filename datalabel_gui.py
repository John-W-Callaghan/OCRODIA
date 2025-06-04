import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk, ImageGrab, ImageOps

class DrawingDatasetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Odia Character Drawing Tool")

        self.pen_color = "white"
        self.bg_color = "black"
        self.brush_size = 5
        self.current_label = 0
        self.image_index = 0
        self.total_labels = 47 

        # Canvas
        self.canvas = tk.Canvas(root, width=280, height=280, bg=self.bg_color, cursor="cross")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Reference image
        self.preview_label = tk.Label(root)
        self.preview_label.grid(row=0, column=1, padx=10, pady=10)

        # Label display
        self.label_display = tk.Label(root, text="", font=("Arial", 16))
        self.label_display.grid(row=1, column=0, columnspan=2)
        self.update_label_display()

        # Buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=2, column=0, columnspan=2)
        # Brush size slider
        self.brush_frame = tk.Frame(root)
        self.brush_frame.grid(row=4, column=0, columnspan=2, pady=5)

        tk.Label(self.brush_frame, text="Brush Size").pack(side=tk.LEFT)
        self.brush_slider = tk.Scale(self.brush_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                                    command=self.set_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT)

        tk.Button(self.button_frame, text="Pen", command=self.use_pen).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Eraser", command=self.use_eraser).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Reset", command=self.reset_canvas).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Save", command=self.save_image).pack(side=tk.LEFT)

        # Navigation Buttons
        self.nav_frame = tk.Frame(root)
        self.nav_frame.grid(row=3, column=0, columnspan=2, pady=5)

        tk.Button(self.nav_frame, text="Prev Label", command=self.prev_label).pack(side=tk.LEFT, padx=5)
        tk.Button(self.nav_frame, text="Next Label", command=self.next_label).pack(side=tk.LEFT, padx=5)
        tk.Button(self.nav_frame, text="Prev Image", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(self.nav_frame, text="Next Image", command=self.next_image).pack(side=tk.LEFT, padx=5)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.update_preview())

        self.last_x = self.last_y = None
        self.use_pen()
        self.update_preview()

    def use_pen(self):
        self.pen_color = "white"

    def use_eraser(self):
        self.pen_color = self.bg_color

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=self.brush_size, fill=self.pen_color,
                                    capstyle=tk.ROUND, smooth=tk.TRUE)
        self.last_x, self.last_y = x, y

    def update_preview(self):
        self.last_x = self.last_y = None
        self.update_reference_image()

    def reset_canvas(self):
        self.canvas.delete("all")
        self.update_preview()

    def save_image(self):
        self.root.update()

        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab(bbox=(x, y, x1, y1))
        img = img.convert("L")  # Grayscale only (no inversion)
        img = img.resize((120, 120), Image.Resampling.LANCZOS)

        label_folder = f"odiaData/characters/{self.current_label}"
        os.makedirs(label_folder, exist_ok=True)

        current_files = [f for f in os.listdir(label_folder) if f.endswith(".jpg")]
        image_count = len(current_files)
        save_path = f"{label_folder}/{self.current_label}_{image_count}.jpg"
        img.save(save_path, format="JPEG")

        self.reset_canvas()
        self.update_reference_image()



    def update_label_display(self):
        self.label_display.config(text=f"Label: {self.current_label}")

    def update_reference_image(self):
        folder = f"odiaData/characters/{self.current_label}"
        os.makedirs(folder, exist_ok=True)
        images = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
        if images:
            self.image_index %= len(images)
            img_path = os.path.join(folder, images[self.image_index])
            try:
                image = Image.open(img_path).resize((140, 140))
                self.reference_img = ImageTk.PhotoImage(image)
                self.preview_label.config(image=self.reference_img)
            except:
                self.preview_label.config(image="")
        else:
            self.preview_label.config(image="")

    def next_label(self):
        self.current_label = (self.current_label + 1) % self.total_labels
        self.image_index = 0
        self.update_label_display()
        self.update_reference_image()

    def set_brush_size(self, value):
        self.brush_size = int(value)

    def prev_label(self):
        self.current_label = (self.current_label - 1 + self.total_labels) % self.total_labels
        self.image_index = 0
        self.update_label_display()
        self.update_reference_image()

    def next_image(self):
        folder = f"odiaData/characters/{self.current_label}"
        if not os.path.exists(folder): return
        files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        if files:
            self.image_index = (self.image_index + 1) % len(files)
            self.update_reference_image()

    def prev_image(self):
        folder = f"odiaData/characters/{self.current_label}"
        if not os.path.exists(folder): return
        files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        if files:
            self.image_index = (self.image_index - 1 + len(files)) % len(files)
            self.update_reference_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingDatasetApp(root)
    root.mainloop()
