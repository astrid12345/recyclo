import matplotlib
matplotlib.use("TkAgg")  # 👈 Ensure GUI backend

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from pathlib import Path
import zipfile
import tkinter as tk
from tkinter import simpledialog

# === UNZIP DATASET IF NEEDED ===
zip_path = Path("20250615_AquaTrash_yolo_369.zip")
extract_dir = Path("20250615_AquaTrash_yolo_369")

if not extract_dir.exists():
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✅ Unzipped to {extract_dir}")
else:
    print(f"📂 Already extracted: {extract_dir}")

# === CONFIG ===
dataset_root = extract_dir
split = "val"
image_dir = dataset_root / "images" / split
label_dir = dataset_root / "labels" / split
label_dir.mkdir(parents=True, exist_ok=True)

image_files = sorted([f for f in image_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
print("📁 image_dir:", image_dir.resolve())
if not image_files:
    raise RuntimeError("No image files found.")

index = 0
boxes = []
class_ids = []
selected_box = None
click_start = None
fig, ax = None, None
rects = []

def load_image_and_labels(idx):
    global img, img_disp, boxes, class_ids
    image_file = image_files[idx]
    label_file = label_dir / f"{image_file.stem}.txt"
    img = cv2.imread(str(image_file))
    img_disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    boxes.clear()
    class_ids.clear()

    if label_file.exists():
        for line in label_file.read_text().splitlines():
            cls, x_c, y_c, bw, bh = map(float, line.split())
            x = int((x_c - bw / 2) * w)
            y = int((y_c - bh / 2) * h)
            bw_px = int(bw * w)
            bh_px = int(bh * h)
            boxes.append([x, y, bw_px, bh_px])
            class_ids.append(int(cls))

def save_labels():
    image_file = image_files[index]
    label_file = label_dir / f"{image_file.stem}.txt"
    h, w = img.shape[:2]
    with open(label_file, "w") as f:
        for cls, (x, y, bw, bh) in zip(class_ids, boxes):
            x_c = (x + bw / 2) / w
            y_c = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")
    print(f"💾 Saved: {label_file.name}")

def draw():
    global fig, ax, rects
    if fig is None or ax is None:
        matplotlib.rcParams['keymap.save'] = []
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("motion_notify_event", on_drag)
        fig.canvas.mpl_connect("button_release_event", on_release)
        fig.canvas.mpl_connect("key_press_event", on_key)

    ax.clear()
    ax.imshow(img_disp)
    ax.set_title(f"{image_files[index].name} [{index+1}/{len(image_files)}]")
    rects.clear()
    for (x, y, w, h) in boxes:
        rect = Rectangle((x, y), w, h, edgecolor='lime', facecolor='none', lw=2)
        rects.append(rect)
        ax.add_patch(rect)
    fig.canvas.draw()
    plt.show()

def on_click(event):
    global click_start, selected_box
    if event.inaxes != ax: return
    click_start = (int(event.xdata), int(event.ydata))
    for i, (x, y, w, h) in enumerate(boxes):
        if x <= click_start[0] <= x + w and y <= click_start[1] <= y + h:
            selected_box = i
            return
    selected_box = None

def on_drag(event):
    if click_start and selected_box is None and event.inaxes == ax:
        draw()
        x0, y0 = click_start
        x1, y1 = int(event.xdata), int(event.ydata)
        temp_rect = Rectangle((min(x0, x1), min(y0, y1)),
                              abs(x1 - x0), abs(y1 - y0),
                              edgecolor='red', facecolor='none', lw=1)
        ax.add_patch(temp_rect)
        fig.canvas.draw()

def on_release(event):
    global click_start
    if not click_start or event.inaxes != ax:
        click_start = None
        return

    x0, y0 = click_start
    x1, y1 = int(event.xdata), int(event.ydata)

    if selected_box is not None:
        print(f"🗑 Deleted box {boxes[selected_box]}")
        boxes.pop(selected_box)
        class_ids.pop(selected_box)
    else:
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w > 10 and h > 10:
            boxes.append([x, y, w, h])
            class_ids.append(0)
            print(f"➕ Added box: {(x, y, w, h)}")

    click_start = None
    draw()

def on_key(event):
    global index, image_files
    if event.key == "right":
        save_labels()
        index = min(index + 1, len(image_files) - 1)
        load_image_and_labels(index)
        draw()
    elif event.key == "left":
        save_labels()
        index = max(index - 1, 0)
        load_image_and_labels(index)
        draw()
    elif event.key == "s":
        save_labels()
    elif event.key == "q":
        plt.close(fig)
    elif event.key == "j":
        root = tk.Tk()
        root.withdraw()  # Hide main window
        user_input = simpledialog.askinteger("Jump to Image", f"Enter image index (1 to {len(image_files)}):")
        root.destroy()
        if user_input is not None and 1 <= user_input <= len(image_files):
            save_labels()
            index = user_input - 1
            load_image_and_labels(index)
            draw()
    elif event.key == "d":
        image_file = image_files[index]
        label_file = label_dir / f"{image_file.stem}.txt"

        confirm = input(f"🗑 Delete {image_file.name}? Type 'yes' to confirm: ").strip().lower()
        if confirm == "yes":
            # Delete the image and label
            print(f"🗑 Deleting {image_file.name} and {label_file.name if label_file.exists() else '(no label)'}")
            image_file.unlink()
            if label_file.exists():
                label_file.unlink()

            # Refresh image list and adjust index
            image_files.pop(index)
            if index >= len(image_files):
                index = len(image_files) - 1
            if image_files:
                load_image_and_labels(index)
                draw()
            else:
                print("🛑 No more images left.")
                plt.close(fig)



# === RUN ===
load_image_and_labels(index)
draw()
