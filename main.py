import tkinter as tk
from tkinter import filedialog
from few_shot import FewShot
from PIL import Image


def select_train_images():
    global train_directory
    train_directory = filedialog.askdirectory(title="Select Train Images")
    train_label.config(text="Train klasörü: " + train_directory)
    train_label.pack()
    print("Selected Train Images:", train_directory)


def select_test_images():
    global test_directory
    test_directory = filedialog.askdirectory(title="Select Test Images")
    test_label.config(text="Test klasörü: " + test_directory)
    test_label.pack()
    print("Selected Test Images:", test_directory)


def start_training():
    print("Training started...", train_directory, test_directory, "--")
    train_start_label.config(text="Train Started")
    global fewshot
    fewshot = FewShot(train_directory, test_directory)
    fewshot.train_model()
    train_start_label.config(text="Train End")
    train_start_label.pack()


def start_test():
    print("Training started...", train_directory, test_directory, "--")
    fewshot.predict_labels()
    fewshot.generate_confusion_matrix()
    fewshot.show_confusion_matrix()


# Tkinter penceresi oluşturma
window = tk.Tk()
window.title("Image Training")
window.geometry("800x600")

input_grp = tk.Frame(window)
input_grp.pack()

# Train Resimleri Yükleme Butonu
train_input = tk.Button(
    input_grp, text="Select Train Images", command=select_train_images
)
train_input.pack(padx=10, pady=10, side=tk.LEFT)

# Test Resimleri Yükleme Butonu
test_input = tk.Button(input_grp, text="Select Test Images", command=select_test_images)
test_input.pack(padx=10, pady=10, side=tk.LEFT)

btn_grp = tk.Frame(window)
btn_grp.pack()

# Eğitimi Başlat Butonu
train_btn = tk.Button(btn_grp, text="Eğit", command=start_training)
train_btn.pack(padx=10, pady=10, side=tk.LEFT)

test_btn = tk.Button(btn_grp, text="Test", command=start_test)
test_btn.pack(padx=10, pady=10, side=tk.LEFT)

train_label = tk.Label(window, text="")
test_label = tk.Label(window, text="")
train_start_label = tk.Label(window, text="")

window.mainloop()
