import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading

# Hàm để chạy chương trình và cập nhật vùng văn bản
def run_program(command, text_widget):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def read_output(process, text_widget):
        for line in iter(process.stdout.readline, ''):
            text_widget.insert(tk.END, line)
            text_widget.see(tk.END)
        process.stdout.close()
        process.wait()
        text_widget.insert(tk.END, "Program finished\n")

    thread = threading.Thread(target=read_output, args=(process, text_widget))
    thread.start()
    return process

# Hàm để dừng chương trình
def stop_program(process, text_widget):
    if process:
        process.terminate()
        process.wait()
        text_widget.insert(tk.END, "Program terminated\n")
        text_widget.see(tk.END)

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Run Programs")

# Tạo vùng văn bản cho chương trình
text_widget1 = ScrolledText(root, wrap=tk.WORD, width=50, height=20)
text_widget1.pack(pady=10)

# Tạo nút chạy chương trình
process1 = None
def start_program():
    global process1
    if process1 is None:
        text_widget1.delete(1.0, tk.END)
        process1 = run_program(["python", "test_p1.py"], text_widget1)

button1 = tk.Button(root, text="Run Program", command=start_program)
button1.pack(pady=5)

# Tạo nút dừng chương trình
def stop_program1():
    global process1
    if process1:
        stop_program(process1, text_widget1)
        process1 = None

stop_button1 = tk.Button(root, text="Stop Program", command=stop_program1)
stop_button1.pack(pady=5)

# Bắt đầu vòng lặp sự kiện
root.mainloop()
