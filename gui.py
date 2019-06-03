import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import re
from PIL import Image, ImageTk

# 定义常量
FILE_PATTERN = r"^[A-Z]:/(.+/)*.+\.((jpg)|(jpeg)|(png))$"
DEFAULT_FONT = ("微软雅黑", 10)
CARD_WIDTH = 674
CARD_HEIGHT = 425

# 按钮函数
def command_select_path():
    label_card.grid_forget()
    button_anal.grid_forget()
    label_number.grid_forget()
    path_ = filedialog.askopenfilename()
    path.set(path_)

def resize_as_card(image):
    w, h = image.size[0:2];
    if (h * CARD_WIDTH > w * CARD_HEIGHT):
        h = h * CARD_WIDTH / w
        w = CARD_WIDTH
        resimage = image.resize((int(w), int(h)), Image.ANTIALIAS)
    else:
        w = w * CARD_HEIGHT / h
        h = CARD_HEIGHT
        resimage = image.resize((int(w), int(h)), Image.ANTIALIAS)
    return resimage

def command_confirm_path():
    path_ = path.get()
    res = re.match(FILE_PATTERN, path_)
    print(path_)
    if (res == None):
        messagebox.showinfo("错误", "所选择的文件或文件路径格式错误！")
    else:
        img_card_ = resize_as_card(Image.open(path_))
        img_card = ImageTk.PhotoImage(img_card_)
        label_card.grid(row=2, column=0, columnspan=4)
        label_card.config(image=img_card)
        label_card.image = img_card
        button_anal.grid(row=3, column=0, columnspan=4)

def command_anal():
    try:
        card_number = "62284 8033 03464 40515"
        number.set("所识别的卡号为：" + card_number)
        label_number.grid(row=4, column=0, columnspan=4)
    except:
        messagebox.showinfo("错误", "无法识别到卡号！")

# 创建窗口
window = tk.Tk()
window.title("银行卡号识别程序")
#window.geometry("800x600")
window.resizable(0, 0)

# 创建控件和布局
label_title = tk.Label(master=window, text="欢迎使用银行卡号识别程序！请在下方选择对应图片文件。", font=DEFAULT_FONT)
label_title.grid(row=0, column=0, columnspan=4)

label_path = tk.Label(master=window, text="目标路径：", font=DEFAULT_FONT)
path = tk.StringVar()
path.set("未选择文件路径")
number = tk.StringVar()
number.set("")
entry_path = tk.Entry(master=window, textvariable=path, font=DEFAULT_FONT, width=60)
button_path = tk.Button(master=window, text="浏览..", command=command_select_path, font=DEFAULT_FONT)
button_confirm = tk.Button(master=window, text="确定", command=command_confirm_path, font=DEFAULT_FONT)
label_card = tk.Label(master=window)
button_anal = tk.Button(master=window, text="开始识别", command=command_anal, font=DEFAULT_FONT)
label_number = tk.Label(master=window, textvariable=number, font=DEFAULT_FONT)

label_path.grid(row=1, column=0)
entry_path.grid(row=1, column=1)
button_path.grid(row=1, column=2)
button_confirm.grid(row=1, column=3)

# 显示窗口
window.mainloop()