# bank_card_recognition

* 银行卡尺寸可缩放为674x425
* 灰度值中0为黑，255为白

### 文件说明

binary.py：进行对二值化、canny算法初步摸索时的试验品，目前没有用处。

boundcut.py：试图用findcontours和多边形拟合提取边缘，效果很差，无用。

change.py：把训练数据暴力四等分成单个数字的程序。

getnumber.py：里面全部是可调用的函数，主要用于一些常用的图像处理，**为方便阅读，加了注释**。

hough.py：用霍夫变换提取、处理卡片边框问题，可以将文件保存在test_image2/文件夹下。

number.py：用于从图中提取出卡号横向位置的程序，能将结果保存在test_numbers/文件夹下。

