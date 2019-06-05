import sys
sys.path.append("../")
import gui.gui as gui
import recognition.card_recognize as card_recognize
import locate.card_locate as card_locate
import locate.number_locate as number_locate
import getopt
import numpy as np
import cv2 as cv
import os


def usage():
    print("usage : python demo.py [-c | -d] [-i input_path] [-o output_path]")
    print("        No args is for GUI")
    print("        The default input_path is \'./test_images/\'")
    print("        The default output_path is \'./test_result/\'")
    print("        -i to set the input_path")
    print("        -o to set the output_path")
    print("        -c to test card recognize")
    print("        -d to test data set recognize")


if __name__ == '__main__':
    input_path = "./test_images/"
    output_path = "./test_result/"
    dataset_model = False
    card_model = False
    try:
        options, args = getopt.getopt(sys.argv[1:], "hcdi:o:")
    except getopt.GetoptError:
        usage()
        sys.exit()
    for name, value in options:
        if name == '-h':
            usage()
        elif name == '-c':
            card_model = True
        elif name == '-d':
            dataset_model = True
        elif name == '-i':
            input_path = value
        elif name == '-o':
            output_path = value

    if dataset_model and card_model:
        usage()
        sys.exit()

    if dataset_model:
        with open(output_path + "result.txt", "w") as f:
            file_list = os.listdir(input_path)
            for file in file_list:
                image = cv.imdecode(np.fromfile(input_path + file, dtype=np.uint8), -1)
                card_number = card_recognize.line_recognize(image)
                _ = len(file) - 1
                while file[_] != '.':
                    _ -= 1
                output = file[0:_] + ":" + card_number
                print(output)
                f.write(output + "\n")

    elif card_model:
        with open(output_path + "result.txt", "w") as f:
            file_list = os.listdir(input_path)
            for file in file_list:
                image = cv.imdecode(np.fromfile(input_path + file, dtype=np.uint8), -1)
                hough_image = card_locate.hough_lines(image, None)
                line_image = number_locate.tophat_locate(hough_image, None)
                card_number = card_recognize.line_recognize(line_image)
                cv.imwrite(output_path + file, line_image)
                _ = len(file) - 1
                while file[_] != '.':
                    _ -= 1
                output = file[0:_] + ":" + card_number
                print(output)
                f.write(output + "\n")
    else:
        gui.main()
