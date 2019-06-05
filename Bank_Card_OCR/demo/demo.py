import sys
sys.path.append("../")
import gui.gui as gui
import recognition.card_recognize as card_recognize
import getopt


def usage():
    print("usage : python demo.py [-c | -d] [-i input_path] [-o output_path]")


if __name__ == '__main__':
    input_path = ""
    output_path = ""
    dataset_model = False
    card_model = False
    dir_model = False
    try:
        options, args = getopt.getopt(sys.argv[1:], "hcdri:o:")
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
        elif name == '-r':
            dir_model = True

    if dataset_model and card_model:
        usage()
        sys.exit()

    if dataset_model:
        pass
    elif card_model:
        pass
    else:
        gui.main()