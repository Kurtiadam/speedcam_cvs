import os
import sys
import imageio.v2 as imageio
import shutil
import pandas as pd


def write_label(file_name, labels_folder, images_folder):
    global img_height, img_width
    original_dir = os.getcwd()
    curr_file_path = original_dir + "\\" + file_name
    if file_name[-4:] == ".jpg":
        shutil.copyfile(curr_file_path, images_folder + "\\" + file_name)
        img = imageio.imread(curr_file_path)
        img_height, img_width = img.shape[:2]
    elif file_name[-4:] == ".txt":
        df = pd.read_csv(curr_file_path, sep=': ', engine='python',
                         header=None, names=['column', 'value'])
        df = df.set_index('column')
        # corners: TOPLEFT_X,TOPLEFT_Y TOPRIGHT_X,topright_y bottomright_x,bottomright_y bottomleft_x,BOTTOMLEFT_Y
        bbox = df.loc["corners"]['value']
        bbox_s = bbox.split(" ")
        corners = [corner.split(",") for corner in bbox_s]
        # YOLO needs x_center y_center width height
        x_min = int(corners[0][0])
        x_max = int(corners[1][0])
        y_min = int(corners[0][1])
        y_max = int(corners[-1][1])
        x_center = round(float((x_min+x_max)/2)/img_width, 6)
        y_center = round(float((y_min+y_max)/2)/img_height, 6)
        bb_height = round(float(y_max - y_min)/img_height, 6)
        bb_width = round(float(x_max - x_min)/img_width, 6)
        numbers = [x_center, y_center, bb_width, bb_height]
        numbers.insert(0, 0)
        text_str = ' '.join(map(str, numbers))
        with open(labels_folder + "\\" + file_name, 'w') as f:
            f.write(text_str)
        os.chdir(original_dir)


def setup():
    init_path = os.getcwd()
    dataset_path = init_path + "\datasets\license_plate_detection_dataset"
    os.chdir(dataset_path)
    if not os.path.exists("labels"):
        os.mkdir("labels")
    else:
        inp = input(
            "The labels folder already exists. Should I delete it and make a new one? (Y/n)")
        if inp == "Y" or inp == "y":
            shutil.rmtree(dataset_path + "\labels")
            os.mkdir("labels")
        else:
            sys.exit("labels folder already exists")
    if not os.path.exists("images"):
        os.mkdir("images")
    else:
        inp = input(
            "The images folder already exists. Should I delete it and make a new one? (Y/n)")
        if inp == "Y" or inp == "y":
            shutil.rmtree(dataset_path + "\images")
            os.mkdir("images")
        else:
            sys.exit("images folder already exists")
    labels_folder = dataset_path + "\labels"
    images_folder = dataset_path + "\images"
    os.chdir(dataset_path + "\images_og")
    return labels_folder, images_folder


def make_dataset():
    labels_folder, images_folder = setup()
    files = os.listdir()
    for idx, file in enumerate(files):
        print(idx)
        write_label(file, labels_folder, images_folder)


def main():
    print("Making dataset...")
    make_dataset()
    print("Dataset is done!")


if __name__ == "__main__":
    main()
