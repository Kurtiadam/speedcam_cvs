import os
import sys
import imageio.v2 as imageio
import shutil
import random
import pandas as pd
random.seed = 42

TRAIN_SET_RATIO = 0.7
VAL_SET_RATIO = 0.2
TEST_SET_RATIO = 0.1

def write_label(file_name, set):
    global img_height, img_width
    original_dir = os.getcwd()
    curr_file_path = original_dir + "\\" + file_name
    if file_name[-4:] == ".jpg":
        shutil.copyfile(curr_file_path, images_folder + "\\" + set +  "\\" + file_name)
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
        with open(labels_folder + "\\" + set + "\\"+ file_name, 'w') as f:
            f.write(text_str)
        os.chdir(original_dir)


def folder_setup():
    global labels_folder, images_folder
    init_path = os.getcwd()
    dataset_path = init_path + "\datasets\license_plate_detection_dataset"
    os.chdir(dataset_path)
    if not os.path.exists("labels"):
        os.mkdir("labels")
        os.chdir(dataset_path + "\labels")
        os.mkdir("train")
        os.mkdir("val")
        os.mkdir("test")
        os.chdir(dataset_path)
    else:
        inp = input(
            "The labels folder already exists. Should I delete it and make a new one? (Y/n)")
        if inp == "Y" or inp == "y":
            shutil.rmtree(dataset_path + "\labels")
            os.mkdir("labels")
            os.chdir(dataset_path + "\labels")
            os.mkdir("train")
            os.mkdir("val")
            os.mkdir("test")
            os.chdir(dataset_path)
        else:
            sys.exit("labels folder already exists")
    if not os.path.exists("images"):
        os.mkdir("images")
        os.chdir(dataset_path + "\images")
        os.mkdir("train")
        os.mkdir("val")
        os.mkdir("test")
    else:
        inp = input(
            "The images folder already exists. Should I delete it and make a new one? (Y/n)")
        if inp == "Y" or inp == "y":
            shutil.rmtree(dataset_path + "\images")
            os.mkdir("images")
            os.chdir(dataset_path + "\images")
            os.mkdir("train")
            os.mkdir("val")
            os.mkdir("test")
        else:
            sys.exit("images folder already exists")
    labels_folder = dataset_path + "\labels"
    images_folder = dataset_path + "\images"
    os.chdir(dataset_path + "\images_og")


def make_file_pairs():
    files = os.listdir()
    grouped_files = {}

    for file in files:
        file_name, file_ext = os.path.splitext(file)
        if file_name not in grouped_files:
            grouped_files[file_name] = []
        grouped_files[file_name].append(file_ext)

    random_pairs = random.sample(list(grouped_files.items()), len(grouped_files))
    return random_pairs

def make_dataset():
    folder_setup()
    random_pairs = make_file_pairs()

    num_of_samples = len(random_pairs)
    for idx, pair in enumerate(random_pairs):
        file_name = pair[0] + pair[1][0]
        if idx < (num_of_samples*TRAIN_SET_RATIO):
            write_label(file_name, "train")
            file_name = pair[0] + pair[1][1]
            write_label(file_name, "train")
        elif (num_of_samples*(TRAIN_SET_RATIO+VAL_SET_RATIO)) >= (idx) >= (num_of_samples*TRAIN_SET_RATIO):
            write_label(file_name, "val")
            file_name = pair[0] + pair[1][1]
            write_label(file_name, "val")
        elif (num_of_samples*(TRAIN_SET_RATIO+VAL_SET_RATIO)) < (idx):
            write_label(file_name, "test")
            file_name = pair[0] + pair[1][1]
            write_label(file_name, "test")



def main():
    print("Making dataset...")
    make_dataset()
    print("Dataset is done!")


if __name__ == "__main__":
    main()
