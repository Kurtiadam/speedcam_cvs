import pandas as pd
import imageio.v2 as imageio
import os
import shutil

# NAVIGATE TO THE LOCATION OF THE CSV FILE AND IMAGES BEFORE RUNNING THE SCRIPT

ground_truth = pd.read_csv(r"gt_train.csv",header=None,dtype={0: str})

first_column = ground_truth.iloc[:, 0]
first_column_int = ground_truth[0].astype(int)

TRAIN_SET_SIZE = 10000
VAL_SET_RATIO = 0.2
TEST_SET_RATIO = 0.1

classes = {
    0:"articulated_truck",
	1:"bicycle",
	2:"bus",
	3:"car",
	4:"motorcycle",
	5:"motorized_vehicle",
	6:"non-motorized_vehicle",
	7:"pedestrian",
	8:"pickup_truck",
	9:"single_unit_truck",
	10:"work_van"
}
key_list = list(classes.keys())
val_list = list(classes.values())

def folder_setup():
    global labels_folder, images_folder, start_path
    start_path = os.getcwd()
    if not os.path.exists("labels"):
        os.mkdir("labels")
    else:
        inp = input("The labels folder already exists. Should I delete it and make a new one? (Y/n)")
        if inp == "Y" or inp == "y":
            shutil.rmtree(start_path + "\labels")
            os.mkdir("labels")
            labels_folder = start_path + "\labels"
    if not os.path.exists("images"):
        os.mkdir("images")
    else:
        inp = input("The images folder already exists. Should I delete it and make a new one? (Y/n)")
        if inp == "Y" or inp == "y":
            shutil.rmtree(start_path + "\images")
            os.mkdir("images")
            images_folder = start_path + "\images"

def write_label(set, index, mode,idx,img_path):
    img = imageio.imread(img_path)
    img_height = img.shape[0]
    if img_height != 480:
        return True
    img_width = img.shape[1]
    if img_width != 720:
        return True

    class_name = ground_truth.iloc[idx,1]
    x_min = ground_truth.iloc[idx,2]
    y_min = ground_truth.iloc[idx,3]
    x_max = ground_truth.iloc[idx,4]
    y_max = ground_truth.iloc[idx,5]

    class_number_pos = val_list.index(class_name)
    class_number = key_list[class_number_pos]

    # original: xmin, ymin, xmax, ymax -> x_center y_center width height
    x_center = round(float((x_min+x_max)/2)/img_width,6)
    y_center = round(float((y_min+y_max)/2)/img_height,6)
    bb_height = round(float(y_max - y_min)/img_height,6)
    bb_width = round(float(x_max - x_min)/img_width,6)

    # class x_center y_center width height
    row_str = str(class_number) + " " + str(x_center) + " " + str(y_center) + " " + str(bb_width) + " " + str(bb_height)
    if mode == "new":
        with open(labels_folder + "\\" + set + "\\" +str(index) + '.txt', 'w') as f:
            f.write(row_str)
    if mode == "add":
        with open(labels_folder + "\\" + set + "\\" +str(index) + '.txt', 'r') as f:
            txt_data = f.read()
        txt_data += '\n' + row_str
        with open(labels_folder + "\\" + set + "\\" +str(index) + '.txt', 'w') as f:
            f.write(txt_data)
    shutil.copyfile(img_path, images_folder + "\\" + set +  "\\" + str(index) + ".jpg")
    return False

def make_set(set, size, start):
    cnt = 0
    skip_cnt = 0
    print("Set: " + set)
    os.chdir(labels_folder)
    os.mkdir(set)
    os.chdir(images_folder)
    os.mkdir(set)
    img_path = start_path + "\\" + "images_og\\train" + "\\" + str(first_column[start]) + ".jpg"
    prev_index = 10000000
    
    for idx, file_name in enumerate(first_column[start:]):
        if file_name == prev_index:
            skip = write_label(set,index=ground_truth.iloc[start + idx,0],mode="add",idx=start+idx, img_path=img_path)
        else:
            img_path = start_path + "\\" + "images_og\\train" + "\\" + str(ground_truth.iloc[start + idx,0]) + ".jpg"
            skip = write_label(set,index=ground_truth.iloc[start + idx,0],mode="new",idx=start+idx, img_path=img_path)
        prev_index = file_name
        if not (img_path == start_path + "\\" + "images_og\\train" + "\\" + str(ground_truth.iloc[start + idx + 1,0]) + ".jpg"):
            cnt += 1
            if skip:
                skip_cnt += 1
            print(cnt - skip_cnt)
            if size == (cnt - skip_cnt):
                return idx



def main():
    print("Making dataset...")
    folder_setup()
    end_idx = make_set(set="train", size = TRAIN_SET_SIZE, start = 0)
    end_idx_val = make_set(set="val", size = int(TRAIN_SET_SIZE*VAL_SET_RATIO), start = end_idx + 1)
    end_idx_test = make_set(set="test", size = int(TRAIN_SET_SIZE*TEST_SET_RATIO), start = end_idx+end_idx_val+2)
    print("Dataset is done!")

if __name__ == "__main__":
    main()