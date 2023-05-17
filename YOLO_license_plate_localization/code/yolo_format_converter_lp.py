import os
import imageio.v2 as imageio
import shutil

def move_file(file_name,labels_folder, images_folder):
    curr_dir = os.getcwd()
    curr_file_path = curr_dir + "\\" + file_name
    if file_name[-4:] == ".txt":
        shutil.copyfile(curr_file_path, labels_folder + "\\" + file_name)
    elif file_name[-4:] == ".jpg":
        shutil.copyfile(curr_file_path, images_folder + "\\" + file_name)

def setup():
    init_path = os.getcwd()
    dataset_path = init_path + "\datasets\license_plate_detection_dataset"
    os.chdir(dataset_path)
    if not os.path.exists("labels"):
        os.mkdir("labels")
    if not os.path.exists("images"):
        os.mkdir("images")
    labels_folder = dataset_path + "\labels"
    images_folder = dataset_path + "\images"
    os.chdir(dataset_path + "\images_og")
    return labels_folder, images_folder

def make_dataset():
    labels_folder, images_folder =setup()
    files = os.listdir()
    for file in files:
        move_file(file,labels_folder, images_folder)
    



def main():
    print("Making dataset...")
    make_dataset()
    print("Dataset is done!")

if __name__ == "__main__":
    main()