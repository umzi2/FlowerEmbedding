import kagglehub
import os
import shutil


def download_flower():
    path = kagglehub.dataset_download("alxmamaev/flowers-recognition")
    dataset_folder = os.path.join(path, "flowers")
    folders = os.listdir(dataset_folder)
    train = "train_data/train"
    test = "train_data/test"
    val = "train_data/val"
    os.makedirs(train, exist_ok=True)
    os.makedirs(test, exist_ok=True)
    os.makedirs(val, exist_ok=True)
    n = 0
    for index in range(len(folders)):
        folder_imgs = os.path.join(dataset_folder, folders[index])
        img_set = set(os.listdir(folder_imgs))
        test_set = set(list(img_set)[:10])
        img_set = img_set - test_set
        val_set = set(list(img_set)[:16])
        img_set = img_set - val_set
        for img_name in img_set:
            shutil.copyfile(
                os.path.join(folder_imgs, img_name),
                os.path.join(train, f"{n:04}_{index}.jpg"),
            )
            n += 1
        for img_name in val_set:
            shutil.copyfile(
                os.path.join(folder_imgs, img_name),
                os.path.join(val, f"{n:04}_{index}.jpg"),
            )
            n += 1
        for img_name in test_set:
            shutil.copyfile(
                os.path.join(folder_imgs, img_name),
                os.path.join(test, f"{n:04}_{index}.jpg"),
            )
            n += 1
