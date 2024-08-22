import numpy as np

# import pil
from PIL import Image
import os
from os.path import join

# dir="data/bdd100k/anomaly-seg-master/seg/train_labels/train/"
# search_classes=[16,18,19]
# for elem in os.listdir(dir):
#     np_image=np.array(Image.open(join(dir,elem)))
#     classes = np.unique(np_image[:, :])
#     if


# create random rgb color from int
def random_color(int_):
    # set seed
    np.random.seed(int_)
    color = np.random.randint(0, 255, 3)
    color[0] = int_
    return color


paths = ["0a0a0b1a-7c39d841_train_id_wrong.png", "0a0a0b1a-7c39d841_train_id_correct.png"]
for path in paths:
    # load image
    image = Image.open(path)
    np_image = np.array(image)
    print(f"image shape: {np_image.shape}")
    # get unique classes from R channel of the image
    classes = np.unique(np_image[:, :])
    # create a color dictionary for each class
    color_dict = {class_: random_color(class_) for class_ in classes}
    # create a colored image
    colored_image = np.zeros(np_image.shape + (3,), dtype=np.uint8)
    for class_, color in color_dict.items():
        colored_image[np_image == class_] = color
    # save the colored image
    colored_image = Image.fromarray(colored_image)
    colored_image.save(f"colored_{path}")

    print(f"colored_{path} saved")
    print(f"classes: {classes}")
