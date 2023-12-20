from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2 as cv
import numpy as np


def main():
    dir_from = 'Image/'
    dir_to = 'augmentation/'
    batch_num = 100

    # Data augmentation #######################################################
    data_generator = ImageDataGenerator(rotation_range=20,
                                        brightness_range=[0.6, 1.0],
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        horizontal_flip=True,
                                        vertical_flip=True
                                        )
    aug = data_generator.flow_from_directory(dir_from, save_prefix='a', batch_size=32,
                                             class_mode="sparse",
                                             classes=['0', '1', '2', '3', '4', '5', '6', 'others'])
    classes = aug.class_indices
    a_map = {}
    for k, v in classes.items():    # reverse dictionary
        a_map[v] = k
    # generate images and save them
    for batch, label in aug:
        for i, l in enumerate(label):
            count = {
                '0': len(os.listdir(dir_to + "0")),
                '1': len(os.listdir(dir_to + "1")),
                '2': len(os.listdir(dir_to + "2")),
                '3': len(os.listdir(dir_to + "3")),
                '4': len(os.listdir(dir_to + "4")),
                '5': len(os.listdir(dir_to + "5")),
                '6': len(os.listdir(dir_to + "6")),
                'others': len(os.listdir(dir_to + "others"))
            }
            label_s = a_map[l]
            direct = dir_to + label_s + '/'
            file = str(count[label_s]) + '.png'
            img_path = os.path.join(direct, file)
            cv.imwrite(img_path, batch[i].astype(np.uint8))
        batch_num -= 1
        if batch_num == 0:
            break


if __name__ == '__main__':
    main()
