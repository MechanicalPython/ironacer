#! /usr/local/bin/python3.7

from PIL import Image
import os
import unittest

class BBoxConvert:
    def __init__(self, images, labels, new_labels):
        self.images = images
        self.labels = labels
        self.new_labels = new_labels

    def convert_single_bbox_file_to_yolo(self, text_file, image_file, object_class=0):
        """
        BBox-label-tool file format:
            Number of boxes
            x, y, x, y,
            bottom left x, height, width, bottom y
        Yolo format:
            <object-class> <x_center> <y_center> <width> <height>
            all relative to width and height of image.
        :param text_file:
        :return:
        """
        return_list = []
        with open(text_file, 'r') as f:
            lines = f.readlines()
        number_of_boxes = int(lines[0])
        for box in range(1, number_of_boxes + 1):  # 1 to 2 gives only 1
            im = Image.open(image_file)
            total_width, total_height = im.size
            xmin, ymin, xmax, ymax = lines[box].split(' ')
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            dw = 1. / total_width
            dh = 1. / total_height
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh

            return_list.append([object_class, x, y, w, h])
            im.close()
        return return_list

    def convert_BBox_label_to_yolo_labels(self):
        """
        BBox-label-tool file format:
            Number of boxes
            bottom left x, height, width, bottom y
        Yolo format:
            <object-class> <x_center> <y_center> <width> <height>
            all relative to width and height of image.

        So conversion:

        :param output_dir:
        :param labels_dir:
        :param image_dir:
        :return:
        """
        labels = os.listdir(self.labels)
        labels = [l.split('.')[0] for l in labels]  # Strips file extension. Also removes .DS_Store and other hidden files
        labels.remove('')
        if not os.path.exists(self.new_labels):
            os.mkdir(self.new_labels)

        for label in labels:
            label_path = os.path.abspath(f'{self.labels}{label}.txt')
            image_path = os.path.abspath(f'{self.images}{label}.jpg')
            print(label_path, image_path)
            boxes_list = self.convert_single_bbox_file_to_yolo(label_path, image_path)
            with open(os.path.join(self.new_labels, os.path.basename(label_path)), 'w') as out:
                for box in boxes_list:
                    box = [str(b) for b in box]
                    out.write(' '.join(box) + '\n')

        print("Formatted " + self.labels)


    def split_images(self, image_dir, output_dir):
        """
        Splits the labeled images into training and testing images. Save files to ./darknet/
        :return:
        """
        image_dir = os.path.abspath(image_dir)
        image_dir_list = os.listdir(image_dir)
        if '.DS_Store' in image_dir_list:
            image_dir_list.remove('.DS_Store')

        percentage_test = .10
        cut_off = int(len(image_dir_list) * percentage_test)
        training_list = image_dir_list[:-cut_off]  # Gets first lot of items
        testing_list = image_dir_list[-cut_off:]  # Get second lot

        with open(f'{output_dir}/train.txt', 'w') as train:
            for file in training_list:
                train.write(f'{image_dir}/{file}\n')
        with open(f'{output_dir}/test.txt', 'w') as test:
            for file in testing_list:
                test.write(f'{image_dir}/{file}\n')


if __name__ == '__main__':
    BBoxConvert('Images/', 'Labels/', 'Yolo_labels/').convert_BBox_label_to_yolo_labels()

    # split_images('darknet/custom_data/images/', 'darknet/custom_data/')
