import math
import numpy as np
import tensorflow as tf
import cv2
import PIL.Image as Image
from matplotlib import pyplot as plt


class CroppedImage:
    def __init__(self, original_image_array, cropped_image_array, left, top, right, bottom):
        self.original_image_array = original_image_array
        self.cropped_image_array = cropped_image_array
        self.left_index = left
        self.top_index = top
        self.right_index = right
        self.bottom_index = bottom
        self.centroid_index = ((top + bottom) / 2, (left + right) / 2)


def _generate_cropping_coordinates(img, frame_shift_vertical, frame_shift_horizontal, v_index, h_index):
    i, j = v_index, h_index
    left = j * frame_shift_horizontal
    top = i * frame_shift_vertical
    right = (j + 1) * frame_shift_horizontal
    bottom = (i + 1) * frame_shift_vertical
    return left, top, right, bottom


def generate_sub_images(img, required_shape):
    img_array = np.array(img)*1.0
    img_shape = img_array.shape

    required_height, required_width = required_shape[0], required_shape[1]
    img_height, img_width = img_shape[0], img_shape[1]
    
    frame_shift_vertical = math.gcd(required_height, img_height)
    frame_shift_horizontal = math.gcd(required_width, img_width)
    
    number_vertical_chunks_initial = int(img_height / frame_shift_vertical)
    number_horizontal_chunks_initial = int(img_width / frame_shift_horizontal)
    
    sub_images = [[None for i in range(number_horizontal_chunks_initial)] for j in range(number_vertical_chunks_initial)]
    for i in range(number_vertical_chunks_initial):
        for j in range(number_horizontal_chunks_initial):
            bbox_coordinates = _generate_cropping_coordinates(img, frame_shift_vertical, frame_shift_horizontal, i, j)
            cropped = CroppedImage(img_array, np.array(img.crop(bbox_coordinates)), *bbox_coordinates)
            sub_images[i][j] = cropped
    return sub_images


# test generating sub images on a 1024x2048 image with a 512x512 constraint
if __name__ == '__main__':
    print('\n1024x2048 image with 512x512 constraint:\n========================================\n')
    img_path = 'test_photo.png'
    img = Image.open(img_path)
    sub_images = generate_sub_images(img, (512, 512))
    print('Shape of sub_images list: ' + str(np.array(sub_images).shape) + '\n')
    sub_images_centroids = [x.centroid_index for row in sub_images for x in row]
    print('sub_images centroids (vertical-axis, horizontal-axis): ' + str(sub_images_centroids) + '\n')
    sample_cropped_img = sub_images[0][0].cropped_image_array
    print('sample cropped image (top-left box): ' + str(sample_cropped_img))
    cropped_to_show = Image.fromarray(sample_cropped_img, 'RGB')
    # uncomment the following two lines to see that the cropping checks out:
    #img.show()
    #cropped_to_show.show()
