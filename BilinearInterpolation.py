from __future__ import division
import numpy as np
import tensorflow as tf
class BilinearInterpolation:
    def __init__(self, factor, input_img):
        self.__factor = factor
        self.__input_img = input_img
        self.__output_img = None
    def __get_ker_size(self):
        return 2*self.__factor - self.__factor%2

    def up_sample(self):
        newH = self.__factor * self.__input_img.shape[0]
        newW = self.__factor * self.__input_img.shape[1]
        number_of_classes = self.__input_img[2]

        with tf.Graph().as_default():
            with tf.Session() as sess:
                with tf.device('/cpu:0'):
                    
                    res = tf.nn.conv2d_transpose(logits_pl, upsample_filt_pl,
                        output_shape = [1, newH, newW, number_of_classes],
                        stride = [1, self.__factor, self.__factor, 1])

                    self.__output_img = sess.run(res, 
                        )

                    return self.__output_img    