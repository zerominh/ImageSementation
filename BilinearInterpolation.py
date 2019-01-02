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


    def __upsample_filt(self, size):
        """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)


    def __bilinear_upsample_weights(self, number_of_classes):
        """
        Create weights matrix for transposed convolution with bilinear filter
        initialization.
        """
        
        filter_size = self.__get_ker_size()
        
        weights = np.zeros((filter_size,
                            filter_size,
                            number_of_classes,
                            number_of_classes), dtype=np.float32)
        
        upsample_kernel = self.__upsample_filt(filter_size)
        
        for i in range(number_of_classes):
            
            weights[:, :, i, i] = upsample_kernel
        
        return weights


    def up_sample(self):
        newH = self.__factor * self.__input_img.shape[0]
        newW = self.__factor * self.__input_img.shape[1]
        number_of_classes = self.__input_img.shape[2]
        expanded_img = np.expand_dims(self.__input_img, axis=0)
        upsample_filter_pl_np = self.__bilinear_upsample_weights(number_of_classes)
        with tf.Graph().as_default():
            with tf.Session() as sess:
                with tf.device('/cpu:0'):
                    upsample_filter_pl = tf.placeholder(tf.float32)
                    logits_pl = tf.placeholder(tf.float32)
                    res = tf.nn.conv2d_transpose(logits_pl, upsample_filter_pl,
                        output_shape = [1, newH, newW, number_of_classes],
                        strides = [1, self.__factor, self.__factor, 1])

                    self.__output_img = sess.run(res, feed_dict = {
                        logits_pl: expanded_img,
                        upsample_filter_pl:upsample_filter_pl_np
                    })

                    self.__output_img = self.__output_img.squeeze()

                    return self.__output_img    