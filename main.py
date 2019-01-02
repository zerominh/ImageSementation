# %matplotlib inline
from numpy import ogrid, repeat, newaxis
from skimage import io
import skimage
from BilinearInterpolation import BilinearInterpolation
def creat_test_img():
    # Generate image that will be used for test upsampling
    # Number of channels is 3 -- we also treat the number of
    # samples like the number of classes, because later on
    # that will be used to upsample predictions from the network
    imsize = 3
    x, y = ogrid[:imsize, :imsize]
    img = repeat((x + y)[..., newaxis], 3, 2) / float(imsize + imsize)
    return img

def upsample_skimage(factor, input_img):
    
    # Pad with 0 values, similar to how Tensorflow does it.
    # Order=1 is bilinear upsampling
    return skimage.transform.rescale(input_img,
                                     factor,
                                     mode='constant',
                                     cval=0,
                                     order=1)


def main():
    img = creat_test_img()
    print(img.shape)
#     upsampled_img_skimage = upsample_skimage(factor=3, input_img=img)
#     io.imshow(upsampled_img_skimage, interpolation='none')
#     io.show()
    bilinearInter = BilinearInterpolation(None, None)

if __name__ == '__main__':
    main()