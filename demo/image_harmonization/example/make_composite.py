import cv2
import numpy as np

def make_composite_of_two_images(img1, img2, mask, save_path):
    # Create a composite image from two images and a mask
    # img1: background image
    # img2: foreground image
    # mask: mask image
    # return: composite image

    # Convert uint8 to float
    foreground = img2.astype(float)
    background = img1.astype(float)
    

    #rescale the mask to fit background image
    mask = cv2.resize(mask, (background.shape[1], background.shape[0]))
    mask = np.tile(np.expand_dims(mask, -1), (1,1,3))
    # mask = mask.astype(float)/255

    # print(mask.shape)
    #resize foreground image to fit background image
    foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
    # print(foreground.shape)
    # Normalize the alpha mask to keep intensity between 0 and 1
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    mask = np.asarray(mask, np.float32)
    # Multiply the foreground with the  on all channels
    foreground = cv2.multiply(np.asarray(foreground, np.float32), mask)

    # Multiply the background with ( 1 - mask )
    background = cv2.multiply(1.0 - mask, np.asarray(background, np.float32))

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)

    # Return a normalized output image for display

    #save the output
    cv2.imwrite(save_path, outImage)

    return outImage/255

if __name__ == '__main__':

    import os
    for bg_image in os.listdir('Harmonizer/demo/image_harmonization/example/backgrounds'):
        for fg_image in os.listdir('Harmonizer/demo/image_harmonization/example/foregrounds'):
            fg_im_name = fg_image.split('.')[0]
            bg_im_name = bg_image.split('.')[0]
            # print(bg_image)
            bg_im = cv2.imread('Harmonizer/demo/image_harmonization/example/backgrounds/'+bg_image)
            fg_im = cv2.imread('Harmonizer/demo/image_harmonization/example/foregrounds/'+fg_image)


            mask = cv2.imread('Harmonizer/demo/image_harmonization/example/masks/'+'mask_'+fg_im_name+'.jpg', 0)
            make_composite_of_two_images(bg_im, fg_im, mask, 
                                         save_path='Harmonizer/demo/image_harmonization/example/composites/'+fg_im_name+bg_im_name+'.jpg')

            #get base file name
