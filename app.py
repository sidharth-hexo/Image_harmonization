import numpy as np
import gradio as gr
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf

from Harmonizer.src import model

def bounding_box_from_mask(mask):
    """Return the bounding box of the mask.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (rmin, rmax, cmin, cmax)

def offset_scale_mask_image(mask_path, image, bg_image, offset_y, offset_x, scale):
    """Offset and scale the mask and apply it to the image.
    """
    offset = (offset_y, offset_x)
    mask = cv2.imread(mask_path, 0)
    image= cv2.imread(image)
    bg_image = cv2.imread(bg_image)
    rmin, rmax, cmin, cmax = bounding_box_from_mask(mask)
    # print(rmin)
    mask = mask[rmin:rmax, cmin:cmax]
    # print(mask.shape)
    image = image[rmin:rmax, cmin:cmax]
    mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    mask = mask.astype(np.uint8)
    image = image.astype(np.uint8)
    bg_image = bg_image.astype(np.uint8)

    fg_image = image*mask[:,:,np.newaxis]
    mask_w, mask_h = mask.shape[:2]
    bg_image_w, bg_image_h = bg_image.shape[:2]

    if offset[0]<0 or offset[0]>bg_image_w-mask_w:
        print("offset[0] is out of range")
        print('offset[0] has to be between ', 0, ' and ',bg_image_h-mask_h)
        return
    if offset[1]<0 or offset[1]>bg_image_h-mask_h:
        print("offset[1] is out of range")
        print('offset[1] has to be between ',0 , ' and ', bg_image_w-mask_w)
        return
    

    rmin_offset = offset[0] 
    rmax_offset = offset[0] + mask_w
    cmin_offset = offset[1] 
    cmax_offset = offset[1] + mask_h

    bg_image[rmin_offset:rmax_offset, cmin_offset:cmax_offset] = bg_image[rmin_offset:rmax_offset, cmin_offset:cmax_offset] * (1.0-mask[:,:,np.newaxis])
    # composite_image = bg_image.copy()             
    
    bg_image[rmin_offset:rmax_offset, cmin_offset:cmax_offset] += fg_image
    #convert from bgr to rgb
    composite_image = Image.fromarray(np.uint8(cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)), mode='RGB')
    composite_mask = np.zeros((composite_image.size[1], composite_image.size[0]))
    composite_mask[rmin_offset:rmax_offset, cmin_offset:cmax_offset] += mask

    return composite_image, composite_mask
    # return composite_image

def harmonize_image(comp, mask, pretrained='Harmonizer/pretrained/harmonizer.pth'):
    cuda = torch.cuda.is_available()
    
    # create/load the harmonizer model
    print('Create/load Harmonizer...')
    harmonizer = model.Harmonizer()
    if cuda:
        harmonizer = harmonizer.cuda()
    harmonizer.load_state_dict(torch.load(pretrained), strict=True)
    harmonizer.eval()
    comp = np.asarray(comp, dtype = np.uint8)
    mask = np.asarray(mask, dtype = np.uint8)
    if comp.shape[0] != mask.shape[0] or comp.shape[1] != mask.shape[1]:
        print('The size of the composite image and the mask are inconsistent')
        exit()

    comp = tf.to_tensor(comp)[None, ...]
    mask = tf.to_tensor(mask)[None, ...]

    if cuda:
        comp = comp.cuda()
        mask = mask.cuda()

    # harmonization
    with torch.no_grad():
        arguments = harmonizer.predict_arguments(comp, mask)
        harmonized = harmonizer.restore_image(comp, mask, arguments)[-1]

    # save the result
    harmonized = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
    harmonized = Image.fromarray(harmonized.astype(np.uint8))
    # harmonized.save(os.path.join(args.example_path, 'harmonized', example))

    print('Finished.')
    print('\n')
    return harmonized


def generate_composite_and_harmonise(mask, fg_image, bg_image, offset_y, offset_x, scale):
    comp, mask = offset_scale_mask_image(mask, fg_image, bg_image, offset_y, offset_x, scale)
    harmonized = harmonize_image(comp, mask)
    return comp, harmonized

def get_possible_offset_ranges(mask, bg_im, scale):
    mask = cv2.imread(mask)
    rmin, rmax, cmin, cmax = bounding_box_from_mask(mask)

    mask = mask[rmin:rmax, cmin:cmax]
    mask_h, mask_w = mask.shape[:2]
    print(mask.shape)
    bg_im = cv2.imread(bg_im)
    possible_offset_range_x = [0, bg_im.shape[1]-mask_w*scale]
    possible_offset_range_y = [0, bg_im.shape[0]-mask_h*scale]
    return possible_offset_range_x, possible_offset_range_y


# # def generate_shadows():


# #put the above parameters to the a python dictionary
#     args = {'batchs':1, 'GPU':0, 'lr':0.0002, 'loadSize':256, 'fineSize':fineSize, 'L1':L1, 'model':model, 'G':G, 'ngf':ngf, 'L_shadowrecons':L_shadowrecons, 'L_imagerecons':L_imagerecons, 'L_GAN':L_GAN, 'DISPLAY_PORT':DISPLAY_PORT, 'D':D, 'lr_D':lr_D, 'checkpoint':checkpoint, 'model_name':model_name, 'NAME':NAME, 'datasetmode':datasetmode}


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            fg_mask = gr.Image(type='filepath', label='foreground mask')
            fg_im =gr.Image(type='filepath', label='foreground image')
            bg_im = gr.Image(type='filepath', label='background image') 
            offset_y = gr.Slider(0, 10000, label='offset_y')
            offset_x = gr.Slider(0, 10000, label= 'offset_x')
            scale = gr.Slider(0, 3, label='scale')
        with gr.Column():
            bg_im_size = gr.Text(label='Background Image Size')
            fg_im_size = gr.Text(label='Foreground Image Size')
            composite_mask = gr.Image(label='Composite Mask')
            possible_offset_range_x = gr.Text(label='Possible Offset Range X')
            possible_offset_range_y = gr.Text(label='Possible Offset Range Y')
            comp = gr.Image(label="Generated Composite Image")
            harmonized = gr.Image(label="Harmonized Image")

    btn1 = gr.Button("Generate Composite and Harmonize")
    btn3 = gr.Button("Generate Composite and foreground mask")
    btn2 = gr.Button("Generate Composite")
    size_btn = gr.Button("Get Image Sizes")
    size_btn.click(lambda fg_im, bg_im: [(cv2.imread(fg_im).shape),cv2.imread(bg_im).shape], inputs=[fg_im, bg_im], outputs=[fg_im_size, bg_im_size])

    offset_btn = gr.Button("Get Offset Range")
    offset_btn.click(get_possible_offset_ranges, inputs=[fg_mask, bg_im, scale], outputs=[possible_offset_range_x, possible_offset_range_y])
    # gr.Title("Image Harmonization")
    btn1.click(generate_composite_and_harmonise, inputs=[fg_mask, fg_im, bg_im, offset_y, offset_x, scale], outputs=[comp, harmonized])
    btn2.click(offset_scale_mask_image, inputs=[fg_mask, fg_im, bg_im, offset_y, offset_x, scale], outputs=[comp])
    # gradio examples
    btn3.click(offset_scale_mask_image, inputs=[fg_mask, fg_im, bg_im, offset_y, offset_x, scale], outputs=[comp, composite_mask])
    # demo.

    gr.Examples([["Harmonizer/demo/image_harmonization/example/foregrounds/01 copy.jpg",
                  'Harmonizer/demo/image_harmonization/example/backgrounds/01 opt 1.jpg',
                  'Harmonizer/demo/image_harmonization/example/masks/mask_01 copy.jpg']], inputs=[fg_im, bg_im, fg_mask])

demo.launch()