#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   changeFace.py
@Time    :   2022/04/22 16:41:42
@Author  :   Tianyi Wang
@Version :   1.0
@Contact :   tianyiwang58@gmail.com
@Desc    :   None
'''

#########changeFaceLib########
import os
import sys
import time
from argparse import Namespace
from tkinter import TRUE
from editings import latent_editor
import dlib
import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import torch
import torchvision.transforms as transforms
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.
from PIL import Image

# here put the import lib
import streamlit as st

########################################

def main():
    st.markdown("# 一键更换表情")
    st.markdown("## 左边可以更改模式~")
    #ChangefaceInit() # 初始化
    st.sidebar.title("选择模式")
    app_mode = st.sidebar.selectbox("选择模式", ["趣味模式", "手动模式","关于我们"])
    if app_mode == "趣味模式":
        funnyMode()
    elif app_mode == "手动模式":
        advancedMode()
    elif app_mode == "关于我们":
        st.markdown("""
        ## 关于我们
        """)
        st.write("这是一个简单的小程序，用于更换人脸表情")
def funnyMode():
    """
        趣味模式，使用词语更换人脸表情
    """
    st.markdown("""
    ## 趣味模式
    """)
    file = uploadImage()
    # 单行文本输入框
    sel = st.selectbox("选择你想要的效果", ["微笑","伤心", "正常","大笑"])
    pars = {"微笑": [0.2,0],"伤心":[-0.7,0], "正常":[0,0],"大笑":[1,0]}
    # 这里后面换成单选框，使用字典或者list映射为参数
    if(st.button("运行！", key="Funny")):
        if file:
            #st.write(type(np.asanyarray(bytearray(file.read()), dtype=np.uint8)))
            bitsFile=file.getvalue()
            savepath = 'saveFiles/'+file.name
            with open(savepath,'wb') as f:
                f.write(bitsFile)
            #st.image(savepath)
            #st.write(pars[sel][0],pars[sel][1])
            st.image(ChangeFaceMain(pars[sel][0],pars[sel][1],savepath))
        else:
            st.write("No file input")
def advancedMode():
    """
        手动模式，手动指定各种参数
    """
    file = uploadImage()
    st.markdown("""
    ## 手动模式
    """)
    st.markdown("## 简单变化")
    smile = st.slider("笑容变化程度", min_value=-1.0, max_value=1.0,step=0.01)
    age = st.slider("年龄变化程度", min_value=-1.0, max_value=1.0,step=0.01)
    #addition = st.selectbox("Which would you like", ["胡子","2", "3"])
    #st.write(addition)
    if(st.button("运行笑容变化！", key="Funny")):
        if file:
            #st.write(type(np.asanyarray(bytearray(file.read()), dtype=np.uint8)))
            bitsFile=file.getvalue()
            savepath = 'saveFiles/'+file.name
            with open(savepath,'wb') as f:
                f.write(bitsFile)
            #st.image(savepath)
            st.image(ChangeFaceMain(smile,savepath,mode=1,normalMode=1))
            # DegreeInput,inputFilePath,mode = 0,normalMode=0,addmode = 1
        else:
            st.write("No file input")
    if(st.button("运行年龄变化！", key="Smile")):
        if file:
            #st.write(type(np.asanyarray(bytearray(file.read()), dtype=np.uint8)))
            bitsFile=file.getvalue()
            savepath = 'saveFiles/'+file.name
            with open(savepath,'wb') as f:
                f.write(bitsFile)
            #st.image(savepath)
            st.image(ChangeFaceMain(age,savepath,mode=1,normalMode=0))
            # DegreeInput,inputFilePath,mode = 0,normalMode=0,addmode = 1
        else:
            st.write("No file input")
    st.markdown("## 面部特效")
    selAdd = st.selectbox("选择你想要的效果", ["胡子","嘴唇", "眼睛"])
    selAddDict = {"眼睛": 0,"胡子" :1,"嘴唇":2}
    if(st.button("运行面部特效！", key="Funny")):
        if file:
            #st.write(type(np.asanyarray(bytearray(file.read()), dtype=np.uint8)))
            bitsFile=file.getvalue()
            savepath = 'saveFiles/'+file.name
            with open(savepath,'wb') as f:
                f.write(bitsFile)
            #st.image(savepath)
            st.image(ChangeFaceMain(selAddDict[selAdd],savepath,mode=0,addmode = selAddDict[selAdd]))
        else:
            st.write("No file input")

def uploadImage():
    uploaded_file = st.file_uploader("上传你的自拍~", type=['png', 'jpg'] )
    return uploaded_file


############ChangeFaceCodes###############
def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))
def ChangefaceInit():
    model_path = "checkpoint/ckpt.pt"
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['is_train'] = False
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    global net 
    net = pSp(opts)
    net.eval()
    net.cuda()
    
    print('Model successfully loaded!')
def ChangeFaceMain(DegreeInput,inputFilePath,mode = 1,normalMode=0,addmode = 1):
# 设置输入图像
    # Setup required image transformations
    EXPERIMENT_ARGS = {
            "image_path": inputFilePath
        }
    EXPERIMENT_ARGS['transform'] = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    global resize_dims
    resize_dims = (256, 256)


    image_path = EXPERIMENT_ARGS["image_path"]
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")

    run_align = True
    # run_alignment后面
    if run_align:
        input_image = run_alignment(image_path)
    else:
        input_image = original_image

    input_image.resize(resize_dims)
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)
    with torch.no_grad():
        x = transformed_image.unsqueeze(0).cuda()

        tic = time.time()
        latent_codes = get_latents(net, x)
        
        # calculate the distortion map
        imgs, _ = net.decoder([latent_codes[0].unsqueeze(0).cuda()],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        result_image, _ = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    # Display inversion:
    display_alongside_source_image(tensor2im(result_image[0]), input_image)
    # 图像编辑
    if mode:
        editor = latent_editor.LatentEditor(net.decoder)
        normalDic ={0:'age',1:'smile'}
        # interface-GAN
        interfacegan_directions = {
                'age': './editings/interfacegan_directions/age.pt',
                'smile': './editings/interfacegan_directions/smile.pt' }
        edit_direction = torch.load(interfacegan_directions[normalDic[normalMode]]).cuda() 
        edit_degree = DegreeInput # 设置微笑幅度
        img_edit, edit_latents = editor.apply_interfacegan(latent_codes[0].unsqueeze(0).cuda(), edit_direction, factor=edit_degree)  # 设置微笑
        # align the distortion map
        img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # fusion
        conditions = net.residue(res_align)
        result, _ = net.decoder([edit_latents],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)

        result = torch.nn.functional.interpolate(result, size=(256,256) , mode='bilinear')
        #这个是显示图片的函数
        return display_alongside_source_image(tensor2im(result[0]), input_image)
    else:
        addDic = {0:'eyes',1:'beard',2:'lip'}
        
        # GANSpace
        # addition
        ganspace_pca = torch.load('./editings/ganspace_pca/ffhq_pca.pt') 
        ganspace_directions = {
            'eyes':            (54,  7,  8,  20),       # 眼睛    
            'beard':           (58,  7,  9,  -20),      # 胡子
            'lip':             (34, 10, 11,  20) }      # 嘴唇
        edit_direction = ganspace_directions[addDic[addmode]]
        img_edit, edit_latents = editor.apply_ganspace(latent_codes[0].unsqueeze(0).cuda(), ganspace_pca, [edit_direction])
        # align the distortion map
        img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))
        conditions = net.residue(res_align)
        result, _ = net.decoder([edit_latents],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        result = torch.nn.functional.interpolate(result, size=(256,256) , mode='bilinear')
        return display_alongside_source_image(tensor2im(result[0]), input_image)


# 图像对齐
def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm
def align_face(filepath, predictor):
    """
    :param filepath: str
    :return: PIL Image
    """

    lm = get_landmark(filepath, predictor)

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Return aligned image.
    return img

def run_alignment(image_path):
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image 

# step 3 高保真逆向映射
def display_alongside_source_image(result_image, source_image):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)

def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes
if __name__ == "__main__":
    main()
