from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import run
import os
import sys
import argparse
from PIL import Image


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


if __name__ == '__main__':

    st.title('YOLOv5 Streamlit App')
    ## From detect.py. add parser for command line run.
    # main function is RUN and it takes a lot of option.
    # when -- is added before each option, parser will seperate them and 
    # add to the run function.
    #  
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(640,640), help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()

    ## opt.argument = xxx to change/access argument value
    

    source = ("Image detection", "Video detection", "Web Cam detection")
    source_index = st.sidebar.selectbox("Select input", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "Upload images", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='loading...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                print(f'picture line 89 : {uploaded_file}, type : {type(uploaded_file)}')
                picture = picture.save(f'data\images\{uploaded_file.name}')
                opt.source = f'data\images\{uploaded_file.name}'
        else:
            is_valid = False
    elif source_index == 1 :
        uploaded_file = st.sidebar.file_uploader("Upload video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='loading...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False
            
    elif source_index == 2 :
        is_valid = True # if webcam detection enable, no need to prepare any file. We just go to prediction
    


    if is_valid:
        print('valid')
        #print(f'filepath : {opt.source}')
        #print(f"HERE ---- {opt} ----")
        if st.button('Start'):
            print("-"*25," RUN ", "-"*25)
            print(opt.weights)
            if source_index in [0,1] :
                run( 
                    opt.weights,
                    opt.source,
                    opt.data,
                    opt.imgsz,
                    opt.conf_thres,
                    opt.iou_thres, 
                    opt.max_det, 
                    opt.device,  
                    opt.view_img,
                    opt.save_txt, 
                    opt.save_conf, 
                    opt.save_crop,  
                    opt.nosave,  
                    opt.classes, 
                    opt.agnostic_nms,  
                    opt.augment,
                    opt.visualize,
                    opt.update,  
                    opt.project, 
                    opt.name, 
                    opt.exist_ok, 
                    opt.line_thickness, 
                    opt.hide_labels,
                    opt.hide_conf,
                    opt.half, 
                    opt.dnn,
                )
            else :
                opt.source = '0' # enable webcam recording
                run(
                    opt.weights,
                    opt.source
                    )

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))

                    st.balloons()
            elif source_index == 1 :
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / vid))

                    st.balloons()