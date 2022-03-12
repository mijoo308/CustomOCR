from craft_onnx.detection import detect
from deep_text_recognition_benchmark.recognition import OPT, load_model, recognize
from config import *
import tempfile
import os
import cv2
import utils
import shutil
import argparse


def main():
    
    ''' [ 0. parse arg ] '''
    parser = argparse.ArgumentParser(description='Custom OCR')
    parser.add_argument('--image_path', type=str, required=True, help='image path')
    parser.add_argument('--recognition_model_path', type=str, default=False, required=True, help='recognition model path')
    parser.add_argument('--detection_threshold', type=float, default=0.5, help='detection threshold')
     # select model
    parser.add_argument('--transformation', default='None', type=str, required=True, help='None/TPS')
    parser.add_argument('--feature_extraction', type=str, required=True, help='ResNet/VGG')
    parser.add_argument('--sequence', default='BiLSTM', type=str, required=True, help='BiLSTM')
    parser.add_argument('--prediction', type=str, required=True, help='CTC/Attn')
        
    args = parser.parse_args()
    
    ''' [ 1. load model & img ] '''
    im = cv2.imread(args.image_path)
    h,w,_ = im.shape
    
    img_name = os.path.basename(args.image_path).split('.')[0]
    
    opt = OPT(args.recognition_model_path, args.transformation, args.feature_extraction, args.sequence, args.prediction)
    model, converter = load_model(opt)
    
    '''[ 2. detection ]'''
    boxes = detect(detection_model, im, args.detection_threshold)

    ltrb_list = []  ## [[(tl),(br)],[],[], .... ]
    for box in boxes:
        tl = tuple(box[0])
        br = tuple(box[2])
        ltrb_list.append([tl, br])

    ltrb_list = sorted(ltrb_list, key = lambda x : x[0][1])

    sorted_row_list = utils.get_row_list(ltrb_list, threshold=6)
    fixed_row_list = utils.fix_format(sorted_row_list, h, w, margin=2)
    # merged_row_list = utils.get_merged_row_list(sorted_row_list, h, w, margin=3)

    temp_detection_dir = tempfile.mkdtemp(prefix=img_name)

    im_gray = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

    '''[ 3. recognition ]'''
    cnt = 0
    for box in fixed_row_list:
        l, t, r, b = box[0][0], box[0][1], box[1][0], box[1][1]
        cropped_img = im_gray[t:b, l:r]
        cv2.imwrite(os.path.join(temp_detection_dir, str(cnt) + ".jpg"), cropped_img)
        cnt += 1

    opt.image_folder = temp_detection_dir
    # print(os.listdir(temp_detection_dir)) # for debugging

    # [[img_name, pred, confidence_score], ...]
    result = recognize(opt, model, converter)

    full_text = utils.get_full_text_result(result)
    print("[full text]")
    print(full_text)
    
    shutil.rmtree(temp_detection_dir)

    '''[ 4. visualize result ] '''
    # draw
    for box in fixed_row_list:
        tl = box[0]
        br = box[1]
        im = cv2.rectangle(im, tl, br, (0, 255, 0), 1)

    # save
    img_name = os.path.basename(args.image_path).split('.')[0]
    save_path = img_name + "_merged_detection.jpg"
    cv2.imshow('result', im)
    cv2.waitKey(0)
    cv2.imwrite(save_path, im)



if __name__ == "__main__":
    main()