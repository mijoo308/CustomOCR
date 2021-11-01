from craft_onnx.onnx_test import detect # detection
from deep_text_recognition_benchmark.recognition import recognize
import tempfile
import os
import cv2
import utils
import shutil

def main():
    test_img_path = r'C:\Users\beyon\Desktop\CustomOCR\news_test\news.jpg'
    threshold = 0.5
    onnx_model_path = r'./craft_onnx/detector_craft.onnx'
    img_name = os.path.basename(test_img_path).split('.')[0]
    im = cv2.imread(test_img_path)


    '''detection'''
    boxes = detect(onnx_model_path, im, threshold)

    ltrb_list = []  ## [[(tl),(br)],[],[], .... ]
    for box in boxes:
        tl = tuple(box[0])
        br = tuple(box[2])
        ltrb_list.append([tl, br])
        # im = cv2.rectangle(im, tl, br, (255, 0, 0), 2)

    ltrb_list = sorted(ltrb_list, key = lambda x : x[0][1])

    sorted_row_list = utils.get_row_list(ltrb_list, threshold=3)
    merged_row_list = utils.get_merged_row_list(sorted_row_list)

    model = ''
    transformation = 'TPS'
    feature_extraction = 'ResNet'
    sequence = 'BiLSTM'
    prediction = 'Attn'

    ## TODO: folder 단위가 아니라 이미지 하나씩 처리하는 방법으로 바꿔야함
    #       1. 그냥 temp 폴더 만들어서 원래 방식대로 폴더를 넘기는 방식도 있음
    #       2. 모델 로드를 한 번만 하는 방식으로 만들어야 함
    # -> temp 폴더 만드는 게 제일 편할 것 같음..


    tempdir = tempfile.mkdtemp(prefix=img_name)
    shutil(tempdir)

    '''recognition'''
    for box in merged_row_list:
        l, t, r, b = box[0][0], box[0][1], box[1][0], box[1][1]
        cropped_img = im[t:b, l:r]
        recognize(cropped_img, model, transformation, feature_extraction, sequence, prediction)

    # draw
    for box in merged_row_list:
        tl = box[0]
        br = box[1]
        im = cv2.rectangle(im, tl, br, (0, 255, 0), 2)
        # cv2.imshow('g', im)
        # cv2.waitKey(0)

    # save
    img_name = os.path.basename(test_img_path).split('.')[0]
    save_path = img_name + "_merged_detection.jpg"
    cv2.imshow('g', im)
    cv2.waitKey(0)
    # cv2.imwrite(save_path, im)



if __name__ == "__main__":
    main()