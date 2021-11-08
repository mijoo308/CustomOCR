from craft_onnx.detection import detect # detection
from deep_text_recognition_benchmark.recognition import OPT, load_model, recognize
import tempfile
import os
import cv2
import utils
import shutil

def main():
    test_img_path = r'C:\Users\beyon\Desktop\CustomOCR\news_test\news.jpg'
    threshold = 0.5
    detection_model = r'./craft_onnx/detector_craft.onnx'
    img_name = os.path.basename(test_img_path).split('.')[0]
    im = cv2.imread(test_img_path)

    recog_model = ''
    transformation = 'TPS'
    feature_extraction = 'ResNet'
    sequence = 'BiLSTM'
    prediction = 'Attn'

    opt = OPT(recog_model, transformation, feature_extraction, sequence, prediction)
    model, converter = load_model(opt)
    
    '''detection'''
    boxes = detect(detection_model, im, threshold)

    ltrb_list = []  ## [[(tl),(br)],[],[], .... ]
    for box in boxes:
        tl = tuple(box[0])
        br = tuple(box[2])
        ltrb_list.append([tl, br])
        # im = cv2.rectangle(im, tl, br, (255, 0, 0), 2)

    ltrb_list = sorted(ltrb_list, key = lambda x : x[0][1])

    sorted_row_list = utils.get_row_list(ltrb_list, threshold=3)
    fixed_row_list = utils.fix_format(sorted_row_list, margin=2)
    # merged_row_list = utils.get_merged_row_list(sorted_row_list)


    ### TODO: Test 필요

    temp_detection_dir = tempfile.mkdtemp(prefix=img_name)

    '''recognition'''
    cnt = 0
    for box in fixed_row_list:
        # print(box[0][0], box[0][1], box[1][0], box[1][1])
        l, t, r, b = box[0][0], box[0][1], box[1][0], box[1][1]
        # l, t, r, b = box[0][0][0], box[0][0][1], box[0][1][0], box[0][1][1]
        cropped_img = im_gray[t:b, l:r]
        cv2.imwrite(os.path.join(temp_detection_dir, str(cnt) + ".jpg"), cropped_img)
        cnt += 1

    opt.image_folder = temp_detection_dir
    print(os.listdir(temp_detection_dir)) # for debugging

    # [[img_name, pred, confidence_score], ...]
    result = recognize(opt, model, converter) # 확인되면 안에서 프린트 되는 곳 삭제

    full_text = utils.get_full_text_result(result)
    print(full_text)
    
    shutil.rmtree(temp_detection_dir)

    '''visualize'''
    # draw
    for box in merged_row_list:
        tl = box[0]
        br = box[1]
        im = cv2.rectangle(im, tl, br, (0, 255, 0), 2)

    # save
    img_name = os.path.basename(test_img_path).split('.')[0]
    save_path = img_name + "_merged_detection.jpg"
    cv2.imshow('g', im)
    cv2.waitKey(0)
    # cv2.imwrite(save_path, im)



if __name__ == "__main__":
    main()