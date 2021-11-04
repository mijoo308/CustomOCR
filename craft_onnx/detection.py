import cv2
import onnxruntime as rt
import numpy as np
import craft_onnx.craft_utils.craft_utils
import craft_onnx.craft_utils.imgproc as im_proc


# parameter

def detect(onnx_model_path, image, threshold = 0.5):

    sess = rt.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    # im = cv2.imread(image_path)
    h,w,_ = image.shape

    img_resized, target_ratio, size_heatmap = im_proc.resize_aspect_ratio(image, w, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio


    x = im_proc.normalizeMeanVariance(image)
    x = x.transpose(2, 0, 1) # [h, w, c] to [c, h, w]
    x = np.expand_dims(x, axis=0) # [c, h, w] to [b, c, h, w]

    # compute ONNX Runtime output prediction
    ort_inputs = {sess.get_inputs()[0].name: x}

    y_onnx_out, feature_onnx_out = sess.run(None, ort_inputs)

    boxes_list, polys_list = [], []
    for out in y_onnx_out:
        # make score and link map
        score_text = out[:, :, 0]
        score_link = out[:, :, 1]

        # Post-processing
        boxes, polys, mapper = craft_onnx.easyocr_utils.craft_utils.getDetBoxes(score_text, score_link, threshold, 0.4, 0.4)
        boxes = craft_onnx.easyocr_utils.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        boxes = np.array(boxes, dtype=int)

        # print(boxes)

    return boxes





