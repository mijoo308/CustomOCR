    
'''options'''
detection_model = r'./craft_onnx/detector_craft.onnx'
RECOGNITION_MODEL = r'./trained_data/korean_news.pth'
DETECTION_THRESHOLD = 0.5


''' Select Model '''
TRANSFORMATION = 'None'
FEATURE_EXTRACTION = 'ResNet'
SEQUENCE = 'BiLSTM'
PREDICTION = 'CTC'
    