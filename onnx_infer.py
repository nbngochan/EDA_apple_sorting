import torch
import pandas as pd
import torch.onnx
import onnxruntime
from PIL import Image
import numpy as np
import time
import os
import torchvision.transforms as transforms
from utils.metrics import model_performance_classifier, eval_model_regressor
from utils.util import to_numpy, softmax


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
    [
        transforms.Resize((368, 368)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def transform_image(image_path):
    """
    Transforms an image to a tensor.

    Args:
        image_path (str): The path to the image file.

    Returns:
        input_tensor (torch.Tensor): The transformed image as a tensor.
    """
    image = Image.open(image_path)
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    return input_tensor


def onnx_infer_cls(ort_session, image_path):
    """
    Performs inference on an image using a pre-trained ONNX model for classification.

    Args:
        ort_session (onnxruntime.InferenceSession): The ONNX runtime session.
        image_path (str): The path to the image file.

    Returns:
        pred_idx (int): The predicted class index.
        pred_proba (float): The predicted class probability.
    """
    img_y = transform_image(image_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    pred_probs = softmax(ort_outs[0][0])
    pred_idx = np.argmax(pred_probs)

    return pred_idx, pred_probs[pred_idx]


def onnx_infer_reg(ort_session, image_path):
    """
    Performs inference on an image using a pre-trained ONNX model for regression.

    Args:
        ort_session (onnxruntime.InferenceSession): The ONNX runtime session.
        image_path (str): The path to the image file.

    Returns:
        ort_outs (float): The predicted weight.
    """
    
    img_y = transform_image(image_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    pred_weight = ort_outs[0][0][0]

    if pred_weight <= 2500:
        pred_class = 0  # low
    elif 2500 < pred_weight <= 3500:
        pred_class = 1  # middle
    else:
        pred_class = 2  # top

    return pred_weight, pred_class
    


def main(onnx_path, infer_type='reg'):
    map_class = {0: 'LOW', 1: 'MIDDLE', 2: 'TOP'}
    infer_type_dict = {'cls': onnx_infer_cls, 'reg': onnx_infer_reg}

    # Inference on a single image
    image_path = '/mnt/data/dataset/apple-weight-dataset/v2/4140-126/image0007368.png'
    ort_session = onnxruntime.InferenceSession(onnx_path)

    if infer_type in infer_type_dict:
        infer_func = infer_type_dict[infer_type]
        if infer_type == 'cls':
            pred_class, pred_proba = infer_func(ort_session, image_path)
            map_class = {0: 'LOW', 1: 'MIDDLE', 2: 'HIGH'}
            print(f'Image predicted as class: {map_class[pred_class]} with probability {pred_proba:.4f}')
        elif infer_type == 'reg':
            pred_weight, pred_class = infer_func(ort_session, image_path)
            print(f'Image: {os.path.basename(image_path)} - Predicted weight: {pred_weight:.2f}, - Predicted class: {map_class[pred_class]:^7}')
    else:
        print(f'Invalid inference type: {infer_type}')

    # Inference on multiple images from test dataset
    test_df = pd.read_csv('./assets/v2/test.csv')
    test_img_list = test_df['image_path'].tolist()

    y_pred = []
    weight_pred = []
    
    begin = time.time()
    
    for img_path in test_img_list:
        image_name = os.path.basename(img_path)
        start = time.time()
        if infer_type == 'cls':
            pred_class, pred_proba = infer_func(ort_session, img_path)
            end = time.time()
            print(f'Image {image_name} - Predicted class: {map_class[pred_class]:^7} with probability {pred_proba:.4f}, processing time: {end-start:.4f} secs')
        elif infer_type == 'reg':
            pred_weight, pred_class = infer_func(ort_session, img_path)
            weight_pred.append(pred_weight)
            end = time.time()
            print(f'Image: {image_name} - Predicted weight: {pred_weight:.2f}, - Predicted class: {map_class[pred_class]:^7}, processing time: {end-start:.4f} secs')

        y_pred.append(pred_class)
        
    end = time.time()

    elapsed = end - begin
    print(f"Total processing time: {elapsed:.2f} seconds")
    print(f'Average processing time per image: {elapsed/len(test_img_list):.2f} seconds')
    
    
    # Evaluation Metrics
    y_true = test_df['label'].tolist()
    weight_true = test_df['weight'].tolist()
    
    if infer_type == 'reg':
        print(eval_model_regressor(weight_true, weight_pred))  # Regression Metric
    
    model_performance_classifier(y_true, y_pred)  # Classification Metric: 0: <=250g; 1: 250-350g; 2: >350g
    

if __name__ == '__main__':
    # Regressor model
    onnx_path = './onnx-weight/mobilenetv3_regressor_ver2.onnx'
    main(onnx_path, infer_type='reg')
    
    # # Classifier model
    # onnx_path = './onnx-weight/mobilenetv3_classifier_ver0.onnx'
    # main(onnx_path, infer_type='cls')
    
