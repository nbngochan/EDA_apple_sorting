import torch
import os
import pickle
import torchvision.transforms as transforms
from train_cls import ClassifierModel
from PIL import Image
from torch.nn import functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


TEST_CSV = '/root/data/EDA_apple_sorting/assets/test_reg.csv'
CHECKPOINT_PATH = '/root/data/EDA_apple_sorting/results/tb_logs/lightning_logs/version_20/checkpoints/best_model_028-0.0002-1.00.ckpt'

ML_OPTIONS = ['decision_tree', 'gradient_boosting', 'knn_weight',
              'logistic_regression', 'naive_bayes', 'random_forest', 'svm']


def load_dl_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ])

    checkpoint_path = CHECKPOINT_PATH
    model = ClassifierModel.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model = model.eval()
    
    return model, transform, device



def main(model_type):
    if model_type == 'machine_learning':
        infer_ml()
    elif model_type == 'deep_learning':
        infer_dl()
    else:
        print("Invalid model type. Please choose 'machine_learning' or 'deep_learning'.")


def load_ml_model(name):
    with open(f'./ml-weight/{name}_weights.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def infer_ml(option='naive_bayes'):
    model = load_ml_model(option)
    inference_data = pd.read_csv(TEST_CSV)
    
    y_true = inference_data['label'].tolist()
    y_pred = model.predict(inference_data['pixels'].values.reshape(-1, 1))

    print_evaluation_metric(y_true, y_pred)


def infer_dl(test_csv=TEST_CSV):
    model, transform, device = load_dl_model()
    inference_data = pd.read_csv(test_csv)
    y_true = inference_data['label'].tolist()
    y_pred = []
    classes = ['low', 'mid', 'top']
    class_list = []
    
    for _, row in inference_data.iterrows():
        img = row['image_path']
        image = Image.open(img)
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        output = model(input_tensor)
        h_x = F.softmax(output, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        
        y_pred.append(idx[0])
        class_list.append(classes[idx[0]])
        
        print(f'Image: {os.path.basename(img)} - Predicted class: {classes[idx[0]]}, Prob: {probs[0]}')


    print_evaluation_metric(y_true, y_pred)
    
    
def print_evaluation_metric(y_true, y_pred):
    # Print confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Calculate accuracy, precision, recall, f1 score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")


if __name__ == '__main__':
    main(model_type='deep_learning')
