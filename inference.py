import torch
import os
import torchvision.transforms as transforms
from regressor.train_reg import RegressorModel
from PIL import Image
from torch.nn import functional as F
import pandas as pd
from torchsummary import summary
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_dl_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
                [
                    transforms.Resize((368, 368)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ])

    checkpoint_path = CHECKPOINT_PATH
    model = RegressorModel.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model = model.eval()
    summary(model, (3, 368, 368))
    return model, transform, device


def main(test_csv):
    model, transform, device = load_dl_model()
    inference_data = pd.read_csv(test_csv)
    
    # Regression inference
    weight_true = inference_data['weight'].tolist()
    weight_pred = []
    
    # Classification inference
    y_true = inference_data['label'].tolist()
    y_pred = []

    
    for _, row in inference_data.iterrows():
        img = row['image_path']
        image = Image.open(img)
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        pred_weight = model(input_tensor).item()  # convert tensor to float
        weight_pred.append(pred_weight)
        
        if pred_weight <= 2500:
            pred_label = 0
            y_pred.append(pred_label)  # low
        elif 2500 < pred_weight <= 3500:
            pred_label = 1
            y_pred.append(pred_label)  # middle
        else:
            pred_label = 2
            y_pred.append(pred_label)  # top

        print(f'Image: {os.path.basename(img)} - Predicted weight: {pred_weight:.2f} gram, Predicted class: {pred_label}')

    # Regression Metric
    eval_model_regressor(weight_true, weight_pred)
    
    # Classification Metric
    eval_model_classifier(y_true, y_pred)
    
    
def eval_model_classifier(y_true, y_pred):
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


def eval_model_regressor(target, pred):
    mse = mean_squared_error(target, pred) 
    r2 = r2_score(target, pred)
    mae = mean_absolute_error(target, pred)
    evs = explained_variance_score(target, pred)

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "MSE": mse,
            "MAE": mae,
            "R-squared": r2,
            "Ex. var score": evs,
        },
        index=[0],
    )

    return df_perf


if __name__ == '__main__':
    TEST_CSV = '/mnt/data/code/EDA_apple_sorting/assets/v2/test.csv'
    CHECKPOINT_PATH = '/mnt/data/code/EDA_apple_sorting/regressor/results/tb_logs/lightning_logs/version_2/checkpoints/best_model_038-5721.23-59.31.ckpt'

    main(test_csv=TEST_CSV)
