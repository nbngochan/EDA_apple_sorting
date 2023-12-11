import os
import torch
import torchvision.transforms as transforms
from regressor.train_reg import RegressorModel
from PIL import Image
from torch.nn import functional as F
import pandas as pd
from torchsummary import summary
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_dl_model(checkpoint_path, device):
    """
    Loads a pretrained model from a checkpoint file.

    Args:
        device (torch.device): Device to use for model computations.

    Returns:
        RegressorModel: The loaded model.
    """
    model = RegressorModel.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model = model.eval()
    summary(model, (3, 368, 368))
    return model


def transform_image(image_path):
    """
    Transforms an image for model input.

    Args:
        image_path (str): Path to the image.

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((368, 368)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = Image.open(image_path)
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)


def predict_weight(image_path, model, device):
    """
    Predicts the weight of an apple from an image.

    Args:
        image_path (str): Path to the image.
        model (RegressorModel): The pretrained model.
        device (torch.device): Device to use for model computations.

    Returns:
        float: Predicted weight of the apple.
    """
    image_tensor = transform_image(image_path)
    image_tensor = image_tensor.to(device)
    pred_weight = model(image_tensor).item()
    return pred_weight


def predict_label(pred_weight):
    """
    Predicts the label (low, middle, or top) of an apple based on its weight.

    Args:
        pred_weight (float): Predicted weight of the apple.

    Returns:
        int: Predicted label (0: low, 1: middle, 2: top).
    """
    if pred_weight <= 2500:
        pred_label = 0
    elif 2500 < pred_weight <= 3500:
        pred_label = 1
    else:
        pred_label = 2
    return pred_label


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

def main():
    checkpoint_path = "/mnt/data/code/EDA_apple_sorting/regressor/results/tb_logs/lightning_logs/version_2/checkpoints/best_model_038-5721.23-59.31.ckpt"
    test_dataset = '/mnt/data/code/EDA_apple_sorting/assets/v2/test.csv'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dl_model(checkpoint_path, device)

    test_data = pd.read_csv(test_dataset)

    label_true = test_data['label'].tolist()
    weight_true = test_data['weight'].tolist()
    label_pred = []
    weight_pred = []
    
    
    # Make predictions for each image in the test data
    for _, row in test_data.iterrows():
        image_path = row["image_path"]
        pred_weight = predict_weight(image_path, model, device)
        pred_label = predict_label(pred_weight)
        
        label_pred.append(pred_label)
        weight_pred.append(pred_weight)
        
        print(f'Image: {os.path.basename(image_path)} - Predicted weight: {pred_weight:.2f} gram, Predicted class: {pred_label}')
    
    # Evaluate predictions
    eval_model_classifier(label_true, label_pred)
    eval_model_regressor(weight_true, weight_pred)


if __name__ == '__main__':
    main()
    
