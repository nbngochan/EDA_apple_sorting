import torch
import torch.onnx
from regressor.train_reg import RegressorModel
from classifier.train_cls import ClassifierModel


CHECKPOINT_PATH_REG = '/mnt/data/code/EDA_apple_sorting/regressor/results/tb_logs/lightning_logs/version_2/checkpoints/best_model_038-5721.23-59.31.ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def converter(checkpoint_path, infer_type='reg', file_name='./onnx-weight/mobilenetv3_regressor.onnx'):
    """
    Converts a PyTorch model to ONNX format.

    Args:
        checkpoint_path (str): The path to the PyTorch checkpoint file.

    Returns:
        None
    """
    
    # Load the PyTorch model from the checkpoint file
    if infer_type == 'reg':
        model = RegressorModel.load_from_checkpoint(checkpoint_path)
    elif infer_type == 'cls':
        model = ClassifierModel.load_from_checkpoint(checkpoint_path)
    
    # Move the model to the device (CPU or GPU)
    model = model.to(device)

    # Create a random input tensor for the model
    torch_input = torch.randn(1, 3, 368, 368).to(device)

    # Export the model to ONNX format
    torch.onnx.export(model, torch_input, file_name, export_params=True,
                                    input_names = ['input'], output_names = ['output'])



if __name__ == '__main__':
    converter(CHECKPOINT_PATH_REG)
    
    