import os
import argparse
import torchaudio
import torch

# AWS Sagemaker
SM_MODEL_DIR=os.environ.get('SM_MODEL_DIR',".")
SM_CHANNEL_TRAINING=os.environ.get('SM_CHANNEL_TRAINING',"opt/ml/input/data/training")
SM_CHANNEL_VALIDATION=os.environ.get('SM_CHANNEL_VALIDATION',"opt/ml/input/data/validation")
SM_CHANNEL_TEST=os.environ.get('SM_CHANNEL_TEST',"opt/ml/input/data/test")

os.environ["PYTHON_CUDA_ALLOC_CONF"]="expandable_segments:True"

def pargs_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--learning-rate",type=int,default=0.001)
    
    # Data directories
    parser.add_argument("--train-dir",type=str,default=SM_CHANNEL_TRAINING)
    parser.add_argument("--test-dir",type=str,default=SM_CHANNEL_TEST)
    parser.add_argument("--val-dir",type=str,default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--model-dir",type=str,default=SM_MODEL_DIR)
    
    return parser.parse_args()


def main():
    
    # For audio we would have to install FFMPEG
    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))
    
    args=pargs_args()
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used=torch.cuda.max_memory_allocated()/1024**3
        print(f"Initial GPU memory used:{memory_used:.2f} GB")
        
    train_loader,dev_loader,test_loader=prepare_dataloaders(
        train_csv=
    )
    
    
if __name__=="__main__":
    main()