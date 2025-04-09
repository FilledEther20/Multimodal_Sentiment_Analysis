import torch
import torch.nn as nn
import os
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from meld_dataset import MELD_Dataset, prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from tqdm import tqdm

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare dataloaders
    print("Loading datasets...")
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        args.train_csv, args.train_video_dir,
        args.dev_csv, args.dev_video_dir,
        args.test_csv, args.test_video_dir,
        batch_size=args.batch_size
    )
    
    # Initialize model
    print("Initializing model...")
    model = MultimodalSentimentModel()
    model = model.to(device)
    
    # Initialize trainer
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader
    )
    
    # Create directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss = trainer.train_epoch()
        
        # Evaluate on validation set
        val_loss, val_metrics = trainer.evaluate(dev_loader, phase="val")
        
        print(f"Train Loss: {train_loss['total']:.4f}, Val Loss: {val_loss['total']:.4f}")
        print(f"Emotion Accuracy: {val_metrics['emotion_accuracy']:.4f}, Sentiment Accuracy: {val_metrics['sentiment_accuracy']:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            best_model_state = model.state_dict().copy()
            best_model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(best_model_state, best_model_path)
            print(f"Saved new best model with val loss: {best_val_loss:.4f}")
    
    # Load best model for final evaluation
    print("\nTraining complete. Loading best model for final evaluation...")
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss['total']:.4f}")
    print(f"Emotion Accuracy: {test_metrics['emotion_accuracy']:.4f}")
    print(f"Sentiment Accuracy: {test_metrics['sentiment_accuracy']:.4f}")
    print(f"Emotion Precision: {test_metrics['emotion_precision']:.4f}")
    print(f"Sentiment Precision: {test_metrics['sentiment_precision']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal sentiment analysis model")
    
    # Dataset paths
    parser.add_argument("--train_csv", type=str, default="../dataset/train/train_sent_emo.csv", 
                        help="Path to training CSV file")
    parser.add_argument("--train_video_dir", type=str, default="../dataset/train/train_splits", 
                        help="Path to training video directory")
    parser.add_argument("--dev_csv", type=str, default="../dataset/dev/dev_sent_emo.csv", 
                        help="Path to validation CSV file")
    parser.add_argument("--dev_video_dir", type=str, default="../dataset/dev/dev_splits_complete", 
                        help="Path to validation video directory")
    parser.add_argument("--test_csv", type=str, default="../dataset/test/test_sent_emo.csv", 
                        help="Path to test CSV file")
    parser.add_argument("--test_video_dir", type=str, default="../dataset/test/output_repeated_splits_test", 
                        help="Path to test video directory")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="saved_models", 
                        help="Directory to save model checkpoints")
    
    args = parser.parse_args()
    main(args)
