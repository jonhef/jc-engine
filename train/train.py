import torch
from torch.utils.data import DataLoader, random_split
from data.data_processing import ChessDataset
from model import ChessNet
from data.settings import TrainingConfig, ModelConfig, SystemConfig
import torch.optim as optim
import torch.nn as nn
import tqdm
import argparse
import logging as log

logging = log.getLogger(__name__)

def load_model(model_path, device):
    model = ChessNet().to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded pretrained weights from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return model

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                       help="Path to pretrained checkpoint")
    parser.add_argument('--dataset', type=str, default='chess_dataset.csv',
                       help="Path to dataset file")
    args = parser.parse_args()
    
    device = torch.device(SystemConfig.DEVICE)
    logging.info("Device:", device)
    if args.checkpoint:
        model = load_model(args.checkpoint, device)
    else:
        model = ChessNet().to(device)
        logging.info("Initializing new model")
    
    dataset = ChessDataset(args.dataset)
    
    # Разделение данных
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    
    # DataLoader с настройками
    train_loader = DataLoader(
        train_set,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=SystemConfig.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=TrainingConfig.BATCH_SIZE,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=SystemConfig.PIN_MEMORY
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=TrainingConfig.LEARNING_RATE,
        weight_decay=TrainingConfig.WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=ModelConfig.USE_AMP)

    best_val_loss = float('inf')
    
    epoch = 0
    for i in tqdm.tqdm(range(TrainingConfig.NUM_EPOCHS)):
        # Training loop
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, (from_true, to_true)) in tqdm.tqdm(enumerate(train_loader)):
            x = x.to(device, non_blocking=True)
            from_true = from_true.to(device, non_blocking=True)
            to_true = to_true.to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=ModelConfig.USE_AMP):
                from_pred, to_pred = model(x)
                loss = (loss_fn(from_pred, from_true) + loss_fn(to_pred, to_true)) / 2
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRAD_CLIP)  # Клиппинг градиентов
            scaler.step(optimizer)
            scaler.update()

            
            if (batch_idx + 1) % TrainingConfig.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if batch_idx % TrainingConfig.LOG_INTERVAL == 0:
                logging.info(f"Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, (from_true, to_true) in val_loader:
                x = x.to(device, non_blocking=True)
                from_true = from_true.to(device, non_blocking=True)
                to_true = to_true.to(device, non_blocking=True)
                
                from_pred, to_pred = model(x)
                val_loss += (loss_fn(from_pred, from_true) + loss_fn(to_pred, to_true)) / 2

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, f'best_model_epoch_pretrained_{epoch+1}.pth')
            logging.error(f"Saved new best model with val loss {best_val_loss:.4f}")
        epoch += 1

    # Final save
    torch.save(model.state_dict(), 'final_model.pth')
    print("Training complete. Final model saved.")

if __name__ == '__main__':
    train()
