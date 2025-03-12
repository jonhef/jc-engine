import os
import gc
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
from model.model import ChessNet
from data.settings import TrainingConfig, BigDataConfig, SystemConfig
from torch.utils.data import Dataset, DataLoader
from data.data_processing import fen_to_tensor, move_to_indices
import argparse

# train_large.py
class ChunkedDataset(Dataset):
    def __init__(self, chunk_path):
        # Читаем с явным указанием заголовков
        self.df = pd.read_csv(
            chunk_path,
        )
        
        # Проверка столбцов
        if not {'fen', 'move'}.issubset(self.df.columns):
            raise ValueError(f"Invalid columns in {chunk_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = fen_to_tensor(row['fen'])
        y_from, y_to = move_to_indices(row['move'])
        return x,   (
                        torch.tensor(y_from, dtype=torch.long),
                        torch.tensor(y_to, dtype=torch.long)
                    )

def train_on_chunk(model, optimizer, chunk_path, device):
    try:
        dataset = ChunkedDataset(chunk_path)
    except ValueError as e:
        print(e)
        return
    
    chunk = pd.read_csv(chunk_path)
    dataset = ChunkedDataset(chunk_path)
    
    # Проверка одного элемента датасета
    sample = dataset[0]
    print("Тип x:", type(sample[0]))  # Должно быть torch.Tensor
    print("Тип y:", type(sample[1]))  # Должно быть tuple of torch.Tensor
    
    loader = DataLoader(
        dataset,
        batch_size=TrainingConfig.BATCH_SIZE,
        collate_fn=lambda batch: (
            torch.stack([x for x, y in batch]),
            (torch.stack([y[0] for x, y in batch]),
             torch.stack([y[1] for x, y in batch]))
        )
    )
    
    for x, y in loader:
        y_from, y_to = y
        x = x.to(device)
        y_from = y_from.to(device)
        y_to = y_to.to(device)
        
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()

    best_loss = float('inf')
    for epoch in range(TrainingConfig.NUM_EPOCHS):
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            y_from, y_to = y
            y_from = y_from.to(device)
            y_to = y_to.to(device)
            y = (y_from, y_to)
            
            optimizer.zero_grad()
            from_pred, to_pred = model(x)
            loss = (loss_fn(from_pred, y_from) + loss_fn(to_pred, y_to)) / 2
            best_loss = min(best_loss, loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"Best loss: {best_loss:.4f}")
            
        # Освобождаем память
        del x, y, y_from, y_to
        gc.collect()
        torch.cuda.empty_cache()

def main():
    device = torch.device(SystemConfig.DEVICE)
    model = ChessNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    start_chunk = BigDataConfig.RESUME_FROM_CHUNK

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="checkpoint.pth",
                       help="Path to pretrained checkpoint")
    parser.add_argument('--dataset', type=str, default='chess_dataset.csv',
                       help="Path to dataset file")
    parser.add_argument("--new-dataset", type=bool, default=False, help="Using dataset instead of chunks")
    args = parser.parse_args()
    
        
    # Загрузка чекпоинта
    if os.path.exists(args.checkpoint):
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Проверка наличия ключей
        required_keys = ['model_state_dict', 'optimizer_state_dict']
        try:     
            model.load_state_dict(checkpoint)
            print("Resuming from checkpoint")
        except:
            if all(key in checkpoint for key in required_keys):
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_chunk = checkpoint.get('chunk_idx', 0) + 1
                print(f"Resuming from chunk {start_chunk}")
            else:
                print("Invalid checkpoint format. Starting from scratch.")
    
    # Итерация по чанкам
    # start_chunk = BigDataConfig.RESUME_FROM_CHUNK
    start_chunk = 0
    for chunk_idx in tqdm(range(start_chunk, len(os.listdir(BigDataConfig.CACHE_DIR)))):
        chunk_path = f"{BigDataConfig.CACHE_DIR}/chunk_{chunk_idx}.csv"
        if not os.path.exists(chunk_path) or args.new_dataset:
            # Генерация чанка из основного файла
            reader = pd.read_csv(args.dataset, chunksize=BigDataConfig.CHUNK_SIZE)
            for i, chunk in enumerate(reader):
                chunk.to_csv(chunk_path, index=False)
            del reader
        
        # Обучение на чанке
        train_on_chunk(model, optimizer, str(chunk_path), device)
        
        # Сохранение чекпоинта
        if chunk_idx % BigDataConfig.CHECKPOINT_INTERVAL == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'chunk_idx': chunk_idx
            }, 'checkpoint.pth')
        
if __name__ == "__main__":
    main()