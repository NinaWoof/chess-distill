import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from . import config
from .model import ChessNet
from .dataset import ChessDataset
from .loss import ChessLoss

def train(data_path, epochs, lr, batch_size, device):
    dataset = ChessDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ChessNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ChessLoss()
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, p_target, v_target in pbar:
            x, p_target, v_target = x.to(device), p_target.to(device), v_target.to(device)
            
            optimizer.zero_grad()
            p_logits, v_pred = model(x)
            
            loss, lp, lv = criterion(p_logits, v_pred, p_target, v_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "p": f"{lp.item():.4f}", "v": f"{lv.item():.4f}"})
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "latest.pt"))
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch+1}.pt"))

    return model
memory_path = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
print(f"Training complete. Model saved to {memory_path}")
