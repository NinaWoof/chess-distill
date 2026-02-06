import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import OneCycleLR
from . import config
from .model import ChessNet
from .dataset import ChessDataset
from .loss import ChessLoss

def train(data_path, epochs, lr, batch_size, device):
    full_dataset = ChessDataset(data_path, augment=True)
    
    # Train/Val split
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Validation dataset should not have augmentation
    val_dataset.dataset.augment = False 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ChessNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # AdamW usually better with scheduling
    
    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=len(train_loader) * epochs,
        pct_start=0.3, anneal_strategy='cos'
    )
    
    criterion = ChessLoss()
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_dataset.dataset.augment = True # Ensure training has augmentation
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x, p_target, v_target in pbar:
            x, p_target, v_target = x.to(device), p_target.to(device), v_target.to(device)
            
            optimizer.zero_grad()
            p_logits, v_pred = model(x)
            
            loss, lp, lv = criterion(p_logits, v_pred, p_target, v_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.1e}"
            })
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        train_dataset.dataset.augment = False # Disable augmentation for validation
        epoch_val_loss = 0
        val_p_acc = 0
        with torch.no_grad():
            for x, p_target, v_target in val_loader:
                x, p_target, v_target = x.to(device), p_target.to(device), v_target.to(device)
                p_logits, v_pred = model(x)
                loss, _, _ = criterion(p_logits, v_pred, p_target, v_target)
                epoch_val_loss += loss.item()
                
                # Policy Accuracy (Top-1 against the distribution's max)
                # Since p_target is now a distribution, we check if model argmax matches target argmax
                preds = torch.argmax(p_logits, dim=1)
                target_best = torch.argmax(p_target, dim=1)
                val_p_acc += (preds == target_best).float().mean().item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_p_acc = val_p_acc / len(val_loader)
        
        print(f"Epoch {epoch+1} done. Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | P-Acc: {avg_val_p_acc:.4f}")
        
        # Save checkpoints
        latest_path = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
        torch.save(model.state_dict(), latest_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved to {best_path}")
            
    return model
memory_path = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
print(f"Training complete. Model saved to {memory_path}")
