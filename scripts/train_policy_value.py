import argparse
from chess_distill import config
from chess_distill.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()
    
    train(args.data, args.epochs, args.lr, args.batch_size, args.device)
