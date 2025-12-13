# train.py
import argparse
import yaml
import torch
import gc
from src.utils import set_seeds
from src.data_loader import get_tokenized_datasets
from src.model import get_model, get_tokenizer
from src.trainer import get_trainer

def main(args):
    # 1. Load Config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. Setup
    set_seeds(cfg['project']['seed'])
    print(f"\n[{cfg['model']['name']}] Training Start")

    # 3. Load Tokenizer & Data
    tokenizer = get_tokenizer(cfg['model']['name'])
    train_ds, val_ds = get_tokenized_datasets(cfg, tokenizer)

    # 4. Load Model
    model = get_model(cfg['model']['name'], cfg['model']['num_labels'])

    # 5. Initialize Trainer
    trainer = get_trainer(model, tokenizer, train_ds, val_ds, cfg)

    # 6. Train
    trainer.train()

    # 7. Save & Evaluate
    trainer.save_model(cfg['project']['output_dir'])
    tokenizer.save_pretrained(cfg['project']['output_dir'])
    
    final_metrics = trainer.evaluate()
    print(f"Final Validation Metrics: {final_metrics}")

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    print("\nTraining Finished Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='make-model/config/koelectra.yaml', help='Path to config file')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        main(args)
    else:
        print("No GPU detected.")