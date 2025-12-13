# predict.py
import argparse
import os
import yaml
import pandas as pd
import torch
from src import get_model, get_tokenizer, Predictor
from src.data_loader import load_and_fix_data

def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model_path = cfg['project']['output_dir']
    print(f"Loading model from: {model_path}")
    
    try:
        model = get_model(model_path, num_labels=cfg['model']['num_labels'])
        tokenizer = get_tokenizer(model_path)
    except OSError:
        print(f"Error: {model_path} 경로에 모델 파일이 없습니다. 학습이 완료되었는지 확인하세요.")
        return

    test_path = cfg['predict']['test_path']
    print(f"Loading test data from: {test_path}")
    
    test_df = load_and_fix_data(
        test_path, 
        delimiter=cfg['data']['delimiter'], 
        quotechar=cfg['data']['quotechar'],
        is_test=True  
    )

    if test_df is None:
        raise ValueError("테스트 데이터를 불러오는데 실패했습니다.")

    predictor = Predictor(model, tokenizer)
    
    preds, probs = predictor.predict(
        texts=test_df['text'].tolist(),
        batch_size=cfg['train']['batch_size'],
        max_len=cfg['data']['max_len']
    )

    target_probs = probs[:, 1]

    save_dir = cfg['predict']['submission_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, cfg['predict']['submission_file'])

    submission = pd.DataFrame({
        'id': test_df['id'],
        'target': target_probs 
    })

    submission.to_csv(save_path, index=False)
    print(f"\n[Done] Submission file saved to: {save_path}")
    print(f" - Sample: {target_probs[:5]}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='make-model/config/koelectra.yaml', help='Path to config file')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        main(args)
    else:
        print("No GPU detected.")