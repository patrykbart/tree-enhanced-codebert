import wandb

import os
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from tree_enhanced_codeberta_mlm import TreeEnhancedRobertaForMaskedLM
from utils import set_seed, setup_logger, load_config, initialize_wandb

def mask_inputs(input_ids, tokenizer, mlm_probability):
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=labels.device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True)
        for seq in input_ids
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    input_ids[masked_indices] = tokenizer.mask_token_id
    return input_ids, labels

def evaluate_model_mlm(model, dataset, tokenizer, batch_size, device, config):
    mlm_probability = config['data']['mlm_probability']
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    total_loss, total_samples = 0.0, 0
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating on test set'):
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            input_ids, labels = mask_inputs(input_ids, tokenizer, mlm_probability)

            if config['model']['extra_embeddings']:
                depths = batch['depths'].to(device)
                sibling_indices = batch['sibling_indices'].to(device)
                tree_attention_mask = batch['tree_attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, depths=depths, sibling_indices=sibling_indices, tree_attention_mask=tree_attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs.loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            predictions = torch.argmax(outputs.logits, dim=-1)
            mask = labels != -100
            all_predictions.extend(predictions[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    return {'loss': avg_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser(description='Training script for TreeStarEncoder')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    current_dir = Path(__file__).parent
    cache_dir = Path(config['experiment']['cache_dir'])
    output_dir = Path('./outputs') / config_path.stem

    if config['experiment']['use_wandb']:
        os.environ['WANDB_MODE'] = 'online'
        initialize_wandb(config, config_path.stem, [__file__, args.config, current_dir / 'tree_enhanced_embeddings.py'])
    else:
        os.environ['WANDB_MODE'] = 'offline'
        logger.info('Wandb is not used.')

    set_seed(config['training']['seed'])

    dataset = load_dataset(config['data']['source'], cache_dir=cache_dir, num_proc=config['training']['num_workers'], split='test')
    columns_to_remove = [col for col in dataset.column_names if col not in ['input_ids', 'attention_mask']]
    if config['model']['extra_embeddings']:
        columns_to_remove = [col for col in columns_to_remove if col not in ['depths', 'sibling_indices', 'tree_attention_mask']]
    dataset.set_format(type='torch', columns=dataset.column_names)
    dataset = dataset.remove_columns(columns_to_remove)
    logger.info(f'Dataset sizes - Test: {len(dataset)}')
    logger.info(f'Dataset columns: {dataset.column_names}')

    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1", cache_dir='./cache/')
    model_config = AutoConfig.from_pretrained("huggingface/CodeBERTa-small-v1", cache_dir='./cache/')
    if config['model']['extra_embeddings']:
        model = TreeEnhancedRobertaForMaskedLM.from_pretrained(output_dir / 'final-model', config=model_config, yaml_config=config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(output_dir / 'final-model')
    logger.info(f'Loaded model: {model.__class__.__name__}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info('Evaluating on test set')
    output = evaluate_model_mlm(model, dataset, tokenizer, config['training']['batch_size'], device, config)
    logger.info(f"Test loss: {output['loss']}")
    logger.info(f"Test accuracy: {output['accuracy']}")
    logger.info(f"Test precision: {output['precision']}")
    logger.info(f"Test recall: {output['recall']}")
    logger.info(f"Test F1 score: {output['f1']}")

    if config['experiment']['use_wandb']:
        wandb.log(output)

if __name__ == '__main__':
    main()