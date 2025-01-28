import wandb

import os
import yaml
import argparse
import logging
import random
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tree_enhanced_codeberta import TreeEnhancedCodeBERTa
from tree_enhanced_codeberta_clone_detector import TreeEnhancedCodeBERTaCloneDetection

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def initialize_wandb(config, name, files_to_save):
    wandb.init(project=config['experiment']['wandb_project'] + '-finetune-clone-detection', config=config, name=name)
    for file in files_to_save:
        wandb.save(file)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    parser = argparse.ArgumentParser(description='Training script for TreeStarEncoder')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    current_dir = Path(__file__).parent
    cache_dir = Path(config['experiment']['cache_dir'])
    output_dir = Path('./outputs_finetune_clone_detection') / config_path.stem

    if config['experiment']['use_wandb']:
        os.environ['WANDB_MODE'] = 'online'
        initialize_wandb(config, config_path.stem, [__file__, args.config])
    else:
        os.environ['WANDB_MODE'] = 'offline'
        logger.info('Wandb is not used.')

    set_seed(config['training']['seed'])

    dataset = load_dataset(
        config['finetune_clone_detection']['data_source'],
        cache_dir=cache_dir,
        num_proc=config['training']['num_workers']
    )
    columns_to_remove = [col for col in dataset['train'].column_names if col not in [
        'input_ids_1', 'attention_mask_1', 'input_ids_2', 'attention_mask_2', 'labels'
    ]]
    if config['model']['extra_embeddings']:
        columns_to_remove = [col for col in columns_to_remove if col not in [
            'depths_1', 'sibling_indices_1', 'tree_attention_mask_1', 'depths_2', 'sibling_indices_2', 'tree_attention_mask_2'
        ]]
    dataset = dataset.remove_columns(columns_to_remove)
    test_valid = dataset['val'].train_test_split(test_size=0.5, seed=config['training']['seed'])
    dataset = DatasetDict({
        'train': dataset['train'],
        'valid': test_valid['train'],
        'test': test_valid['test'],
    })
    dataset['valid'] = dataset['valid'].select(range(5000))
    logger.info(f'Dataset sizes - Train: {len(dataset["train"])}, Valid: {len(dataset["valid"])}, Test: {len(dataset["test"])}')
    logger.info(f'Dataset columns: {dataset["train"].column_names}')

    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1", cache_dir='./cache/')
    model_config = AutoConfig.from_pretrained("huggingface/CodeBERTa-small-v1", cache_dir='./cache/')
    prev_model_path = config['finetune_clone_detection']['model_path']
    if config['model']['extra_embeddings']:
        prev_model = TreeEnhancedCodeBERTa.from_pretrained(prev_model_path, config=model_config, yaml_config=config)
    else:
        prev_model = AutoModelForMaskedLM.from_pretrained(prev_model_path)
    logger.info(f'Loaded previous model: {prev_model.__class__.__name__}')

    model_config.num_labels = 2
    model = TreeEnhancedCodeBERTaCloneDetection(model_config, config)
    logger.info(f'Loaded new model: {model.__class__.__name__}')

    model.roberta = prev_model.roberta
    logger.info('Attached previous model weights to new model')

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        learning_rate=config['finetune_clone_detection']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        max_steps=config['finetune_clone_detection']['max_steps'],
        warmup_steps=config['finetune_clone_detection']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=config['finetune_clone_detection']['logging_steps'],
        eval_steps=config['finetune_clone_detection']['eval_every'],
        save_steps=config['finetune_clone_detection']['eval_every'],
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        report_to='wandb' if config['experiment']['use_wandb'] else None,
        run_name=config_path.stem,
        seed=config['training']['seed'],
        fp16=config['training']['fp16'],
        dataloader_num_workers=config['training']['num_workers'],
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        compute_metrics=compute_metrics,
    )

    logger.info('Starting training')
    trainer.train()
    logger.info('Training completed')

    logger.info('Evaluating on test set')
    eval_results = trainer.evaluate(dataset['test'])
    logger.info(f'Evaluation results: {eval_results}')
    wandb.log({
        'test_loss': eval_results['eval_loss'],
        'test_accuracy': eval_results['eval_accuracy'],
        'test_precision': eval_results['eval_precision'],
        'test_recall': eval_results['eval_recall'],
        'test_f1': eval_results['eval_f1']
    })


if __name__ == '__main__':
    main()