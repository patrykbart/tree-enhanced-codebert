import wandb

import os
import yaml
import argparse
import logging
import random
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from tree_enhanced_codeberta import TreeEnhancedCodeBERTa

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MonitoringTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        outputs = super().training_step(model, inputs, num_items_in_batch)

        if self.state.global_step % self.args.logging_steps == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()
                    wandb.log({f'grad_norm/{name}': grad_norm})

            for name, param in model.named_parameters():
                weight_norm = param.data.norm(2).item()
                wandb.log({f'weight_norm/{name}': weight_norm})

            if model.weighted_sum:
                wandb.log({f'word_weight': model.roberta.embeddings.word_weight.item()})
                wandb.log({f'token_type_weight': model.roberta.embeddings.token_type_weight.item()})
                wandb.log({f'position_weight': model.roberta.embeddings.position_weight.item()})
                wandb.log({f'depth_weight': model.roberta.embeddings.depth_weight.item()})
                wandb.log({f'sibling_index_weight': model.roberta.embeddings.sibling_index_weight.item()})

        return outputs

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
    wandb.init(project=config['experiment']['wandb_project'], config=config, name=name)
    for file in files_to_save:
        wandb.save(file)

def main():
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

    dataset = load_dataset(config['data']['source'], cache_dir=cache_dir, num_proc=config['training']['num_workers'])
    columns_to_remove = [col for col in dataset['train'].column_names if col not in ['input_ids', 'attention_mask']]
    if config['model']['extra_embeddings']:
        columns_to_remove = [col for col in columns_to_remove if col not in ['depths', 'sibling_indices', 'tree_attention_mask']]
    dataset = dataset.remove_columns(columns_to_remove)
    logger.info(f'Dataset sizes - Train: {len(dataset["train"])}, Valid: {len(dataset["validation"])}, Test: {len(dataset["test"])}')
    logger.info(f'Dataset columns: {dataset["train"].column_names}')

    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1", cache_dir='./cache/')
    model_config = AutoConfig.from_pretrained("huggingface/CodeBERTa-small-v1", cache_dir='./cache/')
    if config['model']['extra_embeddings']:
        model = TreeEnhancedCodeBERTa(model_config, config)
    else:
        model = AutoModelForMaskedLM.from_config(model_config)
    logger.info(f'Loaded model: {model.__class__.__name__}')

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        num_train_epochs=config['training']['epochs'],
        warmup_steps=config['training']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=config['evaluation']['logging_steps'],
        eval_steps=config['evaluation']['eval_every'],
        save_steps=config['evaluation']['eval_every'],
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        report_to='wandb' if config['experiment']['use_wandb'] else None,
        run_name=config_path.stem,
        seed=config['training']['seed'],
        fp16=config['training']['fp16'],
        dataloader_num_workers=config['training']['num_workers'],
        gradient_checkpointing=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=3,
    )

    trainer = MonitoringTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=config['data']['mlm_probability']),
    )

    logger.info('Starting training')
    trainer.train()
    logger.info('Training completed')

    logger.info('Evaluating on test set')
    eval_results = trainer.evaluate(dataset['test'])
    logger.info(f'Evaluation results: {eval_results}')
    wandb.log({'test_loss': eval_results['eval_loss']})

    logger.info('Saving model')
    trainer.save_model(output_dir / 'final-model')
    tokenizer.save_pretrained(output_dir / 'final-model')
    logger.info('Model saved')

    wandb.save(output_dir / 'final-model/*')
    logger.info('Model uploaded to W&B')


if __name__ == '__main__':
    main()