import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

from parse_dataset import enhance_code
from utils import setup_logger

def process_example(code: str, tokenizer: AutoTokenizer, max_length: int) -> dict:
    # Instantiate parser locally to avoid global references
    parser = Parser(Language(tspython.language()))

    # Enhance code
    code_tokens, tree_attention_mask, depths, sibling_indices = enhance_code(code, parser)

    # Truncate inputs to max length - 2 for special tokens
    code_tokens = code_tokens[:max_length - 2]
    tree_attention_mask = tree_attention_mask[:max_length - 2]
    depths = depths[:max_length - 2]
    sibling_indices = sibling_indices[:max_length - 2]

    # Convert code tokens to input ids
    input_ids = tokenizer.convert_tokens_to_ids(code_tokens)

    # Add extra values for special tokens
    input_ids.insert(0, tokenizer.encode("")[0])
    input_ids.append(tokenizer.encode("")[1])
    depths.insert(0, -1)
    depths.append(-1)
    sibling_indices.insert(0, -1)
    sibling_indices.append(-1)
    tree_attention_mask.insert(0, 0)
    tree_attention_mask.append(0)

    # Create attention mask
    attention_mask = [1] * len(input_ids)

    # Pad inputs
    input_ids = input_ids + [0] * (max_length - len(input_ids))
    attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
    depths = depths + [-1] * (max_length - len(depths))
    sibling_indices = sibling_indices + [-1] * (max_length - len(sibling_indices))
    tree_attention_mask = tree_attention_mask + [0] * (max_length - len(tree_attention_mask))

    assert max_length == len(input_ids) == len(attention_mask) == len(depths) == len(sibling_indices) == len(tree_attention_mask)

    for idx, token in enumerate(tokenizer.convert_ids_to_tokens(input_ids)):
        if token == tokenizer.unk_token:
            tree_attention_mask[idx] = 0

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'depths': depths,
        'sibling_indices': sibling_indices,
        'tree_attention_mask': tree_attention_mask
    }

def process_batch(batch, tokenizer, max_length):
    processed_input_ids_1 = []
    processed_attention_mask_1 = []
    processed_depths_1 = []
    processed_sibling_indices_1 = []
    processed_tree_attention_mask_1 = []

    processed_input_ids_2 = []
    processed_attention_mask_2 = []
    processed_depths_2 = []
    processed_sibling_indices_2 = []
    processed_tree_attention_mask_2 = []

    for code_1, code_2 in zip(batch['code1'], batch['code2']):
        output_1 = process_example(code_1, tokenizer, max_length)
        output_2 = process_example(code_2, tokenizer, max_length)

        
        processed_input_ids_1.append(output_1['input_ids'])
        processed_attention_mask_1.append(output_1['attention_mask'])
        processed_depths_1.append(output_1['depths'])
        processed_sibling_indices_1.append(output_1['sibling_indices'])
        processed_tree_attention_mask_1.append(output_1['tree_attention_mask'])

        processed_input_ids_2.append(output_2['input_ids'])
        processed_attention_mask_2.append(output_2['attention_mask'])
        processed_depths_2.append(output_2['depths'])
        processed_sibling_indices_2.append(output_2['sibling_indices'])
        processed_tree_attention_mask_2.append(output_2['tree_attention_mask'])

    return {
        'input_ids_1': processed_input_ids_1,
        'attention_mask_1': processed_attention_mask_1,
        'depths_1': processed_depths_1,
        'sibling_indices_1': processed_sibling_indices_1,
        'tree_attention_mask_1': processed_tree_attention_mask_1,
        'input_ids_2': processed_input_ids_2,
        'attention_mask_2': processed_attention_mask_2,
        'depths_2': processed_depths_2,
        'sibling_indices_2': processed_sibling_indices_2,
        'tree_attention_mask_2': processed_tree_attention_mask_2,
        'labels': batch['similar']
    }

def main():
    logger = setup_logger()

    dataset_name = 'PoolC/1-fold-clone-detection-600k-5fold'
    logger.info(f"Using dataset {dataset_name}")

    num_proc = min(multiprocessing.cpu_count() - 1, 16)
    logger.info(f"Using {num_proc} processes") 

    dataset = load_dataset(dataset_name, cache_dir='./cache/', num_proc=num_proc, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1", cache_dir='./cache/')
    logger.info(f"Loaded dataset: {dataset}")

    processed_dataset = dataset.map(
        process_batch,
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': 512
        },
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc='Processing dataset'
    )

    # Delete all other columns
    columns_to_remove = dataset['train'].column_names
    processed_dataset = processed_dataset.remove_columns(columns_to_remove)

    logger.info(f"Processed dataset: {processed_dataset}")
    
    logger.info(f"Pushing dataset to hub")
    processed_dataset.push_to_hub('patrykbart/1-fold-clone-detection-600k-5fold-tree-enhanced')

    logger.info(f"Dataset pushed to hub")

if __name__ == '__main__':
    main()
