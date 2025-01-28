import logging
import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjavascript
import tree_sitter_go as tsgo
import tree_sitter_ruby as tsruby
import tree_sitter_php as tsphp

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def enhance_code(code: str, parser: Parser) -> list:
    code_tokens = []
    tree_attention_mask = []
    depths = []
    sibling_indices = []

    def traverse_tree(cursor, depth=0, sibling_index=0):
        node_text = cursor.node.text.decode("utf-8")

        if cursor.goto_first_child():
            sinbling_index_count = 0
            while True:
                traverse_tree(cursor, depth + 1, sinbling_index_count)
                sinbling_index_count += 1
                if not cursor.goto_next_sibling():
                    break
            cursor.goto_parent()
        else:
            code_tokens.append(node_text)
            tree_attention_mask.append(1)
            depths.append(depth)
            sibling_indices.append(sibling_index)

    tree = parser.parse(bytes(code, 'utf8'))
    cursor = tree.walk()
    traverse_tree(cursor)
    return code_tokens, tree_attention_mask, depths, sibling_indices

def process_example(code: str, func_code_tokens: list, language: str, tokenizer: AutoTokenizer, max_length: int) -> dict:
    # Instantiate parser locally to avoid global references
    if language == 'python': parser = Parser(Language(tspython.language()))
    elif language == 'java': parser = Parser(Language(tsjava.language()))
    elif language == 'javascript': parser = Parser(Language(tsjavascript.language()))
    elif language == 'go': parser = Parser(Language(tsgo.language()))
    elif language == 'ruby': parser = Parser(Language(tsruby.language()))
    elif language == 'php': parser = Parser(Language(tsphp.language()))
    else: raise ValueError(f"Unsupported language: {language}")

    # Enhance code
    code_tokens, tree_attention_mask, depths, sibling_indices = enhance_code(code, parser)

    code_tokens_idx = 0
    func_code_tokens_idx = 0

    while code_tokens_idx < len(code_tokens) and func_code_tokens_idx < len(func_code_tokens):
        if code_tokens[code_tokens_idx] == func_code_tokens[func_code_tokens_idx]:
            code_tokens_idx += 1
            func_code_tokens_idx += 1
            continue
        else:
            tree_attention_mask[code_tokens_idx] = 0
            code_tokens_idx += 1

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

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'depths': depths,
        'sibling_indices': sibling_indices,
        'tree_attention_mask': tree_attention_mask
    }

def process_batch(batch, tokenizer, max_length):
    processed_input_ids = []
    processed_attention_mask = []
    processed_depths = []
    processed_sibling_indices = []
    processed_tree_attention_mask = []

    for code, func_code, language in zip(batch['whole_func_string'], batch['func_code_tokens'], batch['language']):
        output = process_example(code, func_code, language, tokenizer, max_length)
        processed_input_ids.append(output['input_ids'])
        processed_attention_mask.append(output['attention_mask'])
        processed_depths.append(output['depths'])
        processed_sibling_indices.append(output['sibling_indices'])
        processed_tree_attention_mask.append(output['tree_attention_mask'])

    return {
        'input_ids': processed_input_ids,
        'attention_mask': processed_attention_mask,
        'depths': processed_depths,
        'sibling_indices': processed_sibling_indices,
        'tree_attention_mask': processed_tree_attention_mask
    }

def main():
    dataset_name = 'code-search-net/code_search_net'
    logging.info(f"Using dataset {dataset_name}")

    num_proc = min(multiprocessing.cpu_count() - 1, 16)
    logger.info(f"Using {num_proc} processes") 

    dataset = load_dataset('code-search-net/code_search_net', cache_dir='./cache/', num_proc=num_proc, trust_remote_code=True)

    # Filter PHP because its bugged right now (AttributeError: module 'tree_sitter_php' has no attribute 'language')
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda x: x['language'] != 'php', num_proc=num_proc, desc=f"Filtering PHP for {split}")

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
    processed_dataset = processed_dataset.remove_columns(dataset['train'].column_names)

    logging.info(f"Processed dataset: {processed_dataset}")
    
    logging.info(f"Pushing dataset to hub")
    processed_dataset.push_to_hub('patrykbart/code_search_net_tree_enhanced')

    logging.info(f"Dataset pushed to hub")

if __name__ == '__main__':
    main()
