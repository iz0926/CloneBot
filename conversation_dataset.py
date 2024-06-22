import os
import pickle
import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path, block_size=512):
        # read csv file into a DataFrame
        df = pd.read_csv(file_path)
        # adjust block size to account for tokenizer's maximum length for a single sentence
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        # path to the cache directory and file name
        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        # check if cached features file exists and load it if not instructed to overwrite the cache
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            # If no cache file exists, create features from the dataset and save to cache
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = self.construct_conv(row, tokenizer)
                self.examples.append(conv)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # encodes each string in the row, appends an EOS token
    # reverses the order of the conversation (so that the most recent message comes first)
    # flattens the list of encoded tokens
    def construct_conv(self, row, tokenizer):
        flatten = lambda l: [item for sublist in l for item in sublist]
        conv = []
        # encode each string in the row and add EOS token at the end
        for x in row:
            if isinstance(x, str):
                conv += [tokenizer.encode(x) + [tokenizer.eos_token_id]]
        # reverse the order of the conversation and flatten it
        conv = list(reversed(conv))
        conv = flatten(conv)
        return conv

    def __len__(self):
        # return the number of examples in the dataset
        return len(self.examples)

    def __getitem__(self, item):
        # return the example at the given index as a tensor
        return torch.tensor(self.examples[item], dtype=torch.long)