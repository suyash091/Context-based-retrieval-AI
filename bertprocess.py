
import numpy as np

class UbuntuStructure:
    def __init__(self, utterance, context, label):
        self.utterance = utterance
        self.context = context
        self.label = label

    def preprocess(self):
        utterance = self.utterance
        context = self.context
        label = self.label

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Tokenize utterance
        tokenized_utterance = tokenizer.encode(utterance)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_utterance.ids[1:]

        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_utterance.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            token_type_ids = token_type_ids[:max_len]
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label = label
        self.context_token_to_char = tokenized_context.offsets



def create_ubuntu_examples(df):
    ubuntu_examples = []
    i=0
    for index, row in df.iterrows():
                ubuntu = UbuntuStructure(
                    row[1], row[0], row[2]
                )
                ubuntu.preprocess()
                ubuntu_examples.append(ubuntu)
    return ubuntu_examples

def create_inputs_targets(ubuntu_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "label": []
    }
    for item in ubuntu_examples:
        if True:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = dataset_dict["label"]
    return x, y