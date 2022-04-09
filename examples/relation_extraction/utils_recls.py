import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from filelock import FileLock
from transformers import PreTrainedTokenizer
import json
import argparse
import sys
from collections import Counter

import torch
from torch import nn
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


LABELS = [
    'no_relation', 'per:title', 'org:top_members/employees', 'per:employee_of', 
    'org:alternate_names', 'org:country_of_headquarters', 'per:countries_of_residence', 
    'org:city_of_headquarters', 'per:cities_of_residence', 'per:age', 'per:stateorprovinces_of_residence', 
    'per:origin', 'org:subsidiaries', 'org:parents', 'per:spouse', 'org:stateorprovince_of_headquarters', 
    'per:children', 'per:other_family', 'per:alternate_names', 'org:members', 'per:siblings', 
    'per:schools_attended', 'per:parents', 'per:date_of_death', 'org:member_of', 'org:founded_by', 
    'org:website', 'per:cause_of_death', 'org:political/religious_affiliation', 'org:founded', 
    'per:city_of_death', 'org:shareholders', 'org:number_of_employees/members', 'per:date_of_birth', 
    'per:city_of_birth', 'per:charges', 'per:stateorprovince_of_death', 'per:religion', 
    'per:stateorprovince_of_birth', 'per:country_of_birth', 'org:dissolved', 'per:country_of_death'
]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class InputExample:
    example_id: str
    tokens: List[str]
    pos: List[str]
    ner: List[str]
    deprel: List[str]
    head: List[str]
    subj_span: List[int]
    obj_span: List[int]
    subj_type: str
    obj_type: str
    label: str


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    obj_mask: Optional[List[int]] = None
    subj_mask: Optional[List[int]] = None
    label: Optional[int] = None


class TacredDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.
    def __init__(self,
                data_dir: str,
                tokenizer: PreTrainedTokenizer,
                labels: List[str],
                special_tokens: dict,
                model_type: str,
                max_seq_length: Optional[int] = None,
                overwrite_cache=False,
                mode: Split = Split.train,):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
        )
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                examples = read_examples_from_file(data_dir, mode)
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        special_tokens,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.json")
    examples = []
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        for d in data:
            l = len(d['token'])
            examples.append(InputExample(d['id'], 
                                         [convert_token(t) for t in d['token']], 
                                         d['stanford_pos'], 
                                         d['stanford_ner'], 
                                         d['stanford_deprel'], 
                                         d['stanford_head'], 
                                         [d['subj_start'], d['subj_end']],
                                         [d['obj_start'], d['obj_end']],
                                         d['subj_type'], 
                                         d['obj_type'],
                                         d['relation']))
    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    special_tokens=None,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # map label to id
    label_map = {label: i for i, label in enumerate(label_list)}

    def get_special_token(w):
        """ Get special token map"""
        if w not in special_tokens:
            special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]

    num_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0 
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        
        SUBJECT_NER = get_special_token("SUBJ=%s" % example.subj_type)
        OBJECT_NER = get_special_token("OBJ=%s" % example.obj_type)

        label = label_map[example.label]
        tokens = []
        subj_mask = []
        obj_mask = []
        subj_tokens = []
        obj_tokens = []
        subj_span, obj_span = example.subj_span, example.obj_span
        for i, token in enumerate(example.tokens):
            word_tokens = tokenizer.tokenize(token)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                if i == subj_span[0]:
                    subj_mask.extend([1])
                    obj_mask.extend([0])
                    tokens.extend([SUBJECT_NER])
                if i == obj_span[0]:
                    obj_mask.extend([1])
                    subj_mask.extend([0])
                    tokens.extend([OBJECT_NER])
                if i >= subj_span[0] and i <= subj_span[1]:
                    subj_tokens.extend(word_tokens)
                elif i >= obj_span[0] and i <= obj_span[1]:
                    obj_tokens.extend(word_tokens)
                else:
                    tokens.extend(word_tokens)
                    obj_mask.extend([0] * len(word_tokens))
                    subj_mask.extend([0] * len(word_tokens))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        # special_tokens_count += (len(subj_tokens) + len(obj_tokens) + 4) if special_tokens_count == 3 else (len(subj_tokens) + len(obj_tokens) + 2)
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            subj_mask = subj_mask[: (max_seq_length - special_tokens_count)]
            obj_mask = obj_mask[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        subj_mask += [0]
        obj_mask += [0]
        
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            subj_mask += [0]
            obj_mask += [0]

        # tokens += subj_tokens
        # tokens += [sep_token]
        # if sep_token_extra:
        #     # roberta uses an extra separator b/w pairs of sentences
        #     tokens += [sep_token]
        #     subj_mask += [0]
        #     obj_mask += [0]
        
        # tokens += obj_tokens
        # tokens += [sep_token]
        # if sep_token_extra:
        #     # roberta uses an extra separator b/w pairs of sentences
        #     tokens += [sep_token]
        #     subj_mask += [0]
        #     obj_mask += [0]

        # subj_mask += [0] * (len(subj_tokens) + len(obj_tokens) + 2)
        # obj_mask += [0] * (len(subj_tokens) + len(obj_tokens) + 2)

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            subj_mask += [0]
            obj_mask += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            subj_mask = [0] + subj_mask
            obj_mask = [0] + obj_mask
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            subj_mask = ([0] * padding_length) + subj_mask
            obj_mask = ([0] * padding_length) + obj_mask
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            subj_mask += [0] * padding_length
            obj_mask += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(subj_mask) == max_seq_length
        assert len(obj_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("examlpe_id: %s", example.example_id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("subj_mask: %s", " ".join([str(x) for x in subj_mask]))
            logger.info("obj_mask: %s", " ".join([str(x) for x in obj_mask]))
            logger.info("label: %s", label)

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids, 
                subj_mask=subj_mask, obj_mask=obj_mask, label=label
            )
        )

    return features


def get_positions(start_idx, end_idx, length):
    """Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) \
        + list(range(1, length - end_idx))


def get_labels() -> List[str]:
    return LABELS


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_f1(preds, labels):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}

import argparse
import sys
from collections import Counter

NO_RELATION = "no_relation"

def score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print( "Precision (micro): {:.3%}".format(prec_micro) )
    print( "   Recall (micro): {:.3%}".format(recall_micro) )
    print( "       F1 (micro): {:.3%}".format(f1_micro) )
    return {"Precision (micro)": prec_micro, "Recall (micro)": recall_micro, "F1 (micro)": f1_micro}

if __name__ == "__main__":
    from transformers import AutoTokenizer
    data_dir = "data/tacred_rev/"
    # model_path = "bert-large-uncased"
    model_path = "data/spanbert_hf/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # examples = read_examples_from_file(data_dir, Split.train)
    # print(examples[0])
    special_tokens = torch.load("data/tacred_rev/special_tokens.json")
    train_dataset = TacredDataset(data_dir, tokenizer, get_labels(), special_tokens, 'spanbert', 128, True)
    test_dataset = TacredDataset(data_dir, tokenizer, get_labels(), special_tokens, 'spanbert', 128, True, Split.test)
    eval_dataset = TacredDataset(data_dir, tokenizer, get_labels(), special_tokens, 'spanbert', 128, True, Split.dev)
    torch.save(special_tokens, "data/tacred_rev/special_tokens.json")
    