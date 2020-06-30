# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

from functools import partial

import numpy as np
import pandas
import tensorflow as tf

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def convert_single_example(tokenizer, example, max_seq_length=512):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    #print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=512):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
#     for example in tqdm_notebook(examples, desc="Converting examples to features"):
    for example in (examples):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    
    return (
        np.array(input_ids), np.array(input_masks), np.array(segment_ids), np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
#             InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
            InputExample(guid=None, text_a=text, text_b=None, label=label)            
        )
    return InputExamples

def convert_text_to_features(texts, labels, tokenizer, max_seq_len) :

    example = InputExample(guid=None, text_a=texts, text_b=None, label=labels)
    input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_len)   

    return (np.array(input_id), np.array(input_mask),np.array(segment_id)),np.array(label).reshape(1)

def read_csvs(csv_files):
    sets = []
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        sets.append(file)
    # Concat all sets, drop any extra columns, re-index the final result as 0..N
    return pandas.concat(sets, join='inner', ignore_index=True)

def create_dataset(csvs, batch_size, tokenizer, max_seq_len, isinfer=False):
    csvs = csvs.split(',')
    df = read_csvs(csvs)
    if isinfer:
        df['label_id'] = 0

    def generate_values() :
        for text, label_id in zip(df.text, df.label_id) :
            yield convert_text_to_features(text, label_id, 
                                tokenizer=tokenizer,max_seq_len=max_seq_len)

    dataset = tf.data.Dataset.from_generator(generate_values, 
                                            output_types=((tf.int32, tf.int32, tf.int32), tf.int32),
                                            output_shapes=((tf.TensorShape([max_seq_len]), 
                                                           tf.TensorShape([max_seq_len]), tf.TensorShape([max_seq_len])), 
                                                           tf.TensorShape([1])))
    
    dataset = dataset.repeat()
    dataset = dataset.shuffle(100, reshuffle_each_iteration = True).batch(batch_size, drop_remainder=True)

    return dataset
