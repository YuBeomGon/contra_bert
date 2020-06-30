
import tensorflow as tf
print(tf.__version__)
import tensorflow_datasets
from transformers import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tqdm.notebook import tqdm
# from wandb.keras import WandbCallback
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
# !wget https://raw.githubusercontent.com/wangz10/contrastive_loss/master/losses.py
import losses
import feed
from feed import *
import time


from transformers.modeling_tf_utils import (
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    shape_list,
)

class TFAlbertForTextClassification(TFAlbertPreTrainedModel, TFSequenceClassificationLoss) :
    def __init__(self, config, *inputs, **kwargs):  
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels  
        print(kwargs)
        self.isSCR = True
        self.albert = TFAlbertMainLayer(config, name="albert")
        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
             config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier" )
        
#         input_ids = tf.keras.Input(shape=(max_seq_length,), batch_size=BATCH_SIZE, name="input_ids", dtype=tf.int32)
#             attention_mask = tf.keras.Input(shape=(max_seq_length,), batch_size=BATCH_SIZE, name="attention_mask", dtype=tf.int32)
#             token_type_ids = tf.keras.Input(shape=(max_seq_length,), batch_size=BATCH_SIZE, name="token_type_ids", dtype=tf.int32)

#         self.model_input = [input_ids, attention_mask, token_type_ids]  
        
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        training=False,
    ):
        
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            training=training,
        )

        pooled_output = outputs[1] #take cls token

        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.isSCR == True :
                loss = losses.max_margin_contrastive_loss(logits, labels, metric='cosine')
            else :
                loss = self.compute_loss(labels, logits)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)        

    
class TFAlbertTextClassification(TFAlbertForTextClassification) :    
    def __init__(self, config, *inputs, **kwargs):
        print('inputs :',inputs)
        print('kwargs :',kwargs)
        self.isSCR = kwargs['isSCR']
        kwargs.clear()
        super().__init__(config, *inputs, **kwargs)
   
