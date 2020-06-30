import tensorflow as tf
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
import model
from model import *
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification

# Reference: https://github.com/wangz10/contrastive_loss/blob/master/model.py
class UnitNormLayer(tf.keras.layers.Layer):
    '''Normalize vectors (euclidean norm) in batch to unit hypersphere.
    '''
    def __init__(self):
        super(UnitNormLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, axis=1)
        return input_tensor / tf.reshape(norm, [-1, 1])

def encoder_net():
    input_ids = tf.keras.Input(shape=(MAX_SEQ_LEN,), batch_size=BATCH_SIZE, name="input_ids", dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=(MAX_SEQ_LEN,), batch_size=BATCH_SIZE, name="attention_mask", dtype=tf.int32)
    token_type_ids = tf.keras.Input(shape=(MAX_SEQ_LEN,), batch_size=BATCH_SIZE, name="token_type_ids", dtype=tf.int32)

    bert_input = [input_ids, attention_mask, token_type_ids]
#     normalization_layer = UnitNormLayer()
    
#     input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]

    encoder = TFAlbertModel.from_pretrained('albert-base-v2')
    encoder.trainable = True

    embeddings = encoder(bert_input)
#     norm_embeddings = normalization_layer(embeddings)
    encoder_network = tf.keras.Model(inputs=bert_input, outputs=embeddings)

    return encoder_network

# Projector Network
def projector_net():
    projector = tf.keras.models.Sequential([
        Dense(128, activation="relu"),
#         UnitNormLayer()
    ])

    return projector

@tf.function
def train_step(bert_input, labels):
    with tf.GradientTape() as tape:
        r = encoder_r(bert_input, training=True)
        r = r[0][:,0,:] # take cls token
#         r = tf.keras.layers.GlobalAveragePooling1D()(r[0])
        r = normalization_layer(r)
        z = projector_z(r, training=True)
        z = normalization_layer(z)
        loss = losses.max_margin_contrastive_loss(z, labels, metric='cosine')

    gradients = tape.gradient(loss, 
        encoder_r.trainable_variables + projector_z.trainable_variables)
    optimizer.apply_gradients(zip(gradients, 
        encoder_r.trainable_variables + projector_z.trainable_variables))

    return loss

def supervised_model():
    input_ids = tf.keras.Input(shape=(MAX_SEQ_LEN,), batch_size=BATCH_SIZE, name="input_ids", dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=(MAX_SEQ_LEN,), batch_size=BATCH_SIZE, name="attention_mask", dtype=tf.int32)
    token_type_ids = tf.keras.Input(shape=(MAX_SEQ_LEN,), batch_size=BATCH_SIZE, name="token_type_ids", dtype=tf.int32)

    bert_input = [input_ids, attention_mask, token_type_ids]
    encoder_r.trainable = False
    r = encoder_r(bert_input, training=False)

#     encoder_r.trainable = True
#     r = encoder_r(bert_input, training=True)
    r = r[0][:,0,:]
    r = normalization_layer(r)
#     r = tf.keras.layers.GlobalAveragePooling1D()(r[0])
    outputs = Dense(Num_class)(r)

    supervised_model = tf.keras.Model(bert_input, outputs)

    return supervised_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--bucket", help="bucket name", default="petcharts")
    parser.add_argument("-t", "--traindata", help="train file", default="train.csv")
    parser.add_argument("-f", "--testdata", help="test file", default="test.csv")
    parser.add_argument(
        "-p", "--pretrained", help="pretrained model zip file", default="roberta.zip"
    )

    parser.add_argument(
        "-o",
        "--downstream",
        help="downstream model zip file",
        default="classifier.zip",
    )

    parser.add_argument("-c", "--classes", help="classes", type=int, default=26)
    parser.add_argument("-e", "--epochs", help="epochs", type=int, default=24)
    parser.add_argument("-b", "--batchsize", help="batchsize", type=int, default=32)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-H", "--host", help="object server")
    parser.add_argument("-A", "--accesskey", help="access key")
    parser.add_argument("-K", "--secretkey", help="secret key")
    parser.add_argument("--logdir", help="tensorboard logdir", default="./logs")
    parser.add_argument("--weightdecay", help="weight decay", type=float, default=0.01)
    parser.add_argument("--scheduler", help="scheduler type", default="linear")

    args = parser.parse_args()

    try:
        client = connect_server(args.host, args.accesskey, args.secretkey)
        load_object(client, args.bucket, args.traindata)
        load_object(client, args.bucket, args.testdata)
        load_object(client, args.bucket, args.pretrained)
    except:
        pass

    uncompress_object(args.pretrained, ".")
    train_df = pd.read_csv(args.traindata)
    test_df = pd.read_csv(args.testdata)
    Num_class = len(set(train_df.label.value_counts()))

    BATCH_SIZE = 32
    MAX_SEQ_LEN = 128
    REPEAT = 2
    NUM_TRAIN_ITERATION = REPEAT * int(len(train_df) / BATCH_SIZE)
    NUM_TEST_ITERATION = int(len(test_df) / BATCH_SIZE)
    print(NUM_TRAIN_ITERATION)
    print(NUM_TEST_ITERATION)    

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    # model = TFAlbertModel.from_pretrained('albert-base-v2')

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_df['text'], train_df['label_id'])
    val_examples = convert_text_to_examples(val_df['text'], val_df['label_id'])
    test_examples = convert_text_to_examples(test_df['text'], test_df['label_id'])
    pred_examples = convert_text_to_examples(pred_df['text'], np.zeros(len(pred_df) ))

    # Convert to features
    (train_input_ids, train_input_masks, train_segment_ids, train_labels ) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=MAX_SEQ_LEN)
    (val_input_ids, val_input_masks, val_segment_ids, val_labels ) = convert_examples_to_features(tokenizer, val_examples, max_seq_length=MAX_SEQ_LEN)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels ) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=MAX_SEQ_LEN)
    (pred_input_ids, pred_input_masks, pred_segment_ids, pred_labels ) = convert_examples_to_features(tokenizer, pred_examples, max_seq_length=MAX_SEQ_LEN)

    train_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": tf.constant( train_input_ids, dtype=tf.int32),
                                                      "attention_mask": tf.constant( train_input_masks, dtype=tf.int32),
                                                      "token_type_ids": tf.constant( train_segment_ids, dtype=tf.int32),}
                                                        ,tf.constant(train_labels))).repeat()
    train_dataset = train_dataset.shuffle(100, reshuffle_each_iteration=True).batch(BATCH_SIZE,drop_remainder=True).repeat(2)

    val_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": tf.constant( val_input_ids, dtype=tf.int32),
                                                      "attention_mask": tf.constant( val_input_masks, dtype=tf.int32),
                                                      "token_type_ids": tf.constant( val_segment_ids, dtype=tf.int32),}
                                                        ,tf.constant(val_labels)))
    val_dataset = val_dataset.batch(BATCH_SIZE,drop_remainder=False)

    test_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": tf.constant( test_input_ids, dtype=tf.int32),
                                                      "attention_mask": tf.constant( test_input_masks, dtype=tf.int32),
                                                      "token_type_ids": tf.constant( test_segment_ids, dtype=tf.int32),}
                                                        ,tf.constant(test_labels)))
    test_dataset = test_dataset.batch(BATCH_SIZE,drop_remainder=False)

    optimizer = tf.keras.optimizers.Adam()
    encoder_r = encoder_net()
    projector_z = projector_net()
    normalization_layer = UnitNormLayer()

    EPOCHS = 5
    LOG_EVERY = 1
    train_loss_results = []
    test_count = 0

    start = time.time()
    for epoch in tqdm(range(EPOCHS)):
        epoch_loss_avg = tf.keras.metrics.Mean()

        for (bert_input, labels) in train_dataset:
            loss = train_step(bert_input, labels)
            epoch_loss_avg.update_state(loss) 

        train_loss_results.append(epoch_loss_avg.result())

        if epoch % LOG_EVERY == 0:
            print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

    end = time.time()
    print('training duration is ', end - start)

    with plt.xkcd():
        plt.plot(train_loss_results)
        plt.title("Supervised Contrastive Loss")
        plt.show()

    supervised_classifier = supervised_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    supervised_classifier.compile(optimizer=optimizer,
        loss=loss,
        metrics=metric)

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
        restore_best_weights=True, verbose=2)

    supervised_classifier.summary()

    supervised_classifier.fit(train_dataset,steps_per_epoch= NUM_TRAIN_ITERATION,
        validation_data=val_dataset,
        epochs=2,
        callbacks= es)

    supervised_classifier.evaluate(test_dataset, steps = NUM_TEST_ITERATION )

    model.save_pretrained("./pretrained")
    compress_object(args.downstream, "./pretrained")

    try:
        save_object(client, args.bucket, args.downstream)
    except:
        pass    

