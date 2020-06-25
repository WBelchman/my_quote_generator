import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import re
from random import randint

import tensorflow.keras as keras
import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


def main():
    print("Quote Generator v3")
    print("Lets give this another shot")

    model = -1

    while True:
        print()
        print("1. Train\n2. Load\n3. Demo\n0. Exit")
        user = input()
        
        if user == "0":
            break
        elif user == "1":
            train()
        elif user == "2":
            model = load()
        elif user == "3":
            if model == -1: 
                print("[-] Please Load or train model before Demo")
                continue

            demo(model)

def load_dataset(filename="quotes.txt"):
    f = open(filename, "r", encoding="utf8")
    pattern = re.compile('[^A-Za-z., ]+', re.UNICODE)

    data = pattern.sub('', f.read())

    return data

def build_tokenizer(dataset):
    filename="./saved/tokenizer.json"

    if os.path.exists(filename):
        print("[*] Tokenizer found, loading...")

        with open("./saved/tokenizer.json", "r") as f:
            data = json.load(f)
            tokenizer = keras.preprocessing.text.tokenizer_from_json(data)

    else:
        print("[*] Encoder not found, building from dataset")

        tokenizer = keras.preprocessing.text.Tokenizer(char_level=True,
                                        lower=False,
                                        filters='!"#$%&()*+-/:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(dataset)

        with open("./saved/tokenizer.json", "w") as f:
            f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))

    print("[*] Tokenizer ready | vocab size {}".format(len(tokenizer.word_index)+1))
    print(tokenizer.word_index)
    return tokenizer

def tokenize_data(dataset, tokenizer):
    print("[*] Encoding data...")
    return tokenizer.texts_to_sequences([dataset])

def separate_labels(data):

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    return data.map(split_input_target)

def build_model(vocab_size, batch_size):
    model = tf.keras.Sequential([
    keras.layers.Embedding(vocab_size, 128,
                            mask_zero=True,
                            batch_input_shape=[batch_size, None]),
    keras.layers.GRU(1000,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    keras.layers.Dense(vocab_size)
  ])

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    model.summary()
    input("Press Enter to continue")

    return model

def get_callbacks():
    checkpoint_prefix = os.path.join("./checkpoints", "ckpt_{epoch}")

    save_callback = tf.keras.callbacks.ModelCheckpoint(
                                        filepath=checkpoint_prefix,
                                        save_weights_only=True)

    return [save_callback]

def train():
    data = load_dataset()

    seq_length = 100
    
    tokenizer = build_tokenizer(data)
    proc_data = tokenize_data(data, tokenizer)

    vocab_size = len(tokenizer.word_index) + 1

    partial_dataset = tf.data.Dataset.from_tensor_slices(proc_data[0])
    seq_dataset = partial_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = separate_labels(seq_dataset)

    dataset = dataset.shuffle(10000).batch(64, drop_remainder=True)

    model = build_model(vocab_size, 64)

    callbacks = get_callbacks()
    
    model.fit(dataset,
          batch_size=64,
          epochs=14,
          callbacks=callbacks)


def sample_text(model, tokenizer, vocab_size):
    #seed = [randint(1, vocab_size)]
    seed = tokenizer.texts_to_sequences(['.'])[0]
    model_input = tf.expand_dims(seed, 0)

    print(seed)

    text = []
    pred = ''

    temperatures = [1.0]

    print("\nPredictions:")

    for temp in temperatures:

        model.reset_states()
        i = 0
        while pred != "." and i < 1000: 
            pred_dist = model(model_input)

            pred_dist = tf.squeeze(pred_dist, 0) / temp
            pred_token = tf.random.categorical(pred_dist, num_samples=1)[-1, 0].numpy()

            model_input = tf.expand_dims([pred_token], 0)

            pred = tokenizer.sequences_to_texts([[pred_token]])[0]
            text.append(pred)

            i += 1

        print("Temp: {}".format(temp))
        print(''.join(text), "\n")


def load():

    tokenizer = build_tokenizer('') #You should have a tokenizer beforehand (choose train)

    vocab_size = len(tokenizer.word_index) + 1

    model = build_model(vocab_size, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint("./checkpoints")).expect_partial() #Is this a problem?
    model.build(tf.TensorShape([1, None]))

    user = ''
    while user != "q":
        sample_text(model, tokenizer, vocab_size)

        user = input("Press q to quit\n")

    return model


def demo(model):
    pass



if __name__ == "__main__":
    main()