import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bertprocess import create_inputs_targets, create_ubuntu_examples
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel, BertConfig
import gc

class TFIDFmodel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values, data.Utterance.values))

    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]


def random_model(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)

class CABert:
    def __init__(self,model_name,max_len):
        self.max_len=max_len
        if model_name.lower()=='dnn':
            print("Creating DNN Model")
            self.create_model = self.bertDNN_model

    def bertDNN_model(self):
        max_len=int(self.max_len)
        ## BERT encoder
        encoder = TFBertModel.from_pretrained("bert-base-uncased")

        ## QA Model
        input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
        token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
        embedding = encoder(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]
        Den0=layers.Dense(units = 256,activation ='relu')(embedding)
        Den1=layers.Dense(units = 128,activation ='relu')(Den0)
        Den2=layers.Dense(units = 64,activation ='relu')(Den1)
        Drop1=layers.Dropout(0.1)(Den2)
        label = layers.Dense(1, name="start_logit",activation ='sigmoid', use_bias=False)(Drop1)

        model = keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=[label],
        )
        optimizer = keras.optimizers.Adam(lr=5e-5)
        model.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def LoadModel(self,use_tpu):
        if use_tpu:
            # Create distribution strategy
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            # Create model
            print("Loading model on a TPU")
            with strategy.scope():
                self.model = self.create_model()
        else:

            print("Loading model on a CPU/GPU")
            self.model = self.create_model()

        print("Model's Architecture:")
        self.model.summary()

    def trainModel(self,df,epoch,steps):
        max_len=len(df)
        for i in range(epoch):
            print('-----------------------------------------')
            print('-----------------------------------------')
            print('-----------------------------------------')
            print('-----------------------------------------')
            print('Epoch: ' + str(i))
            print('-----------------------------------------')
            print('-----------------------------------------')
            print('-----------------------------------------')
            print('-----------------------------------------')
            gc.collect()
            for j in range(steps):
                print('Training Bucket: ' + str(j+1))
                train_ubuntu_examples = create_ubuntu_examples(df[j * int(max_len/steps):(j + 1) * int(max_len/steps)],int(self.max_len))
                x_train, y_train = create_inputs_targets(train_ubuntu_examples)
                print(f"{len(train_ubuntu_examples)} training points created.")
                self.model.fit(
                    x_train,
                    y_train,
                    epochs=1,  # For demonstration, 3 epochs are recommended
                    verbose=2,
                    batch_size=100,
                )
                del train_ubuntu_examples, x_train, y_train
        return self.model


