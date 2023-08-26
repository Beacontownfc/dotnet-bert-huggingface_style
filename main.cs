using BERT;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;
using ConsoleApp1;

var batch_size = 4;
var learning_rate = (float)2e-5;
var num_classes = 2;
var max_seq_len = 180;
var epoch = 10;
var config = new BertConfig();
var pretrained_weight_path = "./bert_model.h5";
var dataset_path = "./datasets";
var vocab_file = "./vocab.txt";

var model = keras.Sequential();
model.add(keras.layers.Input(max_seq_len, batch_size, dtype: tf.int32));
model.add(new BertMainLayer(config));
model.load_weights(pretrained_weight_path);
model.add(keras.layers.Dense(num_classes));

model.compile(optimizer: keras.optimizers.AdamW(learning_rate, weight_decay: 0.01f, no_decay_params: new List<string> { "gamma", "beta" }),
    loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true), metrics: new[] { "acc" });
model.summary();

Console.WriteLine("Preparing data....");

(int[,] x_train_neg, int[] y_train_neg) = IMDBDataPreProcessor.
    ProcessData(dataset_path + "neg" ,max_seq_len, vocab_file, 0);

(int[,] x_train_pos, int[] y_train_pos) = IMDBDataPreProcessor.
    ProcessData(dataset_path + "pos", max_seq_len, vocab_file, 1);

var np_x_train = np.array(x_train_neg, dtype: tf.int32);
var np_y_train = np.array(y_train_neg, dtype: tf.int32);
np_x_train = np.concatenate((np_x_train, np.array(x_train_pos, dtype: tf.int32)), 0);
np_y_train = np.concatenate((np_y_train, np.array(y_train_pos, dtype: tf.int32)), 0);

Console.WriteLine("Start to train...");
model.fit(np_x_train, np_y_train,
    batch_size: batch_size,
    epochs: epoch,
    shuffle: true,
    validation_split: 0.2f);






