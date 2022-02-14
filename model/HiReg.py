
import mlflow
import numpy as np
import tensorflow as tf
from time import time

@tf.function
def tf_parallel_map(*args, **kwargs):
  # this function is used to map function on an axis using parallel computing
  return tf.map_fn(*args, **kwargs)

def compute_angles(pos, i, dims):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dims))
  return pos * angle_rates

def positional_encoding(position, dims):
  angle_rads = compute_angles(
    np.arange(position)[:, np.newaxis],
    np.arange(dims)[np.newaxis, :],
    dims
  )
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

class AttentionPoolingLayer(tf.keras.layers.Layer):
  def __init__(
    self,
    output_dim=3,
    num_attn_heads=2,
    num_queries=4,
    initializer=tf.keras.initializers.glorot_normal(),
    regularizer=tf.keras.regularizers.l2(0.0005),
    activation='elu',
    is_last=False
  ):
    super().__init__()
    self.initializer = initializer
    self.regularizer = regularizer
    self.num_attn_heads = num_attn_heads
    self.output_dim = output_dim
    self.query = self.add_weight(
      'query', (num_queries, output_dim),
    )
    self.activation = tf.keras.activations.get(activation)
    self.is_last = is_last
    
  def build(self, input_shape):
    input_dim = int(input_shape[-1])
    self.W_key = self.add_weight(f"W_key", shape=(self.num_attn_heads, input_dim, self.output_dim), initializer=self.initializer, regularizer=self.regularizer)
    self.W_val = self.add_weight(f"W_val", shape=(self.num_attn_heads, input_dim, self.output_dim), initializer=self.initializer, regularizer=self.regularizer)
    self.biases = self.add_weight(f"bias", shape=(self.num_attn_heads, self.output_dim))
    
    return

  def call(self, inputs):
    X = inputs
    
    # key/val.shape: (batch_nodes, num_attn_heads, F')
    key = tf.tensordot(X, self.W_key, axes=[[-1], [1]]) 
    #key = self.dropout(key)
    val = tf.tensordot(X, self.W_val, axes=[[-1], [1]])

    # key/val.shape: (num_attn_heads, batch_nodes, F')
    key = tf.transpose(key, [1, 0, 2])
    val = tf.transpose(val, [1, 0, 2])

    # unnorm_attn.shape: (num_attn_heads, batch_nodes, num_queries)
    unnorm_attn = tf.tensordot(key, self.query, axes=[[2], [1]])
    # unnorm_attn.shape: (num_attn_heads, num_queries, batch_nodes)
    unnorm_attn = tf.transpose(unnorm_attn, [0, 2, 1])

    unnorm_attn = tf.keras.layers.LeakyReLU(alpha=0.2)(unnorm_attn)
    norm_attn = tf.keras.layers.Softmax()(unnorm_attn)
    #norm_attn = self.dropout(norm_attn)
    #print(tf.reduce_sum(norm_attn, axis=2))

    # results.shape: (num_attn_heads, num_queries, F')
    # [TODO] this funciton needs to be optimized
    results, _, _ = tf_parallel_map(AttentionPoolingLayer.attention_pooling, (val, norm_attn, self.biases))
    
    # results.shape: (num_attn_heads * num_queries, F')
    results = tf.reshape(results, (-1, self.output_dim))

    # results.shape: (1, F')
    if self.is_last:
      results = tf.reduce_mean(results, axis=0)

    results = self.activation(results)

    return results

  @staticmethod
  def attention_pooling(x):
    # [TODO] this funciton needs to be optimized
    val = x[0]
    attn = x[1]
    bias = x[2]

    res = tf.tensordot(attn, val, axes=[[-1], [0]]) + bias

    return res, res, res

class HiReg(tf.keras.Model):
  def __init__(
    self,
    input_dim=4,
    num_filters=16,
    kernel_size=16,
    latent_dim=32,
    num_attn_heads=2,
    num_queries=4,
    dropout_rate=0.6,
    rpe_bin_size=1000,
    max_rpe=100000
  ):
    super().__init__()
    
    self.input_dim = input_dim
    self.training = True
    # Positional Embedding
    self.max_rpe = max_rpe // rpe_bin_size
    self.pos_encoding = positional_encoding(self.max_rpe * 2 + 1, num_filters + input_dim)

    # Dropout Layer
    self.dropout_rate = dropout_rate
    self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
    self.batch_norm = tf.keras.layers.BatchNormalization()

    # Convolutional Layer
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.convolution = tf.keras.layers.Conv1D(
      filters=self.num_filters,
      kernel_size=self.kernel_size,
      strides=1,
      dilation_rate=1,
      #activation=tf.keras.activations.relu,
      padding='same'
    )
    self.conv_pooling = tf.keras.layers.GlobalMaxPool1D()

    # Attention Pooling Layer
    self.latent_dim = latent_dim
    self.num_attn_heads = num_attn_heads
    self.num_queries = num_queries
    self.attention_pooling = tf.keras.Sequential([
      AttentionPoolingLayer(
        output_dim=self.latent_dim,
        num_attn_heads=self.num_attn_heads,
        num_queries=self.num_queries,
        dropout_rate=self.dropout_rate
      ),
      AttentionPoolingLayer(
        output_dim=self.latent_dim,
        num_attn_heads=1,
        num_queries=1,
        dropout_rate=self.dropout_rate
      )
    ])
    
    # Fully Connected Layer
    self.last_layer = tf.keras.Sequential([
      tf.keras.layers.Dense(self.latent_dim, activation='relu'),
      tf.keras.layers.Dense(1),
    ])
  
  def train_step(self, data):
    # [TODO]
    pass

  def call(self, inputs):
    
    # lower the resolution of relative position
    rpe = tf.where(inputs['RPE'] < -1 * self.max_rpe, -1 * self.max_rpe, inputs['RPE'])
    rpe = tf.where(rpe > self.max_rpe, self.max_rpe, rpe)
    rpe += self.max_rpe
    rpe = tf.cast(rpe, tf.int32)

    # get the corresponding positional encoding
    rpe = tf.gather(self.pos_encoding, rpe, axis=1)[0]

    # forward pass
    seq_features = self.conv_pooling(self.convolution(inputs['seq']))
    all_features = tf.concat([seq_features, inputs['x']], axis=-1)
    all_features = self.batch_norm(all_features, training=self.training)
    attn_pooled_features = self.attention_pooling(all_features + rpe)
    result = self.last_layer(attn_pooled_features)

    return result

if __name__ == '__main__':

  # use the dataset
  dataset = tf.data.experimental.load(
    'datasets/tfsnapshot'
  )

  # mlflow setup
  experiment_name = "HiReg-Devel"
  mlflow.set_experiment(experiment_name)
  experiment = mlflow.get_experiment_by_name(experiment_name)

  # hyperparameter setup
  num_filters = 64
  num_attn_heads = 1
  num_queries = 64
  latent_dim = 64
  dropout_rate = 0
  num_epochs = 20
  learning_rate = 5e-3
  early_stop = 0.999
  early_stop_patience = 20

  with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    
    mlflow.log_params({
      'num_filters': num_filters, 
      'num_attn_heads': num_attn_heads,
      'num_queries': num_queries,
      'latent_dim': latent_dim,
      'dropout_rate': dropout_rate,
      'num_epochs': num_epochs,
      'learning_rate': learning_rate,
      'early_stop': early_stop,
      'early_stop_patience': early_stop_patience
    })

    model = HiReg(
      num_filters=num_filters,
      num_attn_heads=num_attn_heads,
      num_queries=num_queries,
      latent_dim=latent_dim,
      dropout_rate=dropout_rate,
    )
    
    # define parameters for early stopping
    patience = early_stop_patience
    best_loss = 1e6

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
    
    # start training
    start_time = time()
    for epoch in range(num_epochs):
      # used to log the average MSE
      mean_loss_epoch = []
      for batch in dataset:
        with tf.GradientTape() as tape:
          y_pred = model(batch)
          loss = tf.reduce_mean((batch['y'] - y_pred) ** 2)
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
          mean_loss_epoch.append(loss)

      mean_loss_epoch = np.mean(np.array(mean_loss_epoch))
      mlflow.log_metrics({'mse': mean_loss_epoch}, step=epoch+1)
      print(f"{epoch+1:>3}\t{mean_loss_epoch:>10f}\t{patience}")

      # check if the condition for early stopping is met
      if mean_loss_epoch < best_loss * early_stop:
        patience = early_stop_patience
        best_loss = mean_loss_epoch
      else:
        patience -= 1

      if patience == 0:
        break
      
  end_time = time()
  print('Completed')
  print(f"{end_time - start_time:.3f}")