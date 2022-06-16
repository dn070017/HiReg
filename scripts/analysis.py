import os
import pandas as pd
import pickle as pkl
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
from utils import dataset_generator

def create_dataset():
  dataset = tf.data.Dataset.from_generator(
  dataset_generator,
  output_types={
      'nucl': tf.float32,
      'h3k4me3': tf.float32,
      'h3k27ac': tf.float32,
      'atacseq': tf.float32,
      'hicdist': tf.float32,
      'feature': tf.float32,
      'tad_idx': tf.float32,
      'y': tf.float32, 
    }
  )
  tf.data.experimental.save(dataset, 'datasets/tfsnapshot')

def std_of_hic():
  hicdist_dict = defaultdict(list)
  datasets = 'chr19_active'
  datasets_dir = os.path.join('datasets', datasets)
  tad_dir_list = os.listdir(datasets_dir)

  tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, split=',')
  tokenizer.fit_on_texts(['atcg'])

  for tad_idx, tad_name in tqdm(enumerate(tad_dir_list)):
    tad_dir = os.path.join(datasets_dir, tad_name)
    bin_dir_list = os.listdir(tad_dir)
    
    for bin_name in bin_dir_list:
      bin_dir = os.path.join(tad_dir, bin_name)
      if not os.path.isdir(bin_dir):
        continue

      with open(os.path.join(bin_dir, 'dists.pickle'), 'rb') as tfbs_file: 
        hicdist_df = pkl.load(tfbs_file)
      
      hicdist_dict[tad_idx].append(hicdist_df['HiC'])
    
    if len(hicdist_dict[tad_idx]) != 0:
      hicdist_dict[tad_idx] = pd.concat(hicdist_dict[tad_idx], axis=1)

  mean_of_stds = []
  for k, v in hicdist_dict.items():
    if len(v) != 0 and v.shape[1] != 0:
      mean_of_stds.append(v.std(axis=1).mean())