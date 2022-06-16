#%%
import os
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm

def save_dataset(dataset, path='datasets/tfsnapshot'):
  tf.data.experimental.save(dataset, path)

def dataset_generator_legacy(path='datasets', cage_peak_file='K562_cage_tbl.tsv', cage_peak_target='m'):
  all_rpe = []    # just to record all relative position w.r.t CAGE peak
  pos_strand = 0  # just to count how many TFBS on positive strand
  neg_strand = 0  # just to count how many TFBS on negative strand

  # column names for epigenomics feature
  feature_cols = ['score', 'H3K4me3', 'H3K27Ac', 'ATAC']

  # tokenize the TFBS sequence
  tk = tf.keras.preprocessing.text.Tokenizer(char_level=True)
  # all sequence will be transform into lowercase, therefore fit on 'atcg'
  tk.fit_on_texts(['atcg'])

  # file that define all CAGE peaks
  cage_peaks = pd.read_csv(os.path.join(path, cage_peak_file), sep='\t', index_col=0)
  
  cage_list_path = os.path.join(path, 'CAGE_list.txt')
  cage_list = []

  # read CAGE list line-by-line
  # each line looks like this: chr1_17307102_17307157_-
  # the line is also the name of the directory for all features 
  with open(cage_list_path, 'r') as cage_file:
    for file_line in cage_file:
      cage = file_line.strip()
      cage_list.append(cage)
      # take the first position as cage_pos
      cage_pos = int(cage.split('_')[1])
      
      # read TFBS features
      tfbs_path = os.path.join(path, 'TAD_42_cage_peaks', cage, 'TFBSs.pickle')
      with open(tfbs_path, 'rb') as tfbs_file: 
        tfbs_df = pkl.load(tfbs_file)

      # convert sequence to lowercase
      tfbs_df.seq = tfbs_df.seq.str.lower()

      # normalize score
      tfbs_df.score = tfbs_df.score.astype(int) / 1000
      tfbs_df.start = tfbs_df.start.astype(int)
      tfbs_df.end = tfbs_df.end.astype(int)

      # compute relative position w.r.t to CAGE peak
      tfbs_df['RPE'] = tfbs_df.start - cage_pos
      
      all_rpe.append(tfbs_df['RPE'])              # just to record all relative position w.r.t CAGE peak
      pos_strand += (tfbs_df.strand == '+').sum() # just to count how many TFBS on positive strand
      neg_strand += (tfbs_df.strand == '-').sum() # just to count how many TFBS on negative strand
    
      rpe = tf.convert_to_tensor(tfbs_df.RPE) # change variable name

      tfbs_feature = tf.convert_to_tensor(tfbs_df[feature_cols].astype(float)) # change variable name

      # one-hot encoding for sequence shape: (51, 4)
      tfbs_seq = tf.one_hot(tf.convert_to_tensor(tk.texts_to_sequences(tfbs_df.seq)), depth=4)

      # read adjancency matrix (not used for now)
      adj_path = os.path.join(path, 'TAD_42_cage_peaks', cage, 'adj_mat.pickle')
      with open(adj_path, 'rb') as adj_file: 
        adj_df = pkl.load(adj_file)
      np.fill_diagonal(adj_df.values, 1.0)

      # read transcription factor binding site chromosome region
      # transform string region into integer, starting from 0
      # encoded_region is the id mapping with key: chromosome_region (CAGE-peak), value: integer 
      # TODO change to list
      encoded_region = {}
      for i, region_ID in enumerate(adj_df.columns):
        encoded_region[region_ID] = i
      tfbs_region = tf.convert_to_tensor(tfbs_df.region_ID.replace(encoded_region).values)
      adj = tf.convert_to_tensor(adj_df.values)

      cage_peak = tf.convert_to_tensor(cage_peaks.loc[cage, cage_peak_target])

      yield {
        'x': tfbs_feature,     # epigenomic features
        'RPE': rpe,            # relative positional embedding
        'seq': tfbs_seq,       # sequence
        'region': tfbs_region, # integer encoded chromosome_region
        'adj': adj,            # adjancency matrix (not used for now)
        'y': cage_peak         # prediction target
      }

def load_dataset(path='datasets/tfsnapshot'):
  return tf.data.experimental.load(path)

def split_to_tensor(df, field='H3K4me3', delimiter=','):
  return tf.math.log(tf.expand_dims(
    tf.convert_to_tensor([list(map(lambda y: float(y), x)) for x in df[field].str.split(delimiter)]),
    axis=-1
  ) + 1)

def dataset_generator(datasets='chr19_active'):

  datasets_dir = os.path.join('datasets', datasets)
  tad_dir_list = os.listdir(datasets_dir)

  tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, split=',')
  tokenizer.fit_on_texts(['atcg'])

  for tad_idx, tad_name in tqdm(enumerate(tad_dir_list)):
    tad_dir = os.path.join(datasets_dir, tad_name)
    
    with open(os.path.join(tad_dir, 'TFBSs.pickle'), 'rb') as tfbs_file: 
      X_df = pkl.load(tfbs_file)

    X_df['DNA_sequence'] = X_df['DNA_sequence'].str.lower()
    X_df['score'] = X_df['score'].astype(int) / 1000

    nucl = tf.one_hot(tf.convert_to_tensor(tokenizer.texts_to_sequences(X_df['DNA_sequence'])), depth=4)
    h3k4me3 = split_to_tensor(X_df, 'H3K4me3')
    h3k27ac = split_to_tensor(X_df, 'H3K27Ac')
    atacseq = split_to_tensor(X_df, 'ATAC-seq')

    bin_dir_list = os.listdir(tad_dir)
    for bin_name in bin_dir_list:
      bin_dir = os.path.join(tad_dir, bin_name)
      if not os.path.isdir(bin_dir):
        continue

      with open(os.path.join(bin_dir, 'dists.pickle'), 'rb') as tfbs_file: 
        hicdist_df = pkl.load(tfbs_file)
      
      hicdist = tf.reshape(tf.convert_to_tensor(hicdist_df['HiC'], dtype=tf.float32), (1, -1))

      y_df = pd.read_csv(os.path.join(bin_dir, 'target.tsv'), sep='\t')

      feature_cols = ['zscore', 'Data_in_HiC']
      feature_df = pd.concat([X_df['score'], hicdist_df[feature_cols]], axis=1)
      feature = tf.convert_to_tensor(feature_df, dtype=tf.float32)
      y = tf.reshape(tf.convert_to_tensor(y_df['CAGE_sum']), (1, 1))
      tad_idx = tf.reshape(tf.convert_to_tensor(tad_idx, dtype=tf.float32), (1, 1))
      
      yield {
        'nucl': nucl,    
        'h3k4me3': h3k4me3,           
        'h3k27ac': h3k27ac,     
        'atacseq': atacseq, 
        'hicdist': hicdist,
        'feature': feature,
        'tad_idx': tad_idx,
        'y': y
      }
