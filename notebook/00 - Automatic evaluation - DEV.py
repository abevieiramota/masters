#!/usr/bin/env python
# coding: utf-8

# In[52]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[53]:


import os
import pandas as pd
import glob
import pickle

os.sys.path.insert(0, '../evaluation')

from evaluate import evaluate_all_systems, preprocess_all_models


# In[54]:


models = [os.path.basename(p) for p in glob.glob(f'../data/models/dev/*')]


# In[55]:


dfs = []
for model in models:
    if os.path.isfile(f'../data/models/dev/{model}/system_evaluation.csv'):
        df_ = pd.read_csv(f'../data/models/dev/{model}/system_evaluation.csv', index_col=['subset', 'references', 'metric'])
        dfs.append(df_)

scores_df = pd.concat(dfs, keys=models).unstack()
scores_df.columns = scores_df.columns.droplevel()


# In[56]:


params_dfs = []
for model in models:
    with open(f'../data/models/dev/{model}/params.pkl', 'rb') as f:
        params_dfs.append(pd.DataFrame([pickle.load(f)], index=[model]))
    
params_df = pd.concat(params_dfs)
params_df.tems_lm_n.fillna(0, inplace=True)
params_df.txs_lm_n.fillna(0, inplace=True)


# In[57]:


elapsed_time_dfs = []
for model in models:
    try:
        with open(f'../data/models/dev/{model}/elapsed_time.txt', 'r') as f:
            elapsed_time_dfs.append(pd.DataFrame(data=[float(f.readline()[:-1])], index=[model], columns=['elapsed_time']))
    except FileNotFoundError:
        pass
elapsed_time_df = pd.concat(elapsed_time_dfs)


# In[58]:


scores_all = scores_df.loc[(slice(None), 'all-cat', slice(None)), :].reset_index(level=[1, 2], drop=True)

df_all = pd.merge(scores_all, params_df, left_index=True, right_index=True)
df_all = pd.merge(df_all, elapsed_time_df, left_index=True, right_index=True)

cols = ['bleu', 'meteor', 'ter', 'elapsed_time', 'dp_scorer_n', 'sa_scorer_n', 'tems_lm_n', 'referrer_lm_n', 'txs_lm_n', 'max_dp', 'max_sa', 'max_refs', 'max_tems']
df_all = df_all[cols]
