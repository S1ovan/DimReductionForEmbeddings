import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import TruncatedSVD

# загрузим модель
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# загрузим датасет
df_dev = load_dataset("stsb_multi_mt", name="ru", split="dev")

# конвертнем датасет в датафрейм
res = []
F = True
for df in df_dev:
    score = float(df['similarity_score'])/5.0 # нормализация эталонной оценки
    embeddings = model.encode([df['sentence1'], df['sentence2']])
    semantic_sim = 1 - cosine(embeddings[0], embeddings[1]) # косинусное сходство между парами предложений
    res.append([df['sentence1'], df['sentence2'], score, semantic_sim])
    if F == True:
        mas_embed = embeddings
        F = False
    else:    
        mas_embed = np.concatenate((mas_embed, embeddings), axis=0)

df = pd.DataFrame(res, columns=['sentence1', 'sentence2', 'score', 'semantic_sim'])

# создадим список размерностей
dims = [x for x in range(50, 451, 50)]

# рассчитаем Евклидово расстояние для базовой модели
df['eucl_dis'] = np.square(df['score'] - df['semantic_sim'])
tmp_targ = np.sqrt(df['eucl_dis'].sum())
targ = [tmp_targ for _ in range(len(dims))]

# для каждого метода уменьшения размерности
# найдем эмбеддинги новых размерностей
# и для каждой пары предложений косинусное сходство
# для каждой размерности найдем евклидово расстояние до эталонной оценки 

# ICA
eucl_dis_ica = []
for el in dims:
    ica =  FastICA(n_components = el)
    mas_embed_fit = ica.fit_transform(mas_embed)
    
    # семантическое сходство
    tmp_res = []
    for i in range (0, 3000, 2):
        semantic_sim = 1 - cosine(mas_embed_fit[i], mas_embed_fit[i+1])
        tmp_res.append(semantic_sim)
    
    # евклидово расстояние 
    df[f'reduce_sim_ica_{el}'] = tmp_res
    df['eucl_dis_ica'] = np.square(df['score'] - df[f'reduce_sim_ica_{el}'])
    eucl_dis_ica.append(np.sqrt(df['eucl_dis_ica'].sum()))

# PCA
eucl_dis_pca = []
for el in dims:
    pca =  PCA(n_components = el)
    mas_embed_fit = pca.fit_transform(mas_embed)
    
    # семантическое сходство
    tmp_res = []
    for i in range (0, 3000, 2):
        semantic_sim = 1 - cosine(mas_embed_fit[i], mas_embed_fit[i+1])
        tmp_res.append(semantic_sim)
    
    # евклидово расстояние
    df[f'reduce_sim_pca_{el}'] = tmp_res
    df['eucl_dis_pca'] = np.square(df['score'] - df[f'reduce_sim_pca_{el}'])
    eucl_dis_pca.append(np.sqrt(df['eucl_dis_pca'].sum()))

# FA
eucl_dis_fa = []
for el in dims:
    fa =  FactorAnalysis(n_components = el)
    mas_embed_fit = fa.fit_transform(mas_embed)
    
    # семантическое сходство
    tmp_res = []
    for i in range (0, 3000, 2):
        semantic_sim = 1 - cosine(mas_embed_fit[i], mas_embed_fit[i+1])
        tmp_res.append(semantic_sim)
    
    # евклидово расстояние
    df[f'reduce_sim_fa_{el}'] = tmp_res
    df['eucl_dis_fa'] = np.square(df['score'] - df[f'reduce_sim_fa_{el}'])
    eucl_dis_fa.append(np.sqrt(df['eucl_dis_fa'].sum()))

# TSVD
eucl_dis_tsvd = []
for el in dims:
    tsvd =  TruncatedSVD(n_components = el)
    mas_embed_fit = tsvd.fit_transform(mas_embed)
    
    # семантическое сходство
    tmp_res = []
    for i in range (0, 3000, 2):
        semantic_sim = 1 - cosine(mas_embed_fit[i], mas_embed_fit[i+1])
        tmp_res.append(semantic_sim)
    
    # евклидово расстояние
    df[f'eucl_dis_tsvd_{el}'] = tmp_res
    df['eucl_dis_tsvd'] = np.square(df['score'] - df[f'eucl_dis_tsvd_{el}'])
    eucl_dis_tsvd.append(np.sqrt(df['eucl_dis_tsvd'].sum()))

# нарисуем график
plt.figure(figsize=(12,7.5), dpi= 80)

plt.plot(dims, eucl_dis_pca, color='tab:red', label='PCA')
plt.text(dims[-1], eucl_dis_pca[-1], 'PCA', fontsize=12, color='tab:red')
plt.plot(dims, eucl_dis_ica, color='tab:blue', label='ICA')
plt.text(dims[-1], eucl_dis_ica[-1], 'ICA', fontsize=12, color='tab:blue')
plt.plot(dims, eucl_dis_fa, color='tab:green', label='FA')
plt.text(dims[-1], eucl_dis_fa[-1], 'FA', fontsize=12, color='tab:green')
plt.plot(dims, eucl_dis_tsvd, color='tab:green', label='TSVD')
plt.text(dims[-1], eucl_dis_tsvd[-1], 'TSVD', fontsize=12, color='tab:green')
plt.plot(dims, targ, color='tab:orange', label='Target', linestyle='dashed')

plt.ylabel('Евклидово расстояние')
plt.xlabel('Размерность')

plt.legend(loc='upper right', ncol=2, fontsize=12)

plt.show()
