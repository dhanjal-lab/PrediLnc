import pickle
import numpy as np 
import pandas as pd

with open("lncRNA_feature.pickle", 'rb') as file:
    lncRNA_feature=pickle.load(file)

with open("gdi.pickle", 'rb') as file:
    gdi=pickle.load(file)


with open("lda.pickle", 'rb') as file:
    lda=pickle.load(file)

with open("updated_dis.pickle", 'rb') as file:
    diseases=pickle.load(file)

with open("lncRNA_names.pickle", 'rb') as file:
    lncRNA_names=pickle.load(file)

with open("useful_genes.pickle", 'rb') as file:
    useful_genes=pickle.load(file)

with open("lncTarget.pickle", 'rb') as file:
    lncTarget=pickle.load(file)
    
with open("sequences.pickle", 'rb') as file:
    sequences=pickle.load(file)



with open("targetNames.pickle", 'rb') as file:
    targetNames=pickle.load(file)
    
with open("disease_genes.pickle", 'rb') as file:
    disease_genes=pickle.load(file)

    

with open("gip_dis.pickle", 'rb') as file:
    result_dis=pickle.load(file)

with open("gip_lnc.pickle", 'rb') as file:
    result_lnc=pickle.load(file)

with open("disease_feature.pickle", 'rb') as file:
    disease_feature=pickle.load(file)

with open("disease_genes_association.pickle", 'rb') as file:
    gdi=pickle.load(file)
with open("genes_names_relatedToDisease.pickle", 'rb') as file:
    genes=pickle.load(file)

with open("doid_dic.pickle", 'rb') as file:
    doid_dic=pickle.load(file)
with open("gene_pathway_matrix.pickle", 'rb') as file:
    gene_pathway_matrix=pickle.load(file)

with open("genes_matrix.pickle", 'rb') as file:
    genes_matrix=pickle.load(file)
with open("genes_func_sim.pickle", 'rb') as file:
    genes_func_sim=pickle.load(file)

with open("genes_gaussian_sim.pickle", 'rb') as file:
    genes_gaussian_sim=pickle.load(file)
doids=list(doid_dic.values())
disease_names=list(doid_dic.keys())


import numpy as np
import numpy.linalg as LA
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import csv
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf


def topk(ma1,gip,nei):
    for i in range(ma1.shape[0]):
        ma1[i,i]=0
        gip[i,i]=0
    ma=np.zeros((ma1.shape[0],ma1.shape[1]))
    for i in range(ma1.shape[0]):
        if sum(ma1[i]>0)>nei:
            yd=np.argsort(ma1[i])
            ma[i,yd[-nei:]]=1
            ma[yd[-nei:],i]=1
        else:
            yd=np.argsort(gip[i])
            ma[i,yd[-nei:]]=1
            ma[yd[-nei:],i]=1
    return ma

def adj_matrix(lda, ll, dd, gl, gd, gg):
    mat1 = np.hstack((ll, lda, gl.T))
    mat2 = np.hstack((lda.T, dd, gd.T))
    mat3 = np.hstack((gl, gd, gg))
    return np.vstack((mat1, mat2, mat3))


