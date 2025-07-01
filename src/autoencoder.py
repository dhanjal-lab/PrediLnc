from src import *
import pickle
import numpy as np 
import pandas as pd
import numpy as np
import numpy.linalg as LA
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import csv
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf


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

d2=disease_feature[:,0:468]
d1=disease_feature[:,468:]

l1=lncRNA_feature[:,0:4458]
l2=lncRNA_feature[:,4458:4458*2]
l3=lncRNA_feature[:,4458*2:4458*3]

dd=np.hstack((d1,d2))
encoding_dim=512
input_img = Input(shape=(len(dd[0]),))
encoded_input = Input(shape=(encoding_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(936, activation='sigmoid')(encoded)
autoencoder = Model(inputs=input_img, outputs=decoded)
decoder_layer = autoencoder.layers[-1]
encoder = Model(inputs=input_img, outputs=encoded)
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(dd,dd,  epochs=100, shuffle=True)
encoded_dis=encoder.predict(dd)

encoder.save("encoder_dis.h5")


ll=lncRNA_feature
encoding_dim=512
input_img = Input(shape=(len(ll[0]),))
encoded_input = Input(shape=(encoding_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(4458*3, activation='sigmoid')(encoded)
autoencoder = Model(inputs=input_img, outputs=decoded)
decoder_layer = autoencoder.layers[-1]
encoder = Model(inputs=input_img, outputs=encoded)
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
# autoencoder.fit(x, x, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test, x_test))
autoencoder.fit(ll,ll,  epochs=100, shuffle=True)
encoded_lnc=encoder.predict(ll)

# Save the model
encoder.save("encoder_lnc.h5")

