from src import *
def read_fasta_file(filename):
    sequences = {}
    with open(filename, 'r') as fasta_file:
        lines = fasta_file.readlines()
        current_id = None
        current_sequence = ''
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = current_sequence
                st=line.find('>') + 1
                end=line.find(':',st)
                current_id=line[st:end]
                current_sequence = ''
            else:
                current_sequence += line
        if current_id is not None:
            sequences[current_id] = current_sequence
    return sequences

import pickle

with open("lncRNA_names.pickle", 'rb') as file:
    lncRNA_names=pickle.load(file)

with open("disease_names.pickle", 'rb') as file:
    disease_names=pickle.load(file)

with open("lda.pickle", 'rb') as file:
    lda=pickle.load(file)

# Example usage
filename = 'lncRNA_new.out'  # Replace with your .fa file path
sequences = read_fasta_file(filename)

import numpy as np
from scipy.spatial.distance import euclidean
import itertools
k = 3
possible_kmers = [''.join(p) for p in itertools.product('ATCGatcg', repeat=k)]

# Create a dictionary to store the k-mer frequencies for each lncRNA ID
kmer_frequencies = {}
lncRNAseq=sequences
# Iterate over the lncRNA sequences
for lncRNA_id, sequence in lncRNAseq.items():
    # Initialize a dictionary to store the k-mer frequencies for the current sequence
    sequence_kmers = {kmer: 0 for kmer in possible_kmers}

    # Iterate over the sequence with a sliding window of size k
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        sequence_kmers[kmer] += 1

    # Store the k-mer frequencies for the current lncRNA ID
    kmer_frequencies[lncRNA_id] = sequence_kmers

similarity_matrix = np.zeros((len(lncRNAseq), len(lncRNAseq)))
for i, (lncRNA_id1, sequence_kmers1) in enumerate(kmer_frequencies.items()):
    for j, (lncRNA_id2, sequence_kmers2) in enumerate(kmer_frequencies.items()):
        distance = euclidean(list(sequence_kmers1.values()), list(sequence_kmers2.values()))
        similarity = 1 / (1 + distance)
        similarity_matrix[i, j] = similarity
print(similarity_matrix)


import numpy as np



import Levenshtein
lncRNAseq=sequences
feature_matrix = np.zeros((len(lncRNAseq), len(lncRNAseq)))

for i, (lncRNA_id1, sequence1) in enumerate(lncRNAseq.items()):
    for j, (lncRNA_id2, sequence2) in enumerate(lncRNAseq.items()):
        # Calculate the Levenshtein distance between the sequences
        distance = Levenshtein.distance(sequence1, sequence2)
        similarity = 1 / (1 + distance)
        # Store the distance in the feature matrix
        feature_matrix[i, j] = similarity
        # disfeature_matrix[i, j] = distance

# Print the feature matrix
print(feature_matrix)





import pandas as pd

lncRNAdic=sequences
# Use the retrieved dictionary
names=[]
# names=lncRNAdic.keys()
for i in lncRNAdic.keys():
  names.append(i)

# # Read the Excel file
data_frame = pd.read_excel('experimental lncRNA-disease information.xlsx')
dis = data_frame['Disease Name'].tolist()
lncname = data_frame['ncRNA Symbol'].tolist()
nid= data_frame['NONCODE'].tolist()
name=[]

print(len(lncname))
print(len(names))

name=list(set(names))
print(len(name))

data_frame = pd.read_excel('output.xlsx')
lncRNAs = data_frame['ncName'].tolist()
genes = data_frame['tarID'].tolist()
genes_name = data_frame['tarName'].tolist()
gene_type=data_frame['tarType']
useful_gene_type=[]
print(len(lncRNAs))
# # print(lncRNAs)
useful_lncRNAs=[]
useful_genes=[]
genes1=[]
for i in range(len(lncRNAs)):
    if lncRNAs[i] in name:
        useful_lncRNAs.append(lncRNAs[i])
        useful_genes.append(genes[i])
        useful_gene_type.append(gene_type[i])
        genes1.append(genes_name[i])


print(len(useful_genes))

lncRNA_unique = list(set(useful_lncRNAs))
genes_unique = list(set(useful_genes))
print(len(lncRNA_unique))
print(len(genes_unique))




# # Create an empty matrix
import numpy as np
matrix = np.zeros((len(name), len(genes_unique)), dtype=int)


for i in range(len(useful_lncRNAs)):
    if useful_genes[i]!='-':
      lnc_idx=name.index(useful_lncRNAs[i])
      gene_idx=genes_unique.index(useful_genes[i])
      matrix[lnc_idx,gene_idx]=1



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_similarity_matrix(association_matrix):
    # Calculate cosine similarity between lncRNAs
    similarity_matrix = cosine_similarity(association_matrix)


    return similarity_matrix

f3 = create_similarity_matrix(matrix)


print(len(list(set(lncRNA_names))))

import requests
import pickle


base_url = "https://www.ebi.ac.uk/ols/api"

# disease_names = ["colorectal cancer", "diabetes", "hypertension"]  # Replace with your list of disease names
not_found=[]
count=0
doids=[]
doid_dic={}
new_dis=[]
for disease_name in disease_names:
    search_url = f"{base_url}/search?q={disease_name}&ontology=doid"

    response = requests.get(search_url)
    data = response.json()
    count+=1
    if count%100==0:
      print(count)
    if 'response' in data and 'docs' in data['response']:
        docs = data['response']['docs']
        try:
            if docs:
                doid = docs[0]['obo_id']
                term_label = docs[0]['label']
                doids.append(doid)
                new_dis.append(disease_name)
                doid_dic[disease_name]=doid
                # print(f"Disease Name: {disease_name}")
                # print(f"DOID: {doid}")
                # print(f"Term Label: {term_label}")
            else:
                # print(f"No results found for: {disease_name}")
                not_found.append(disease_name)
        except:
                not_found.append(disease_name)

    else:
        print(f"Error retrieving data for: {disease_name}")


print(len(disease_names))
print(len(not_found))
print(doids)


import xml.etree.ElementTree as ET

# Your XML string (with potential leading characters)

def find_mesh_term(xml_string):
  xml_declaration_index = xml_string.find('<?xml')

  # Extract the substring starting from the XML declaration
  xml_cleaned = xml_string[xml_declaration_index:]

  # Parse the cleaned XML string
  root = ET.fromstring(xml_cleaned)

  # Find the TranslationSet section
  translation_set = root.find('TranslationSet')

  # Extract Mesh terms
  mesh_terms = [translation.find('To').text for translation in translation_set.findall('Translation')]

  # Print extracted Mesh terms
  mesh=""
  f=0
  # print(mesh_terms[0])
  for s in mesh_terms[0]:
    # print(s)
    if s=='"' and f==1:
      break
    if s=='"':
      f=1
    if s!='"':
      mesh+=s

  return mesh


import requests
count=0
new_dis_name=[]
new_doid_dic={}
for disease in disease_names:
    try:
      count+=1
      if count%100==0:
        print(count)
      url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term={disease}"

      response = requests.get(url)

      if response.status_code == 200:
          new_dis=find_mesh_term(response.text)
          new_dis_name.append(new_dis)
          new_doid_dic[new_dis]=doid_dic[disease]
      else:
          print("Failed to retrieve MeSH terms.")
    except:
        print(disease)
        new_dis_name.append(disease)




import pandas as pd

# Replace 'your_file.tsv' with the actual path to your TSV file
tsv_file_path = 'human_disease_textmining_filtered.tsv'

# Load the TSV file into a Pandas DataFrame
df = pd.read_csv(tsv_file_path, sep='\t')

# Now you can work with the 'df' DataFrame containing the data from the TSV file
diseases=list(df.iloc[:,3])
genes=list(df.iloc[:,1])
doid=list(df.iloc[:,2])
dis=disease_names
d=[]
count=0
c=0
gdi_new={}
for i in dis:
    x=i.split(" ")

    for j in range(len(diseases)):
        c=0
        for k in x:
            if k in diseases[j]:
                c+=1
        if c==len(diseases[j]) or c>=2:
            if i not in gdi_new:
                d.append(doid[j])
                gdi_new[i]=[genes[j]]
            else:
                gdi_new[i].append(genes[j])
    # if i in dis:
    #     count+=1
print(len(dis))
print(len(gdi_new))
# print(gdi_new)
print(len(d))
print(d)


import numpy as np

def gKernel(nl, nd, inter_lncdis):
    # Compute Gaussian interaction profile kernel of lncRNAs
    sl = np.zeros(nl)
    for i in range(nl):
        sl[i] = np.linalg.norm(inter_lncdis[i, :]) ** 2
    gamal = nl / np.sum(sl) * 1
    pkl = np.zeros((nl, nl))
    for i in range(nl):
        for j in range(nl):
            pkl[i, j] = float(np.exp(-gamal * (np.linalg.norm(inter_lncdis[i, :] - inter_lncdis[j, :])) ** 2))

    # Compute Gaussian interaction profile kernel of diseases
    sd = np.zeros(nd)
    for i in range(nd):
        sd[i] = np.linalg.norm(inter_lncdis[:, i]) ** 2
    gamad = nd / np.sum(sd) * 1
    pkd = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(nd):
            pkd[i, j] = float(np.exp(-gamad * (np.linalg.norm(inter_lncdis[:, i] - inter_lncdis[:, j])) ** 2))

    result_lnc = pkl
    result_dis = pkd

    return result_lnc, result_dis


gip_lnc,gip_dis=gKernel(len(lncRNA_names), len(disease_names), lda)
