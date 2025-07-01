import pickle
import os
os.makedirs("doSim", exist_ok=True)

import re
import gc
import requests
import pickle
import numpy as np 
import pandas as pd
import csv
import Levenshtein
import numpy as np
from scipy.spatial.distance import euclidean
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import conversion, default_converter
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import keras

import math

import torch
import time
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from flask_caching import Cache

from bs4 import BeautifulSoup
import rpy2.robjects as robjects
from rpy2.robjects import default_converter, conversion

with open("saved_models_latest/lncRNA_feature.pickle", 'rb') as file:
    lncRNA_feature=pickle.load(file)
l1=lncRNA_feature[:,0:4365]
l2=lncRNA_feature[:,4365:4365*2]
l3=lncRNA_feature[:,4365*2:]

with open("saved_models_latest/disease_feature.pickle", 'rb') as file:
    gdi=pickle.load(file)
    
d2=gdi[:,0:467]
d1=gdi[:,467:]
gdi=np.hstack((d1,d2))

with open("saved_models_latest/lda.pickle", 'rb') as file:
    lda=pickle.load(file)



with open("saved_models_latest/total_genes_names.pickle", 'rb') as file:
    lncTarget=pickle.load(file)

with open("saved_models_latest/new_sequence.pickle", 'rb') as file:
    sequences=pickle.load(file)
lncRNA_names = list(sequences.keys())   


with open("saved_models_latest/total_genes_names.pickle", 'rb') as file:
    targetNames=pickle.load(file)


with open("saved_models_latest/doid_dic.pickle", 'rb') as file:
    dis_doid_dic=pickle.load(file)


dis_doid_dic = {key.lower(): value for key, value in dis_doid_dic.items()}

diseases = list(dis_doid_dic.keys())
disease_names = diseases

with open("saved_models_latest/total_genes_names.pickle", 'rb') as file:
    disease_genes=pickle.load(file)
    
with open("saved_models_latest/adj.pickle", 'rb') as file:
    adj=pickle.load(file)

def extract_submatrices(adj, n_lnc, n_dis, n_gene):
    # Top-left block (ll)
    ll = adj[:n_lnc, :n_lnc]
    
    # Top-middle block (lda)
    lda = adj[:n_lnc, n_lnc:n_lnc + n_dis]
    
    # Top-right block (gl.T)
    gl_T = adj[:n_lnc, n_lnc + n_dis:]
    gl = gl_T.T  # Get original gl
    
    # Middle-left block (lda.T)
    lda_T = adj[n_lnc:n_lnc + n_dis, :n_lnc]
    assert np.allclose(lda_T, lda.T), "Mismatch in lda.T"
    
    # Middle-middle block (dd)
    dd = adj[n_lnc:n_lnc + n_dis, n_lnc:n_lnc + n_dis]
    
    # Middle-right block (gd.T)
    gd_T = adj[n_lnc:n_lnc + n_dis, n_lnc + n_dis:]
    gd = gd_T.T  # Get original gd
    
    # Bottom-left block (gl)
    gl_check = adj[n_lnc + n_dis:, :n_lnc]
    assert np.allclose(gl_check, gl), "Mismatch in gl"
    
    # Bottom-middle block (gd)
    gd_check = adj[n_lnc + n_dis:, n_lnc:n_lnc + n_dis]
    assert np.allclose(gd_check, gd), "Mismatch in gd"
    
    # Bottom-right block (gg)
    gg = adj[n_lnc + n_dis:, n_lnc + n_dis:]
    
    return lda, ll, dd, gl, gd, gg
lda_, ll_, dd_, gl_, gd_, gg_ = extract_submatrices(adj, 4365, 467, 13335)

with open("saved_models_latest/doids.pickle", 'rb') as file:
    doids=pickle.load(file)

with open("saved_models_latest/gip_dis.pickle", 'rb') as file:
    result_dis=pickle.load(file)

with open("saved_models_latest/gip_lnc.pickle", 'rb') as file:
    result_lnc=pickle.load(file)




import tensorflow as tf

from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import time
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        # Convolution operation
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.W_l = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.W_r = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.W_h = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.W_g = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.gamma = nn.Parameter(torch.FloatTensor([0]))  # trainable parameter γ
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W_h)
        nn.init.xavier_uniform_(self.W_g)

    def forward(self, H, adj):
        # Compute attention scores
        H_l = torch.matmul(H, self.W_l)
        H_r = torch.matmul(H, self.W_r)
        S = torch.matmul(H_l, torch.transpose(H_r, 0, 1))

        # Apply softmax to normalize attention scores along the last dimension
        beta = F.softmax(S, dim=-1)

        # Weighted sum of input elements based on attention weights
        B = torch.matmul(beta, H)

        # Calculate attention feature
        O = torch.matmul(B, self.W_h)
        O = torch.matmul(O, self.W_g)

        # Interpolation step
        output = torch.matmul(adj,H) + self.gamma * O

        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.relu1 = nn.ReLU()
        self.gc2 = GraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.relu2 = nn.ReLU()
        self.attention = AttentionLayer(nhid)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.gc2(x, adj)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.attention(x,adj)
        return x


def calculate_laplacian(adj):
    # Calculate the degree matrix
    degree = torch.sum(adj, dim=1)
    degree_matrix = torch.diag(degree)

    # Calculate the Laplacian matrix
    laplacian = degree_matrix - adj
    return laplacian

def adj_norm(adj):
    adj_hat = adj + torch.eye(adj.size(0), device=adj.device)

    # Compute degree matrix
    degree = torch.sum(adj_hat, dim=1)
    degree = torch.diag(degree)

    # Compute D^-0.5
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

    # Normalize adjacency matrix
    adj_normalized = torch.mm(torch.mm(degree_inv_sqrt, adj_hat), degree_inv_sqrt)

    return adj_normalized


def calculate_laplacian(adj):
    # Calculate the degree matrix
    degree = torch.sum(adj, dim=1)
    degree_matrix = torch.diag(degree)
    
    # Calculate the Laplacian matrix
    laplacian = degree_matrix - adj
    return laplacian

def adj_norm(adj):
    adj_hat = adj + torch.eye(adj.size(0), device=adj.device)

    # Compute degree matrix
    degree = torch.sum(adj_hat, dim=1)
    degree = torch.diag(degree)

    # Compute D^-0.5
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

    # Normalize adjacency matrix
    adj_normalized = torch.mm(torch.mm(degree_inv_sqrt, adj_hat), degree_inv_sqrt)
    
    return adj_normalized


with open("saved_models_latest/scaler.pkl",'rb') as file:
    scaler1=pickle.load(file)

with open("saved_models_latest/base_models.pkl", 'rb') as file:
    base_models=pickle.load(file)

with open("saved_models_latest/meta_model.pkl", 'rb') as file:
    meta_model1=pickle.load(file)



diseases=np.array(diseases)
lncRNA_names=np.array(lncRNA_names)
dis_genes=gdi[:,0:467]
ddsim=gdi[:,467:2*467]


from flask import Flask, render_template, request, redirect, jsonify,url_for, session
from Bio import SeqIO
from io import TextIOWrapper
from flask_session import Session
import uuid
import redis
import json
from flask import request, jsonify, session
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)


app = Flask(__name__)

app.secret_key = 'your_secret_key'

# Use Redis for session and cache
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis.StrictRedis(host='localhost', port=6379)
app.config['SESSION_SERIALIZER'] = json

app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'session:'

# Use Redis for caching
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_HOST'] = 'localhost'
app.config['CACHE_REDIS_PORT'] = 6379
# Initialize the Session extension
cache = Cache(app)
Session(app)

disease_list = diseases 
target_list = targetNames



from celery import Celery

celery = Celery(__name__)
celery.conf.broker_url = 'redis://localhost:6379/0'

# pandas2ri.activate()
@celery.task
def run_r_code(session_id):
    with conversion.localconverter(default_converter):

        # Install the DOSE package if not already installed
        robjects.r('''
            if (!requireNamespace("DOSE", quietly = TRUE)) {
                install.packages("DOSE", repos="http://cran.r-project.org")
            }
        ''')

        # Load the DOSE package
        robjects.r('library(DOSE)')

        # Read the CSV file into a data frame and process it
        robjects.r('''
            data <- read.csv("create_x1_forDisease.csv", header = FALSE)  
            disease_list <- as.list(data[[1]])
        ''')

        # Extract the target disease
        robjects.r('''
            target <- tail(disease_list, n = 1)
            disease_list <- disease_list[-length(disease_list)]
        ''')

        # Execute the doSim function and process the result
        robjects.r('''
            ddsim <- doSim(disease_list, target, measure = "Wang")
            ddsim[is.na(ddsim)] <- 0
        ''')

        # Write the result to a CSV file
        robjects.r(f'write.csv(ddsim, file = "doSim/ddsim_target_{session_id}.csv", row.names = FALSE)')

    print("R code execution completed and result saved to 'ddsim_target.csv'.")

def find_diseases_details(query):
    base_url = "https://disease-info-api.herokuapp.com/diseases"
    params = {"name": query}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data= response.json()
        return data['name'],data['description'],data['symptoms']
    return "Information not available","Information not available","Information not available"
def find_lncRNA_details(query):
    url = f"https://rest.ensembl.org/lookup/id/{query}?content-type=application/json"
    response = requests.get(url)
    if response.status_code == 200:
        lncrna_info = response.json()
        return lncrna_info["display_name"],lncrna_info['biotype'],lncrna_info['description']
    return "Information not available","Information not available","Information not available"
def find_info(sel_item,sel_list,f,scores, session_id, session_data):
    
    if 'selected_item' not in session_data:
        session_data['selected_item']=None
    if 'information_dic' not in session_data:
        session_data['information_dic']={}
    if 'selected_list' not in session_data:
        session_data['selected_list']=[]
    session_data['selected_item']=sel_item
    session_data['selected_list']=sel_list
    session_data['information_dic']={}
    API_KEY = 'c112bfffe2fe8a14645942743d2b4fc72008'
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    details=[]
    for x in range(len(sel_list)):
        details=[]
        query=sel_item+" AND "+sel_list[x]
        search_query = f'{base_url}esearch.fcgi?db=pubmed&api_key={API_KEY}&term={query}&retmode=json'
        response = requests.get(search_query)
        count = 0
        # Checking the status of the request
        if response.status_code == 200:
            data = response.json()
            # Extracting the PubMed IDs (PMIDs) of the articles
            pmids = data['esearchresult']['idlist']
            
            # Access individual articles using their PMIDs
            for pmid in pmids:
                if count==5:
                        break
                q={}
                article_query = f'{base_url}efetch.fcgi?db=pubmed&api_key={API_KEY}&id={pmid}&retmode=xml'
                article_response = requests.get(article_query)
                
                # Process the article data (in XML format) or perform other operations
                if article_response.status_code == 200:
                    count+=1
                    
                    article_data = article_response.text
                    soup = BeautifulSoup(article_data, 'xml')
                    
                    # Extracting title, link, and abstract
                    article_title = soup.find('ArticleTitle').text
                
                    article_link = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
              
                    q["title"]=article_title
                    q["link"]=article_link
                    # q["abstract"]=abstract_text
                    details.append(q)
                    print(f"Title: {article_title}")
                    print(f"Link: {article_link}")
                    
                else:
                    print(f"Error fetching article with PMID: {pmid}")

            session_data['information_dic'][sel_list[x]]=[round(scores[x],4),details]
            cache.set(f"session_output_{session_id}", {
                'selected_dis': session_data['selected_dis'],
                'information_dic': session_data['information_dic'],
                'selected_item': session_data['selected_item'],
                'selected_list': session_data['selected_list']
            })
        else:
            print("Failed to retrieve data")

        
    





@app.route('/run-r-code')
def execute_r_code():
    run_r_code.delay()  # Execute R code asynchronously
    return 'R code execution triggered!'


@app.route('/input_diseases', methods=['POST'])
def input_diseases():
    # global selected_diseases
    if 'selected_diseases' not in session:
        session['selected_diseases']=[]
    session['selected_diseases']=[]

    session['selected_diseases'] = request.form.getlist('diseases', type=str)
    
    if 'fileInput' in request.files:
        uploaded_file = request.files['fileInput']
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            file_diseases = [disease.strip() for disease in file_content.split('\n') if disease.strip()]
            
            # Extend the selected diseases with the file-based diseases
            session['selected_diseases'].extend(file_diseases)

    # Print or process the selected diseases
    print("Selected Diseases:", session['selected_diseases'])
    # Process the selected diseases further if needed
    # ...

    return render_template('topDiseasepred.html', selected_diseases=session['selected_diseases'])




@app.route("/input_lncRNA_forDiseease", methods=['POST'])
def input_lncRNA_forDiseease():
    if 'selected_lncRNAs_list' not in session:
        session['selected_lncRNAs_list']=[]

    session['selected_lncRNAs_list']=[]
    session['selected_lncRNAs_list'] = request.form.getlist('lncRNAs', type=str)
    
    if 'fileInput' in request.files:
        uploaded_file = request.files['fileInput']
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            file_lncRNAs = [lncRNA.strip() for lncRNA in file_content.split('\n') if lncRNA.strip()]
            
            # Extend the selected diseases with the file-based diseases
            session['selected_lncRNAs_list'].extend(file_lncRNAs)

    # Print or process the selected diseases
    print("Selected lncRNAs:", session['selected_lncRNAs_list'])
    # Process the selected diseases further if needed
    # ...

    return render_template('toplncRNApred.html', selected_lncRNAs_list=session['selected_lncRNAs_list'])











@app.route("/input_target",methods=['POST'])
def input_target():
    
    if 'selected_target' not in session:
        session['selected_target']=[]

    session['selected_target']=[]
    session['selected_target'] = request.form.getlist('targets',type=str)
    if 'fileInput' in request.files:
        uploaded_file = request.files['fileInput']
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            file_targets = [target.strip() for target in file_content.split('\n') if target.strip()]
            
            # Extend the selected diseases with the file-based diseases
            session['selected_target'].extend(file_targets)

    # Print or process the selected diseases
    print("Selected Targets:", session['selected_target'])
 
    return render_template('topDiseasepred.html', selected_target=session['selected_target'])

@app.route("/input_genes_disease",methods=['POST'])
def input_genes_disease():
    
    if 'selected_genes_dis' not in session:
        session['selected_genes_dis']=[]
    session['selected_genes_dis']=[]
    session['selected_genes_dis']=request.form.getlist('genes',type=str)
    if 'fileInput' in request.files:
        uploaded_file = request.files['fileInput']
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            file_targets = [target.strip() for target in file_content.split('\n') if target.strip()]
            
            # Extend the selected diseases with the file-based diseases
            session['selected_genes_dis'].extend(file_targets)

    # Print or process the selected diseases
    print("Selected Genes:", session['selected_genes_dis'])
 
    return render_template('toplncRNApred.html', selected_genes_dis=session['selected_genes_dis'])


@app.route("/input_sequence",methods=['POST'])
def input_sequence():
    if 'seq' not in session:
        session['seq']=""
    # print(request.files)
    if 'file' in request.files:
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Wrap the file object in TextIOWrapper to open it in text mode
            file_wrapper = TextIOWrapper(uploaded_file, encoding='utf-8')

            # Parse the uploaded file as a FASTA file
            sequences = list(SeqIO.parse(file_wrapper, "fasta"))

            # Process the sequences (e.g., print their IDs and sequences)
            for record in sequences:
                session['seq']+=str(record.seq)

    if 'sequence' in request.form:
        # Get sequence from manual input
        session['seq'] = request.form['sequence'] + session['seq']

    session['seq'] = re.sub(r"\s+", "", session['seq'])
    print("Sequence:", session['seq'])
    return render_template('topDiseasepred.html', seq=session['seq'])


@app.route("/input_lncRNA",methods=['POST'])
def input_lncRNA():
    # session.clear()
    if 'seq' not in session:
        session['seq']=""

    if 'selected_dis' not in session:
        session['selected_dis']=None
    
    if "selected_lncRNA" not in session:
        session['selected_lncRNA']=None
    if "selected_diseases" not in session:
        session['selected_diseases']=[]
    if 'selected_target' not in session:
        session['selected_target']=[]
    if 'selected_dis' not in session:
        session['selected_dis']=None
    session['selected_lncRNA']=None
    session['selected_diseases']=[]
    session['selected_target']=[]
    session['seq']=""
    session['selected_dis']=None
    session['selected_lncRNA'] = request.form.get('lncRNA')
    # Check if the value was selected from the dropdown or typed manually
    if session['selected_lncRNA'] == '':
        session['selected_lncRNA'] = request.form.get('custom-lncRNA')
    show_prediction_button = session['selected_lncRNA'] in lncRNA_names
    print(f"Selected or typed lncRNA: {session['selected_lncRNA']}")

    # Return JSON response instead of directly rendering the template
    return jsonify(selected_lncRNA=session['selected_lncRNA'], show_prediction_button=show_prediction_button)

def check_doid(dis):
    d=None
    flag=False
    base_url = "https://www.ebi.ac.uk/ols/api"
    search_url = f"{base_url}/search?q={dis}&ontology=doid"

    response = requests.get(search_url)
    data = response.json()

    if 'response' in data and 'docs' in data['response']:
        docs = data['response']['docs']
        try:
            if docs:
                doid = docs[0]['obo_id']
                term_label = docs[0]['label']
                d=doid
                flag=True
        
            
        except:
                print("Doid not found")

    else:
        print("Doid not found")
    
    return d,flag


@app.route("/input_disease_selected",methods=['POST'])
def input_disease_selected():
    # session.clear()
    if "selected_lncRNA" not in session:
        session['selected_lncRNA']=None

    if "selected_lncRNAs_list" not in session:
        session['selected_lncRNAs_list']=[]

    if "dis_doid" not in session:
        session['dis_doid']=[]

    if "selected_dis" not in session:
        session['selected_dis']=None

    if "selected_genes_dis" not in session:
        session['selected_genes_dis']=[]


    session['selected_dis']=None
    session['selected_lncRNAs_list']=[]
    session['selected_genes_dis']=[]
    session['dis_doid']=None
    session['selected_lncRNA']=None
    session['selected_dis']=request.form.get('disease')
    if session['selected_dis'] == '':
        session['selected_dis']=request.form.get('custom-disease')
    
    print(f"Selected disease: {session['selected_dis']}")
    show_button1=False
    show_button2=False
    if session['selected_dis'] in diseases:
        session['dis_doid']=dis_doid_dic[session['selected_dis']]
        show_button1=True
    else:
        d, flag = check_doid(session['selected_dis'])
        if flag:
            session['dis_doid'] = d
        else:
            session['dis_doid'] = None
            return jsonify({
                "error": f"DOID not found for disease: {session['selected_dis']}. Please enter a valid disease name.",
                "selected_dis": session['selected_dis'],
                "show_button1": False,
                "show_button2": False,
                "dis_doid": None
            }), 400


    print("Doid for this disease: ",session['dis_doid'])
    return jsonify(selected_dis=session['selected_dis'],show_button1=show_button1,show_button2=show_button2,dis_doid=session['dis_doid'])

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/applications', methods=['GET'])
def applications():
    return render_template('applications.html')


@app.route('/get_doid', methods=['GET'])
def get_doid():
    return jsonify({'doid': session['dis_doid']})


from threading import Thread
import time

@app.route('/progress')
def get_progress():
    session_id = session.get('user_id')
    progress = cache.get(f"progress_{session_id}") or {'percent': 0, 'message': 'Starting...'}
    return jsonify(progress)


@app.route('/submit', methods=['POST'])
def submit():
    session_id = session.get('user_id', str(time.time()))
    print(session_id)
    session['user_id'] = session_id
    cache.set(f"progress_{session_id}", {'percent': 0, 'message': 'Starting...', 'done': False})
    session_data = deepcopy(dict(session))
    Thread(target=printlist, args=(session_id, session_data)).start()
    return jsonify(success=True)


@app.route('/contribute', methods=['GET'])
def contribute_page():
    return render_template('contribute.html')

@app.route('/evidences', methods=['GET'])
def evidences():
    return render_template('evidences.html')


@app.route('/about-us', methods=['GET'])
def about_us():
    return render_template('about-us.html')


@app.route('/submit_contribution', methods=['POST'])
def submit_contribution():
    text_data = request.form['text_data']
    file_data = request.files['file_data']
    
    # Process or print the received data
    print("Text Data:", text_data)
    print("File Name:", file_data.filename)
    
    # Additional processing or saving the file can be done here
    
    return render_template('contribute.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    session_id = session.get('user_id')
    print(f"Session ID: {session_id}")
    data = cache.get(f"session_output_{session_id}")

    if not data:
        return "Error: No session data found", 400

    check_item = "lncRNA" if data['selected_dis'] is None else "disease"

    return render_template('result.html',
                           check_item=check_item,
                           information_dic=data['information_dic'],
                           selected_item=data['selected_item'],
                           selected_list=data['selected_list'])

def find_simi(similarity_stats):
    ans=None
    print(similarity_stats)
    if len(similarity_stats)==len(lncRNA_names):
        y=np.array(similarity_stats)
        idx=y.argsort()[-10:][::-1]
        lnc=lncRNA_names[idx]
        
        find_info(session['selected_lncRNA'],lnc,0, y[idx])
    else:
        y=np.array(similarity_stats)
        idx=y.argsort()[-10:][::-1]
        lnc=diseases[idx]
        find_info(session['selected_dis'],lnc,1, y[idx])

@app.route('/goToSimilarity', methods=['POST'])
def goToSimilarity():
    
    return redirect(url_for('similarity'))


@app.route('/similarity', methods=['GET','POST'])
def similarity():
    check_item=None
    if session['selected_dis']==None:
        check_item="lncRNA"

    else:
        check_item="disease"

    find_simi(session['similarity_stats'])
    return render_template('show_similarity.html',check_item=check_item,information_dic=session['information_dic'],selected_item=session['selected_item'],selected_list=session['selected_list'])


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/toplncRNApred',methods=['GET','POST'])
def toplncRNApred():
    print("I am the best")
    return render_template('toplncRNApred.html',diseases=diseases,lncRNA_names=lncRNA_names,disease_genes=disease_genes)

@app.route('/topDiseasepred', methods=['GET', 'POST'])
def topDiseasepred():
    
    return render_template('topDiseasepred.html', lncRNA_names=lncRNA_names, disease_list=diseases,target_list=target_list)



import numpy as np
from scipy.spatial.distance import cdist

def gKernel(nl, nd, inter_lncdis):
    # Compute Gaussian interaction profile kernel of lncRNAs
    sl = np.sum(np.square(inter_lncdis), axis=1)
    gamal = nl / np.sum(sl)

    # Efficient pairwise squared Euclidean distances for lncRNAs
    dist_lnc = cdist(inter_lncdis, inter_lncdis, 'sqeuclidean')
    pkl = np.exp(-gamal * dist_lnc)

    # Compute Gaussian interaction profile kernel of diseases
    sd = np.sum(np.square(inter_lncdis), axis=0)
    gamad = nd / np.sum(sd)

    # Efficient pairwise squared Euclidean distances for diseases
    dist_dis = cdist(inter_lncdis.T, inter_lncdis.T, 'sqeuclidean')
    pkd = np.exp(-gamad * dist_dis)

    return pkl, pkd



def create_similarity_matrix(association_matrix):
    # Calculate cosine similarity between lncRNAs
    similarity_matrix = cosine_similarity(association_matrix)
    return similarity_matrix

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

# def adj_matrix(lnc_dis_matrix, lnc_matrix, dis_matrix):
#     mat1 = np.hstack((lnc_matrix, lnc_dis_matrix))
#     mat2 = np.hstack((lnc_dis_matrix.T, dis_matrix))
#     return np.vstack((mat1, mat2))

def adj_matrix(lda, ll, dd, gl, gd, gg):
    mat1 = np.hstack((ll, lda, gl.T))
    mat2 = np.hstack((lda.T, dd, gd.T))
    mat3 = np.hstack((gl, gd, gg))
    return np.vstack((mat1, mat2, mat3))

def compute_known_features():
    global l1,l3,d1,d2,result_lnc,result_dis
    d2=gdi[:,0:467]
    d1=gdi[:,467:]
    print(gdi.shape)
    with tf.device('/CPU:0'):
        encoder_lnc = keras.models.load_model("saved_models_latest/encoder_lnc.h5")
        encoder_dis = keras.models.load_model("saved_models_latest/encoder_dis.h5")

    
        encoded_lnc=encoder_lnc.predict(lncRNA_feature)
        encoded_dis=encoder_dis.predict(gdi)
    del encoder_lnc, encoder_dis
    lnc1=topk(l1,result_lnc,10)
    # lnc2=topk(l3,result_lnc,10)

    print(d1.shape)
    dis1=topk(d1,result_dis,10)
    # dis2=topk(d2,result_dis,10)
    adj1=adj_matrix(lda,lnc1,dis1,gl_,gd_,gg_)
    # adj2=adj_matrix(lda,lnc2,dis2)
    features=np.vstack((encoded_lnc,encoded_dis,np.zeros((13335, 512))))

    features_tensor = torch.Tensor(features)
    adj1t = torch.Tensor(adj1)
    # adj2t = torch.Tensor(adj2)
    GCN_node1 = GCN(nfeat=512, nhid=512,dropout=0.4)
    # GCN_node2 = GCN(nfeat=512, nhid=256,dropout=0.4)

    GCN_node1 = torch.load('saved_models_latest/GCN_node1.pth')

    # GCN_node2 = torch.load('saved_models/GCN_node2.pth')

    GCN_node1.eval()
    # GCN_node2.eval()
    node_output1 = GCN_node1(features_tensor, adj1t).detach().numpy()
    # node_output2 = GCN_node2(features_tensor, adj2t).detach().numpy()
    del GCN_node1
    torch.cuda.empty_cache()
    gc.collect()
    return node_output1

   

def printlist(session_id, session_data):
    try:

        global d1,d2,l1,l2,l3
        if 'd2l' not in session_data:
            session_data['d2l']=[]

        if 'seq' not in session_data:
            session_data['seq']=""
        if 'l2d' not in session_data:
            session_data['l2d']=[]
        if 'similarity_stats' not in session_data:
            session_data['similarity_stats']=None
        if 'selected_genes_dis' not in session_data:
            session_data['selected_genes_dis']=[]
        if 'dis_doid' not in session_data:
            session_data['dis_doid']=None
        if 'selected_lncRNAs_list' not in session_data:
            session_data['selected_lncRNAs_list']=[]
        if 'seq' not in session_data:
            session_data['seq']=""
        # if 'selected_dis' not in session_data:
        #     session_data['selected_dis']=None

        if 'selected_diseases' not in session_data:
            session_data['selected_diseases']=[]

        if 'selected_target' not in session_data:
            session_data['selected_target']=[]

        if 'seq' not in session_data:
            session_data['seq']=""

        print(session_data['selected_lncRNA'])
        # print(l)
        print(session_data['selected_diseases'])
        print(session_data['selected_target'])
        print(session_data['selected_dis'])
        print(session_data['selected_genes_dis'])
        print(session_data['dis_doid'])
        print(session_data['selected_lncRNAs_list'])
        # session['seq']=l
        if session_data['selected_dis']==None:
            print("It is coming here..................................................................s")
            if session_data['selected_lncRNA'] in lncRNA_names and (len(session_data['seq'])<=1) and session_data['selected_diseases']==[] and session_data['selected_target']==[]:
                cache.set(f"progress_{session_id}", {'percent': 10,'message': "Collecting features for the selected lncRNA...", 'done': False})

                node_output1=compute_known_features()
                idx=np.where(lncRNA_names==session_data['selected_lncRNA'])
                x1=node_output1[idx]
                # x2=node_output2[idx]
                dd1=node_output1[4365:4365+467]
                # dd2=node_output2[4365:4365+467]
                xx1 = np.concatenate((dd1,x1.repeat(dd1.shape[0], axis=0)), axis=1)
                # xx2 = np.concatenate((dd2,x2.repeat(dd2.shape[0], axis=0)), axis=1)
                y1=[]
                xx1s=scaler1.transform(xx1)
                cache.set(f"progress_{session_id}", {'percent': 30,'message': "Structuring and aligning feature matrices...", 'done': False})

                for model in base_models:
                    y1.append(base_models[model].predict_proba(xx1s)[:, 1])
                y1=np.array(y1)
                print(y1)
                yy1=meta_model1.predict_proba(y1.T)[:, 1]
                cache.set(f"progress_{session_id}", {'percent': 60,'message': "Performing inference to rank top disease associations...", 'done': False})

                # y2=[]
                # xx2s=scaler2.transform(xx2)
                # for model in base_models2:
                #     y2.append(base_models2[model].predict_proba(xx2s)[:, 1])
                # y2=np.array(y2)


                # yy2=meta_model2.predict_proba(y2.T)[:, 1]

                y=(yy1)
                print(y)
                top_10_indices = y.argsort()[-10:][::-1]
                scores=y[top_10_indices]
                session_data['l2d']=(diseases[top_10_indices].tolist())
                print(session_data['selected_lncRNA'])
                cache.set(f"progress_{session_id}", {'percent': 80,'message': "Extracting supporting evidences...", 'done': False})

                find_info(session_data['selected_lncRNA'],session_data['l2d'],0,scores, session_id, session_data)

                print(session_data['l2d'])
                cache.set(f"progress_{session_id}", {'percent': 100,'message': "Done", 'done': False})

                return " "

            else:
                # cache.set(f"progress_{session_id}", {'percent': 0,'message': "step 0 done"})
                selected_sequence=session_data['seq']
                f1 = np.zeros((1, len(sequences)))

                # Create a dictionary to store the k-mer frequencies for each lncRNA ID
                
                kmer_frequencies = {}
                lncRNAseq=sequences
                k = 3
                possible_kmers = [''.join(p) for p in itertools.product('ATCGatcg', repeat=k)]
                
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
                # Store the k-mer frequencies for the current lncRNA ID
                
                sequence_kmers = {kmer: 0 for kmer in possible_kmers}

                # Iterate over the sequence with a sliding window of size k
                for i in range(len(selected_sequence) - k + 1):
                    kmer = selected_sequence[i:i+k]
                    sequence_kmers[kmer] += 1

                # kmer_frequencies[selected_lncRNA] = sequence_kmers
                for j, (lncRNA_id2, sequence_kmers2) in enumerate(kmer_frequencies.items()):
                    distance = euclidean(list(sequence_kmers.values()), list(sequence_kmers2.values()))
                    similarity = 1 / (1 + distance)
                    # print(j)
                    f1[0, j] = similarity
                    
                print("Collecting LncRNA Sequence based features...")
                cache.set(f"progress_{session_id}", {'percent': 10,'message': "Collecting features for the selected lncRNA...", 'done': False})

                f2 = np.zeros((1, len(sequences)))
                
                for j, (lnc, seq1) in enumerate(sequences.items()):
                    # Calculate the Levenshtein distance between the sequences
                    distance = Levenshtein.distance(selected_sequence, seq1)
                    similarity = 1 / (1 + distance)
                    # Store the distance in the feature matrix
                    f2[0, j] = similarity
                session_data['similarity_stats']=f2.tolist()[0]
                x1=np.hstack((f1,f2))
                
                print("Collecting disease based features...")

                cache.set(f"progress_{session_id}", {'percent': 40,'message': "Structuring and aligning feature matrices...", 'done': False})

                # xx1 = np.concatenate((x1.repeat(ddsim.shape[0], axis=0), ddsim), axis=1)



                target_input = session_data['selected_target']
                unique_targets = list(set(targetNames))
                genes_found = []
                idx = []

                for idx_val, target in enumerate(unique_targets):
                    if target in target_input:
                        genes_found.append(target)
                        idx.append(idx_val)


                tar_lnc = np.zeros(len(unique_targets))
                tar_lnc[idx]=1
                new_lncTarget=np.vstack((gl_.T,tar_lnc))
               
                new_lncTargetFeature=create_similarity_matrix(new_lncTarget)
                x2=new_lncTargetFeature[-1]
                f3=x2
                x2=x2[:-1]
                x2=x2.reshape(-1,1).T
                ll=np.hstack((x1,x2))
                print(x1.shape)
                print(x2.shape)
                print(lncRNA_feature.shape)
                print(ll.shape)
                ll=np.vstack((lncRNA_feature,ll))
                # xx2 = np.concatenate((x2.repeat(dis_genes.shape[0], axis=0),dis_genes), axis=1)

                print("Organizing feature matrices...")

                cache.set(f"progress_{session_id}", {'percent': 60,'message': "Performing inference to rank top disease associations...", 'done': False})


                disease_list = session_data['selected_diseases']
                disease_found = []
                disidx = []#

                for idx, disease in enumerate(diseases):
                    if disease in disease_list:
                        disease_found.append(disease)
                        disidx.append(idx)

                dis_lnc=np.zeros(len(diseases))
                dis_lnc[disidx]=1
                x3=dis_lnc
                x3=x3.reshape(-1,1).T
                # xx3 = np.concatenate((x3.repeat(lda.T.shape[0], axis=0),lda.T), axis=1)

                new_lda=np.vstack((lda,dis_lnc))
            
                gip_lnc,gip_dis=gKernel(len(lncRNA_names)+1, len(diseases),new_lda)
                x4=gip_lnc[-1]
                x4=x4[:-1]
                x4 = x4.reshape(-1, 1).T
                # xx4 = np.concatenate((x4.repeat(gip_dis.shape[0], axis=0), gip_dis), axis=1)
                print("step 4 done")
                print(ll)
                with tf.device('/CPU:0'):
                    encoder_lnc = keras.models.load_model("saved_models_latest/encoder_lnc.h5")
                    encoder_dis = keras.models.load_model("saved_models_latest/encoder_dis.h5")

                    encoded_lnc=encoder_lnc.predict(ll)           
                    encoded_dis=encoder_dis.predict(gdi)
                del encoder_lnc,encoder_dis
                f4=np.append(f1,1)
                l11=ll[:,0:4365]
                l11=np.hstack((l11,f4.reshape(-1,1)))
                l33=ll[:,4365*2:4365*3]
                l33=np.hstack((l33,f3.reshape(-1,1)))
                print(l11.shape)
                print(gip_lnc.shape)
                lnc1=topk(l11,gip_lnc,10)
                # lnc2=topk(l33,gip_lnc,10)
                d1=gdi[:,0:467]
                d2=gdi[:,467:]
                dis1=topk(d1,gip_dis,10)
                # dis2=topk(d2,gip_dis,10)
                adj1=adj_matrix(new_lda,lnc1,dis1,new_lncTarget.T,gd_,gg_)  
                # adj2=adj_matrix(new_lda,lnc2,dis2)
                features=np.vstack((encoded_lnc,encoded_dis, np.zeros((13335, 512))))
                print(features.shape)
                print(adj1.shape)
                features_tensor = torch.Tensor(features)
                adj1t = torch.Tensor(adj1)
                # adj2t = torch.Tensor(adj2)
                GCN_node1 = GCN(nfeat=512, nhid=512,dropout=0.4)
                # GCN_node2 = GCN(nfeat=512, nhid=256,dropout=0.4)

                GCN_node1 = torch.load('saved_models_latest/GCN_node1.pth')

                # GCN_node2 = torch.load('saved_models/GCN_node2.pth')

                GCN_node1.eval()
                # GCN_node2.eval()
                node_output1 = GCN_node1(features_tensor, adj1t).detach().numpy()
                # node_output2 = GCN_node2(features_tensor, adj2t).detach().numpy()

                del GCN_node1
                torch.cuda.empty_cache()
                gc.collect()
                # print(node_output1.shape)

                x1=node_output1[4365].reshape(-1,1).T
                # x2=node_output2[4365].reshape(-1,1).T
                dd1=node_output1[4365:4365+467]
                # dd2=node_output2[4365:4365+467]
                xx1 = np.concatenate((dd1,(x1.repeat(dd1.shape[0], axis=0))), axis=1)
                # xx2 = np.concatenate((dd2,(x2.repeat(dd2.shape[0], axis=0))), axis=1)
                y1=[]
                xx1s=scaler1.transform(xx1)
                for model in base_models:
                    y1.append(base_models[model].predict_proba(xx1s)[:, 1])
                y1=np.array(y1)
                print(y1)
                yy1=meta_model1.predict_proba(y1.T)[:, 1]
                # y2=[]
                # xx2s=scaler2.transform(xx2)
                # for model in base_models2:
                #     y2.append(base_models2[model].predict_proba(xx2s)[:, 1])
                # y2=np.array(y2)


                # yy2=meta_model2.predict_proba(y2.T)[:, 1]

                y=(yy1)
                print(y)
                top_10_indices = y.argsort()[-10:][::-1]
                scores=y[top_10_indices]
                session_data['l2d']=(diseases[top_10_indices].tolist())


                cache.set(f"progress_{session_id}", {'percent': 80,'message': "Extracting supporting evidences...", 'done': False })

                find_info(session_data['selected_lncRNA'],session_data['l2d'],0,scores, session_id, session_data)
                

                print(session_data['l2d'])
                cache.set(f"progress_{session_id}", {'percent': 100,'message': "Done", 'done': True})



        else:
            if session_data['selected_dis'] in diseases and session_data['selected_lncRNAs_list']==[] and session_data['selected_genes_dis']==[]:
                cache.set(f"progress_{session_id}", {'percent': 10,'message': "Collecting features for the selected disease...", 'done': False})

                node_output1=compute_known_features()
                idx=np.where(diseases==session_data['selected_dis'])
                x1=node_output1[idx[0]+4365].reshape(-1,1).T
                # x2=node_output2[idx[0]+4365].reshape(-1,1).T
                cache.set(f"progress_{session_id}", {'percent': 40,'message': "Structuring and aligning feature matrices...", 'done': False})

                dd1=node_output1[0:4365]
                # dd2=node_output2[0:4365]
                xx1 = np.concatenate((dd1,x1.repeat(dd1.shape[0], axis=0)), axis=1)
                # xx2 = np.concatenate((dd2,x2.repeat(dd2.shape[0], axis=0)), axis=1)
                y1=[]
                xx1s=scaler1.transform(xx1)
                for model in base_models:
                    y1.append(base_models[model].predict_proba(xx1s)[:, 1])
                y1=np.array(y1)
                print(y1)
                cache.set(f"progress_{session_id}", {'percent': 60,'message': "Performing inference to rank top lncRNA associations...", 'done': False})

                yy1=meta_model1.predict_proba(y1.T)[:, 1]

                y=(yy1)
                top_10_indices = y.argsort()[-10:][::-1]
                scores=y[top_10_indices]
                session_data['d2l']=(lncRNA_names[top_10_indices].tolist())
                cache.set(f"progress_{session_id}", {'percent': 80,'message': "Extracting supporting evidences...", 'done': False})

                find_info(session_data['selected_dis'],session_data['d2l'],1,scores, session_id, session_data)
                cache.set(f"progress_{session_id}", {'percent': 100,'message': "Done", 'done': True})

            else:
                
                cache.set(f"progress_{session_id}", {'percent': 10,'message': "Collecting features for the selected disease...", 'done': False})

                file_path = 'create_x1_forDisease.csv'
                d=np.array(doids)
                d=np.append(d,session_data['dis_doid'])
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    for row in d:
                        writer.writerow([row])
                
                run_r_code(session_id)

                ddsim_data=pd.read_csv(f"doSim/ddsim_target_{session_id}.csv")
                target_ddsim=ddsim_data.iloc[:, 0] 
                target_ddsim=list(target_ddsim)
                session_data['similarity_stats']=target_ddsim
                print("step 1 is done")
                cache.set(f"progress_{session_id}", {'percent': 30,'message': "Structuring and aligning feature matrices...", 'done': False})

                x1=np.array(target_ddsim) 
                dd1=np.vstack((d1,x1))
                a1=np.append(x1,1)
                a1=np.hstack((dd1,a1.reshape(-1,1)))
                x1=x1.reshape(-1,1).T

               

                target_input = session_data['selected_genes_dis']
                unique_targets = list(disease_genes)
                genes_found = []
                idx = []

                for idx_val, target in enumerate(unique_targets):
                    if target in target_input:
                        genes_found.append(target)
                        idx.append(idx_val)


                tar_lnc = np.zeros(len(unique_targets))
                tar_lnc[idx]=1
                new_lncTarget=np.vstack((gd_.T,tar_lnc))
                new_lncTargetFeature=create_similarity_matrix(new_lncTarget)
                x2=new_lncTargetFeature[-1]
                x2=x2[:-1]
                dd2=np.vstack((d2,x2))
                a2=np.append(x2,1)
                a2=np.hstack((dd2,a2.reshape(-1,1)))
                x2=x2.reshape(-1,1).T

                dd=np.hstack((x1,x2))
                dd=np.vstack((gdi,dd))
                # xx2 = np.concatenate((d2,x2.repeat(d2.shape[0], axis=0)), axis=1)
                disease_list = session_data['selected_lncRNAs_list']
                disease_found = []
                disidx = []

                for idx, disease in enumerate(lncRNA_names):
                    if disease in disease_list:
                        disease_found.append(disease)
                        disidx.append(idx)

                dis_lnc=np.zeros(len(lncRNA_names))
                dis_lnc[disidx]=1
                x3=dis_lnc
                x3=x3.reshape(-1,1).T
                # xx3 = np.concatenate((lda,x3.repeat(lda.shape[0], axis=0)), axis=1)
                new_lda=np.hstack((lda,x3.T))

                cache.set(f"progress_{session_id}", {'percent': 60,'message': "Performing inference to rank top lncRNA associations...", 'done': False})
            
                gip_lnc,gip_dis=gKernel(len(lncRNA_names), len(diseases)+1,new_lda)
                x4=gip_dis[-1]
                x4=x4[:-1]
                x4 = x4.reshape(-1, 1).T

                # xx4 = np.concatenate((gip_lnc,x4.repeat(gip_lnc.shape[0], axis=0)), axis=1)
                with tf.device('/CPU:0'):
                    encoder_lnc = keras.models.load_model("saved_models_latest/encoder_lnc.h5")
                    encoder_dis = keras.models.load_model("saved_models_latest/encoder_dis.h5")

                    encoded_lnc=encoder_lnc.predict(lncRNA_feature)           
                    encoded_dis=encoder_dis.predict(dd)
                del encoder_lnc,encoder_dis
                dd1=dd[:,0:467]
                dd2=dd[:,467:]

                lnc1=topk(l1,gip_lnc,10)
                # lnc2=topk(l3,gip_lnc,10)
                dis1=topk(a1,gip_dis,10)
                # dis2=topk(a2,gip_dis,10)
                adj1=adj_matrix(new_lda,lnc1,dis1,gl_,new_lncTarget.T,gg_)
                # adj2=adj_matrix(new_lda,lnc2,dis2)
                features=np.vstack((encoded_lnc,encoded_dis, np.zeros((13335, 512))))
                features_tensor = torch.Tensor(features)
                adj1t = torch.Tensor(adj1)
                # adj2t = torch.Tensor(adj2)
                GCN_node1 = GCN(nfeat=512, nhid=512,dropout=0.4)
                # GCN_node2 = GCN(nfeat=512, nhid=256,dropout=0.4)

                GCN_node1 = torch.load('saved_models_latest/GCN_node1.pth')

                # GCN_node2 = torch.load('saved_models_latest/GCN_node2.pth')

                GCN_node1.eval()
                # GCN_node2.eval()
                node_output1 = GCN_node1(features_tensor, adj1t).detach().numpy()
                # node_output2 = GCN_node2(features_tensor, adj2t).detach().numpy()

                del GCN_node1
                torch.cuda.empty_cache()
                gc.collect()
                x1=node_output1[4365+467].reshape(-1,1).T
                # x2=node_output2[4365+467].reshape(-1,1).T
                dd1=node_output1[0:4365]
                # dd2=node_output2[0:4365]
                xx1 = np.concatenate((dd1,(x1.repeat(dd1.shape[0], axis=0))), axis=1)
                # xx2 = np.concatenate((dd2,(x2.repeat(dd2.shape[0], axis=0))), axis=1)
                y1=[]
                xx1s=scaler1.transform(xx1)
                for model in base_models:
                    y1.append(base_models[model].predict_proba(xx1s)[:, 1])
                y1=np.array(y1)
                print(y1)
                yy1=meta_model1.predict_proba(y1.T)[:, 1]
               

                y=(yy1)
                print(y)
                top_10_indices = y.argsort()[-10:][::-1]
                scores=y[top_10_indices]
                session_data['l2d']=(lncRNA_names[top_10_indices].tolist())
                cache.set(f"progress_{session_id}", {'percent': 80,'message': "Extracting supporting evidences...", 'done': False})
                find_info(session_data['selected_dis'],session_data['l2d'],0,scores, session_id, session_data)
                print(session_data['l2d'])
                cache.set(f"progress_{session_id}", {'percent': 100,'message': "Done", 'done': True})

        return " "

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        print(traceback.format_exc())  # For console log
        cache.set(f"progress_{session_id}", {
            'percent': 100,
            'message': error_msg,
            'done': True,
            'error': True
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
