import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import tensorflow as tf
import numpy as np
import scipy.spatial.distance as ds
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import json
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from numpy.linalg import norm                                                                                            
import torch                                                                               
from transformers import AutoTokenizer, AutoModel
import warnings  
warnings.filterwarnings(action='ignore')  

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

class EpochSaver(CallbackAny2Vec):
    def __init__(self):
        epoch = 0
        cur_time = datetime.datetime.now()

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(epoch),flush=True)
        cur_time = datetime.datetime.now()

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(epoch),flush=True)
        delta = datetime.datetime.now()-cur_time
        print("Time taken : ",delta,flush=True)
        epoch += 1

def cosine_similarity(data1, data2):
    return 1 - ds.cosine(data1, data2)

def normalize_l2(data_part):
    return normalize([data_part], norm="l2")[0]

def merge_vectors(list1,list2):
    temp = []
    for x in range(min(len(list1),len(list2))):
        temp.append(max(list1[x],list2[x]))
    return temp

def print_pretty(text):
    print("\n"*5)
    print("-"*100)
    print(text)


class CodeELBE_model:

    def load_CodeELBE_initial(self):
        model_name = "checkpoints/CodeELBE_initial/initial.bin"
        self.CodeELBE_initial = Word2Vec.load(model_name)

    def generate_embeddings_CodeELBE_initial_word(self,wrd):
        vec = [1e-6 for i in range(200)]
        if wrd in self.CodeELBE_initial.wv.vocab:
            vec = self.CodeELBE_initial[wrd]
        return vec

    def generate_embeddings_CodeELBE_initial_sent(self,sent):
        tokenized_context = sent.split()
        data_part = [1e-9 for i in range(200)]
        cnt = 0
        for wrd in tokenized_context:
            vec_r = self.generate_embeddings_CodeELBE_initial_word(wrd)
            cnt+=1
            for i in range(200):
                data_part[i]+=vec_r[i]
        if cnt>0:
            for i in range(200):
                data_part[i]/=cnt
        return data_part

    def load_CodeELBE_part1(self):
        datadir = "checkpoints/CodeELBE_part1"
        vocab_file = os.path.join(datadir, 'vocab.txt')
        options_file = os.path.join(datadir, 'options.json')
        weight_file = os.path.join(datadir, 'weights.hdf5')
        self.batcher = Batcher(vocab_file, 50)
        self.context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
        bilm = BidirectionalLanguageModel(options_file, weight_file)
        context_embeddings_op = bilm(self.context_character_ids)
        self.CodeELBE_part1_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def generate_embeddings_CodeELBE_part1_sent(self,sent):
        tokenized_context = [sent.split()]
        context_ids = self.batcher.batch_sentences(tokenized_context)
        CodeELBE_part1_context_input_ = self.sess.run(
            self.CodeELBE_part1_context_input['weighted_op'],
            feed_dict={self.context_character_ids: context_ids}
        )
        data_part = [1e-9 for i in range(CodeELBE_part1_context_input_.shape[2])]
        for x in range(CodeELBE_part1_context_input_.shape[1]):
            for y in range(CodeELBE_part1_context_input_.shape[2]):
                data_part[y] += CodeELBE_part1_context_input_[0][x][y]
        if CodeELBE_part1_context_input_.shape[1]>0:
            for y in range(CodeELBE_part1_context_input_.shape[2]):
                data_part[y]/=CodeELBE_part1_context_input_.shape[1]
        return data_part

    def generate_embeddings_CodeELBE_part_1_2_combined(self,sent):
        CodeELBE_initial = normalize_l2(self.generate_embeddings_CodeELBE_part1_sent(sent))
        CodeELBE_part1 = normalize_l2(self.generate_embeddings_CodeELBE_initial_sent(sent))
        return merge_vectors(CodeELBE_initial,CodeELBE_part1)

    def load_CodeELBE_part2(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("checkpoints/CodeELBE_part2", local_files_only=True)
        self.model.to(device)

    def generate_embeddings_CodeELBE_part2_word(self,x):
        tokenized_text = self.tokenizer.tokenize(x)[:512]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [0]*len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensor = torch.tensor([segment_ids]).to(device)
        with torch.no_grad():
            output = self.model(tokens_tensor)[0][0]
            t = torch.sum(output,dim=0).tolist()
        return t

    def generate_embeddings_CodeELBE_part2_sent(self,text1,text2):
        x = '<s>' + text1 + '</s>' + text2  
        tokenized_text = self.tokenizer.tokenize(x)[:512]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        try:
            segment_ids = [0]*(tokenized_text.index("</s>")+1) + [1]*(len(tokenized_text)-tokenized_text.index("</s>")-1) 
        except:
            return ([1e-9]*768,[1e-9]*768)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensor = torch.tensor([segment_ids]).to(device)
        if len(indexed_tokens)>0:
             with torch.no_grad():
                    output = self.model(tokens_tensor)[0][0]
                    t1 = torch.sum(output[1:tokenized_text.index("</s>")],dim=0).tolist()
                    t2 = torch.sum(output[tokenized_text.index("</s>")+1:],dim=0).tolist()
        if len(t1)==0 or len(t2)==0:
            t1 = [1e-9]*768
            t2 = [1e-9]*768
        return (t1,t2)

    def load_all_CodeELBE_models(self):
        self.load_CodeELBE_initial()
        self.load_CodeELBE_part1()
        self.load_CodeELBE_part2() 

    def generate_final_CodeELBE_embeddings_sent(self,text1,text2):    
        e1,e2 = self.generate_embeddings_CodeELBE_part2_sent(text1,text2)
        e3 = self.generate_embeddings_CodeELBE_part_1_2_combined(text1)
        e4 = self.generate_embeddings_CodeELBE_part_1_2_combined(text2)
        e1 = merge_vectors(normalize_l2(e1[:200]),e3)
        e2 = merge_vectors(normalize_l2(e2[:200]),e4)  
        return e1,e2

    def generate_final_CodeELBE_embeddings_word(self,text):    
        e1 = self.generate_embeddings_CodeELBE_part2_word(text)
        e2 = self.generate_embeddings_CodeELBE_part_1_2_combined(text)
        e1 = merge_vectors(normalize_l2(e1[:200]),e2) 
        return e1


embeddingmodel = CodeELBE_model()
embeddingmodel.load_all_CodeELBE_models()

# Example 1: Get CodeELBE embedding for a given word
v1 = embeddingmodel.generate_final_CodeELBE_embeddings_word("neighbour")
print_pretty("CodeELBE word embedding for the word 'neighbour' is")
print(v1)
print_pretty("Cosine similarity between CodeELBE word embeddings for the words 'neighbour' and 'search' is")
print(cosine_similarity(v1, embeddingmodel.generate_final_CodeELBE_embeddings_word("search")))


# Example 2: Get CodeELBE embedding for a given word in some context
e1,e2 = embeddingmodel.generate_final_CodeELBE_embeddings_sent("dirty read is a concurrency problem","dirty")
print_pretty("CodeELBE word embedding for the word 'dirty' in the context of 'dirty read is a concurrency problem' is")
print(e2)