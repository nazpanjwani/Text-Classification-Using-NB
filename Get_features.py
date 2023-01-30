from collections import Counter
from pprint import pprint
from nltk.tokenize import word_tokenize
import re
import numpy
import os
import ast
from urllib.request import urlopen
from bs4 import BeautifulSoup
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from regex import D
import spacy
import nltk
lemma=WordNetLemmatizer()
from nltk.tag import pos_tag
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer

# DIR= os.listdir('./course-cotrain-data/fulltext/course')
# count_f=len(DIR)
# DIR= os.listdir('./course-cotrain-data/fulltext/non-course')
# count_f=count_f+len(DIR)
dt={}
dt['Data']=[]
dt['Type']=[]
V=[] 
D_all=[]
index={} 
index_dup={}

def preproc(C):
    global V
    sourcepath = os.listdir('course-cotrain-data/fulltext/'+C+'/')
    for file in sourcepath:
        f = 'course-cotrain-data/fulltext/'+C+'/' + file
        file=open(f,'r')
        html = file.read()
        soup = BeautifulSoup(html, features="html.parser")

        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        text=re.sub(r'[0–9]+', '', text )
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
    
        stl = open("Stopword-List.txt", 'r')
        for line in stl:
            for st in line.split():
                text=text.lower().replace(' '+st+' ',' ').replace('\t',' ').replace('\n',' ').replace('®',' ').replace('¦',' ').replace('§',' ').replace('ì',' ').replace('¨',' ').replace('.',' ').replace('. '," ").replace(', ',' ').replace('-',' ').replace(')',' ').replace('(',' ').replace('%',' ').replace('@',' ').replace(' +',' ').replace(':',' ').replace('/',' ').replace('_',' ').replace('--',' ').replace('[',' ').replace(']',' ').replace(' #',' ').replace('&',' ').replace('"',' ').replace('\uf0b7',' ').replace('–',' ').replace('“',' ').replace('”',' ').replace('•',' ').replace('\ufeff',' ').replace('’',' ').replace('|',' ').replace('?',' ').replace('!',' ').replace('=',' ').replace('0',' ').replace('1',' ').replace('2',' ').replace('3',' ').replace('4',' ').replace('5',' ').replace('6',' ').replace('7',' ').replace('8',' ').replace('9',' ').replace('~',' ').replace('»',' ').replace('«',' ').replace('¹',' ').replace('}',' ').replace('²',' ').replace('©',' ').replace('{',' ').replace('³',' ').replace('¤',' ').replace('é',' ').replace('#',' ')
        
        doc=word_tokenize(text)
        lemmatization=[]
        doc1=[]
        for w in doc:
            if len(w)>1:
                doc1.append(w)
        doc=doc1
        doc=[s for s in doc if s]
        for j in doc:
            lemmatization.append(lemma.lemmatize(j))

        index_dup[f]=lemmatization

        temp=[]
        for w in lemmatization: 
            D_all.append(w)
            if w not in temp:
                temp.append(w)
                V.append(w)

        index[f]=temp
    
    temp1=[]
    for w in V:
        if w not in temp1:
            temp1.append(w)
    V=temp1
 
    for key in index_dup:
        data=''
        for i in index_dup[key]:
            data=data+i+' '
        dt['Data'].append(data)
        dt['Type'].append(C)
    if C=='non-course':
        return V, D_all, dt, index, index_dup

######### I did created manual tf_idf function but since the input in classifier needs differnt DS, I implemented predefine tf-idf vectorizer 
# def tf_idf(Vec, index, index_dup):
#     wgt={}
#     df={}
#     idf={}
#     tf={}
#     data=''
#     for w in Vec:
#         df.setdefault(w,0) 
#         for key in index:
#             if(w in index[key]):
#                 df[w]+=1

#     #calculating idf
#     for key in df:
#         idf.setdefault(key,0)
#         # if(df[key]<=0):
#             # print(key)
#             # print(df[key])
#         idf[key]=numpy.log2(count_f/df[key])

#     #calculating tf
#     for key in index_dup:
#         tf.setdefault(key,{})
#         for w in Vec:
#             tf[key][w]=index_dup[key].count(w)

#     #calculating tf*idf for each document
#     for key in tf:
#         wgt.setdefault(key,{})
#         for w in Vec:
#             wgt[key][w]=float(format(tf[key][w]*idf[w],'.3F'))

#     dt=[]
#     for key in wgt:
#         dt.append(wgt[key])
#     wgt_all={}
#     for key in wgt:
#         for k in wgt[key]:
#             wgt_all.setdefault(k,wgt[key][k])

#     for key in wgt_all:
#         sorted_wgt=sorted(wgt_all.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

#     top=0
#     features_wgt=[]
#     for key in sorted_wgt:
#         features_wgt.append(key[0])
#         top=top+1
#         if(top==100):
#             break
#     # print(features_wgt)

#     f = open("top_tf_idf.txt", "w")
#     f.write(str(features_wgt))
#     f.close()
#     return dt

def find_noun(V, D_all):
    nlp = spacy.load("en_core_web_sm")
    corpus_V=''
    for w in V:
        corpus_V=corpus_V+' '+w

    noun=[]
    doc = nlp(corpus_V)
    for w in doc:
        if(w.tag_ == 'NNP'):
            noun.append(w)

    noun_count={}
    for w in noun:
        noun_count[w]=D_all.count(str(w))

    noun_count=Counter(noun_count).most_common(50)

    top_noun=[]
    for w in noun_count:
        top_noun.append(str(w[0]))

    f = open("top_noun.txt", "w")
    f.write(str(top_noun))
    f.close()

def prune(lexical):
    final_chain = []
    while lexical:
        result = lexical.pop()
        if len(result.keys()) == 1:
            for value in result.values():
                if value != 1: 
                    final_chain.append(result)
        else:
            final_chain.append(result)
    return final_chain

def create_lexical_chain(nouns, relation_list):
    lexical = []
    threshold = 0.5
    for noun in nouns:
        flag = 0
        for j in range(len(lexical)):
            if flag == 0:
                for key in list(lexical[j]):
                    if key == noun and flag == 0:
                        lexical[j][noun] +=1
                        flag = 1
                    elif key in relation_list[noun][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if syns1 and syns2:
                            if syns1[0].wup_similarity(syns2[0]) >= threshold:
                                lexical[j][noun] = 1
                                flag = 1
        if flag == 0: 
            dic_nuevo = {}
            dic_nuevo[noun] = 1
            lexical.append(dic_nuevo)
            flag = 1
    return lexical

def relation_list(nouns):

    relation_list = defaultdict(list)
    
    for k in range (len(nouns)):   
        relation = []
        for syn in wordnet.synsets(nouns[k], pos = wordnet.NOUN):
            for l in syn.lemmas():
                relation.append(l.name())
                if l.antonyms():
                    relation.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    relation.append(l.hyponyms()[0].name().split('.')[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    relation.append(l.hypernyms()[0].name().split('.')[0])
        relation_list[nouns[k]].append(relation)
    return relation_list

def find_Lexical_feature(D_all):
    corpus_D=''
    for w in D_all:
        corpus_D=corpus_D+' '+w

    position = ['NN', 'NNS', 'NNP', 'NNPS']
    
    sentence = nltk.sent_tokenize(corpus_D)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [tokenizer.tokenize(w) for w in sentence]
    tagged =[pos_tag(tok) for tok in tokens]
    nouns = [word.lower() for i in range(len(tagged)) for word, pos in tagged[i] if pos in position ]
        
    relation = relation_list(nouns)
    lexical = create_lexical_chain(nouns, relation)
    final_chain = prune(lexical)

    LF=[]
    chain={}
    for w in final_chain:
        for key in w:
            LF.append(key)

    f = open("top_LF.txt", "w")
    f.write(str(LF))
    f.close()

C='course'
preproc(C)
C='non-course'
Vec, D_all, dt, index, index_dup=preproc(C)

f = open("DataFrame.txt", "w")
f.write(str(dt))
f.close()

# tf_idf(Vec, index, index_dup)
find_noun(Vec, D_all)
find_Lexical_feature(D_all)
