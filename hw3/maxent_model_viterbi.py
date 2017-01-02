# -*- coding: utf-8 -*-
"""
Created on Mon Nov 02 13:37:43 2015

@author: Dongyoung
"""


import numpy as np
from sklearn.linear_model import LogisticRegression
import string
import sys

"""
Set up features
"""
vowels = ['a', 'e', 'u', 'i', 'o']

def f0(x, w, z):
    return int('zeta' in x.lower())
# a single capital letter that follows a gene
def f00(x,w,z):
    return int( len(x)==1 and x.isupper() and z<0 )    
# 'enhancer' that follows a gene
def f01(x,w,z):
    return int( 'enhancer' in x.lower() and z<0 )    
# word that follows 'the'
#def f02(x,w,z):
#    return int( x.lower()=='the' )
## 'receptor' that follows gene
def f03(x,w,z):
    return int( 'receptor' in x.lower() and z<0)
# a single capital letter that follows 'protein'
def f04(x,w,z):
    return int( len(x)==1 and x.isupper() and 'protein' in w.lower() )  
# more then 12 alphabet letters
def f05(x,w,z):
    return int( len(x) >= 10 )
# tumor necrosis
def f06(x,w,z):
    return int( 'necrosis' in x.lower() and 'tumo' in w.lower() )
# necrosis factor
def f07(x,w,z):
    return int( 'factor' in x.lower() and 'necrosis' in w.lower() )
#ends with 'eron'
def f08(x,w,z):
    return int( len(x) > 4 and x[-4:].lower() == 'eron' )
# Roman numbers
def f09(x,w,z):
    return int( x == 'II' )
#rbcX is a gene
def f1(x, w, z):
    return int((len(x) == 4 and x[0:3] == 'rbc'))   
def f10(x,w,z):
    return int( x == 'III' )
def f11(x,w,z):
    return int( x == 'IV' )
def f12(x,w,z):
    return int( x == 'V' )
def f13(x,w,z):
    return int( x == 'VI' )
def f14(x,w,z):
    return int( x == 'VII' )
def f15(x,w,z):
    return int( x == 'VIII' )    

#polymerase that follows uppercase letters
def f16(x,w,z):
    return  int( 'polymerase' in x.lower() and fj(w, 'dummy', 'dummy') > 0  )

#'collagen' in word
def f17(x,w,z):
    return int( 'collagen' in x.lower()  )


def f19(x,w,z):
    return int( "immunoglobulin" in x.lower() )

def f20(x, w, z):
    return int( "immunoglobulin" in w.lower() )
    
def f21(x,w,z):
    return int(len(x) > 3 and x[-3:].lower=='tin')

def f22(x,w,z):
    return int(len(x) > 3 and x[-3:].lower=='nin')

def f23(x,w,z):
    return int(len(x) > 3 and x[-3:].lower=='ase')

def f24(x,w,z):
    return int( x == r'/' and z<0 )
def f25(x,w,z):
    return int( w == r'/' and z<0 )

def f26(x,w,z):
    return int( 'sf' in x.lower() and len(x) <= 5 )

# a single capital letter that does not follow start
def f27(x,w,z):
    return int( len(x)==1 and x.isupper() and w != 'start')
    
def f28(x,w,z):
    return int( 'element' in x.lower() and z <0 )
def f29(x,w,z):
    return int( x=='transcript' or x=='transcripts' and z<0  )
    

#rbc is not a gene    
def f2(x, w, z):
    return int(x == 'rbc')
    
# if O2 is a substring, it's not a gene
def f3(x, w, z):
    return int('O2' in x)

def f30(x,w,z):
    return int(x=='chain' and z<0)

def f31(x,w,z):
    return int(len(x) >= 3 and 'jun' in x[-3:].lower())

# capital letter more than length 3
def f32(x,w,z):
    return int(len(x) > 3 and x.isupper())

# is not at start of the sentence, has two lowercase, and at least one uppercase
def f33(x,w,z):
    return int(  w!='start' and sum([1 for c in x if c.islower()]) >= 2 and sum([1 for c in x if c.isupper()]) >= 1    )

# contains at least two numbers and atleast one alphabet
def f34(x,w,z):
    return int( sum([1 for c in x if c.isdigit()]) >= 2 and sum([1 for c in x if c.isalpha()]) >= 1 )
    
#all lowercase and then capital at end
def f35(x,w,z):
    return int( len(x) >= 2 and x[:-1].islower() and x[-1].isupper() )

    
def f36(x,w,z):
    return int('globin' in x.lower())

def f37(x,w,z):
    return int('endorphin' in x.lower())
    
def f38(x,w,z):
    return int( 'mu' == x.lower() )

def f39(x,w,z):
    return int( 'X' in x or 'Z' in x)


# if CO is a substring, it's not a gene
def f4(x, w, z):
    return int('CO' in x)
    
# greek letters are likely to be genes
def f5(x, w, z):
    return int('alpha' in x.lower())
    
def f6(x, w, z):
    return int('beta' in x.lower())
    
def f7(x, w, z):
    return int('gamma' in x.lower())
    
def f8(x, w, z):
    return int('kappa' in x.lower())
    
def f9(x, w, z):
    return int('epsilon' in x.lower())
    
def fa(x, w, z):
    return int('sigma' in x.lower())

def fa1(x, w, z):
    return int('rho' in x.lower())

# synthetase is a gene
def fb(x, w, z):
    return int('synthetase' in x.lower())

# 'gene' following a gene is a gene
def fc(x, w, z):
    return int('gene' == x or 'genes' == x and z <0)
    

    
# if it ends with 'rin', it's likely to be a gene
def fd(x, w, z):
    return int(  len(x) >= 3 and x[-3:] == 'rin'  )

# 'kinase' is a gene
def fe(x, w, z):
    return int('kinase' in x.lower())
    
# 'mutant' is a gene
def ff(x, w, z):
    return int('mutant' in x.lower())

# if lowercase vowel is not present, it is not an english word
def fg(x, w, z):
    bools = [(v in x) for v in vowels]
    return int( not any(bools)  )


# '-' following gene is likely a gene    
def fh1(x,w,z):
    return int( x=='-' and z<0 )
# word following a '-' gene is likely a gene
def fh2(x,w,z):
    return int( w=='-' and z <0 )
# '-' following a single character is likely a gene   
def fh3(x,w,z):
    return int( x=='-' and len(w)==1 )


    
# 'RNA" is likely in a gene
def fi(x, w, z):
    return int( 'RNA' in x )

# more than one uppercase letter
def fj(x, w, z):
    #remove_lower = x.translate(None, string.ascii_lowercase)
    #return int( len(remove_lower) >= 2 )
    return int(  sum([1 for c in x if c.isupper()]) > 2  and ('DNA' not in x ) and ('RNA' not in x) and ('PCR' not in x) )

# containts 'protein'
def fk(x, w, z):
    return int( 'protein' in x )
    
# combination of alphabet and numbers
def fl(x, w, z):
    remove_letter = x.translate(None, string.ascii_letters)
    return int( len(remove_letter) >= 1 and len(x) > len(remove_letter) )
   
# punctuations
def fm(x, w, z):
    return int( x==',' ) 
    
def fn(x, w, z):
    return int( x=='.' )

#'ase' is likely to be a gene
def fo(x, w, z):
    return int(  len(x) >= 3 and x[-3:] == 'ase'  )
    
 # punctuations or english words following a gene
def fp(x,w,z):
    return int( z < 0 and x == 'and'  )
def fq(x,w,z):
    return int( z < 0 and (x == ',' or x=='.') )
def fr(x,w,z):
    return int( z < 0 and ( x=='is' or x=='are' or x=='was' or x=='were' ) )
    
# 'factor' is likely to be gene
def fs(x,w,z):
    return int( 'factor' in x.lower() )
# 'factor' following a gene is more likely to be gene
def ft(x,w,z):
    return int( z<0 and 'factor' in x.lower() )
    
# a word that follows a gene
#def fu(x,w,z):
#    return int( z < 0 )
    
# parenthesis that follows a gene
def fv(x,w,z):
    return int( z<0 and x==')' )
# 'protein' that follows a gene
def fw(x,w,z):
    return int( 'protein' in x.lower() and z<0 )

# period used not as EOF
def fx(x,w,z):
    return int( '.' in x and len(x) > 1 )

# '+' at the end of word
def fy(x,w,z):
    return int( len(x) > 1 and x[-1] == '+'  )

# 'site' that follows a gene
def fz(x,w,z):
    return int( 'site' in x.lower() and z<0 )
    


    
feature_list = np.array( [f for fname, f in sorted(globals().items()) if callable(f) and fname[0] == 'f' and (len(fname) == 2 or len(fname)==3)] )

#%%
"""
read raw data
"""
input_file = open('./data/train.tag', 'rb')
raw = []

count = 0
for line in input_file:
    count += 1
    if count%2 == 1 :
        continue
    raw = raw + line.split()

raw = [ r.rsplit('_', 1) for r in raw ]
raw = np.array(raw)
words = raw[:,0]
tags = raw[:,1]



#%%
"""
Process data
"""

X = np.zeros(shape=( len(words), len(feature_list)) , dtype=float)
Y = np.zeros(shape=(len(words), 1) , dtype=float)
Y[:,0] = (tags=='TAG')
Y[Y < 0.5] = -1.0
last_words = np.delete( np.insert(words,0, 'start'), len(words))
last_tags = np.delete( np.insert(Y, 0, 1.0), len(Y))

for i in xrange(len(feature_list)):
    fn = feature_list[i]
    vfn = np.vectorize(fn)
    X[:,i] = vfn(words, last_words, last_tags)
    
#%%
"""
Fit regression
"""

logreg = LogisticRegression(C=1e5)
logreg.fit(X,Y)

def toFeature(x, w, z):
    return np.array([ fn(x, w, z) for fn in feature_list ])
    
#logreg.predict( toFeature('WDX') )


#%%
"""
predict output
"""
input_dev = open('./data/dev.tag', 'rb')
output = open('./output.tag', 'w')
sys.stdout = output

count = 0
switch = True
lastgene = False

def signToTag(x, switch):
    if x==1.0:
        return 'TAG'
    else:
        if switch:    
            return 'GENE1'
        else:
            return 'GENE2'

for line in input_dev:
    count += 1
    if count%2 == 1 :
        print(line.split()[0])
        continue
    
    wt = line.split()
    lwords = [ token.rsplit('_',1)[0] for token in wt ]  
    
    viterbi = np.zeros(shape=(2, len(lwords)))
    
    for i in xrange(len(lwords)) :
        if i==0:
            log_prob  = logreg.predict_log_proba(toFeature(lwords[i], 'start', 1.0))[0]
            viterbi[0][i] = log_prob[0]
            viterbi[1][i] = log_prob[1]
            continue
        
        lastisminus = logreg.predict_log_proba( toFeature(lwords[i], lwords[i-1], -1.0) )[0][0] + viterbi[0][i-1]
        lastisplus  = logreg.predict_log_proba( toFeature(lwords[i], lwords[i-1], 1.0) )[0][0] + viterbi[1][i-1]
        viterbi[0][i] =  max(lastisminus,lastisplus)
        
        lastisminus = logreg.predict_log_proba( toFeature(lwords[i], lwords[i-1], -1.0) )[0][1] + viterbi[0][i-1]
        lastisplus  = logreg.predict_log_proba( toFeature(lwords[i], lwords[i-1], 1.0) )[0][1] + viterbi[1][i-1]
        viterbi[1][i] =  max(lastisminus,lastisplus)
    
    #backtrack
    ltags  = np.zeros(len(lwords), dtype=float)
    for i in np.arange(len(lwords)-1, -1, -1): 
        if i == (len(lwords)-1):        
            ltags[i] = 1.0 if viterbi[1][i] > viterbi[0][i] else -1.0
            continue
        
        zn = 1 if ltags[i+1] > 0 else 0        
        lastisminus = viterbi[0][i] + logreg.predict_log_proba( toFeature(lwords[i+1],lwords[i], -1.0) )[0][zn]
        lastisplus  = viterbi[1][i] + logreg.predict_log_proba( toFeature(lwords[i+1],lwords[i], 1.0)  )[0][zn]
        ltags[i] = 1.0 if lastisplus > lastisminus else -1.0
        
        

#    for i in xrange(len(lwords)): 
#        if i > 0 :
#            lastword = lwords[i-1] 
#            lasttag  = ltags[-1]
#        else:
#            lastword = '.'
#            lasttag  = 1.0
#        
#        predicted_tag = logreg.predict(toFeature( lwords[i], lastword, lasttag ))[0]
#        ltags.append(predicted_tag)
    
    
    ltagged = []
    for word, sign in zip(lwords, ltags):
        tag = signToTag(sign, switch)
        wtagged = word + '_' + tag
        ltagged.append(wtagged)
        
        if lastgene and sign > 0:
            switch = not switch
        
        lastgene = (sign<0)
        
    
    print " ".join(ltagged)
        


