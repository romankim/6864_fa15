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

def f0(x):
    return int('zeta' in x.lower())

# word that follows 'the'
#def f02(x):
#    return int( x.lower()=='the' )
## 'receptor' that follows gene

# a single capital letter that follows 'protein'

# more then 12 alphabet letters
def f05(x):
    return int( len(x) >= 10 )
# tumor necrosis

#ends with 'eron'
def f08(x):
    return int( len(x) > 4 and x[-4:].lower() == 'eron' )
# Roman numbers
def f09(x):
    return int( x == 'II' )
#rbcX is a gene
def f1(x):
    return int((len(x) == 4 and x[0:3] == 'rbc'))   
def f10(x):
    return int( x == 'III' )
def f11(x):
    return int( x == 'IV' )
def f12(x):
    return int( x == 'V' )
def f13(x):
    return int( x == 'VI' )
def f14(x):
    return int( x == 'VII' )
def f15(x):
    return int( x == 'VIII' )    


#'collagen' in word
def f17(x):
    return int( 'collagen' in x.lower()  )


def f19(x):
    return int( "immunoglobulin" in x.lower() )

def f21(x):
    return int(len(x) > 3 and x[-3:].lower=='tin')

def f22(x):
    return int(len(x) > 3 and x[-3:].lower=='nin')

def f23(x):
    return int(len(x) > 3 and x[-3:].lower=='ase')

def f26(x):
    return int( 'sf' in x.lower() and len(x) <= 5 )



#rbc is not a gene    
def f2(x):
    return int(x == 'rbc')
    
# if O2 is a substring, it's not a gene
def f3(x):
    return int('O2' in x)

def f31(x):
    return int(len(x) >= 3 and 'jun' in x[-3:].lower())

# capital letter more than length 3
def f32(x):
    return int(len(x) > 3 and x.isupper())


# contains at least two numbers and atleast one alphabet
def f34(x):
    return int( sum([1 for c in x if c.isdigit()]) >= 2 and sum([1 for c in x if c.isalpha()]) >= 1 )
    
#all lowercase and then capital at end
def f35(x):
    return int( len(x) >= 2 and x[:-1].islower() and x[-1].isupper() )

    
def f36(x):
    return int('globin' in x.lower())

def f37(x):
    return int('endorphin' in x.lower())
    
def f38(x):
    return int( 'mu' == x.lower() )

def f39(x):
    return int( 'X' in x or 'Z' in x)


# if CO is a substring, it's not a gene
def f4(x):
    return int('CO' in x)
    
# greek letters are likely to be genes
def f5(x):
    return int('alpha' in x.lower())
    
def f6(x):
    return int('beta' in x.lower())
    
def f7(x):
    return int('gamma' in x.lower())
    
def f8(x):
    return int('kappa' in x.lower())
    
def f9(x):
    return int('epsilon' in x.lower())
    
def fa(x):
    return int('sigma' in x.lower())

def fa1(x):
    return int('rho' in x.lower())

# synthetase is a gene
def fb(x):
    return int('synthetase' in x.lower())


    
# if it ends with 'rin', it's likely to be a gene
def fd(x):
    return int(  len(x) >= 3 and x[-3:] == 'rin'  )

# 'kinase' is a gene
def fe(x):
    return int('kinase' in x.lower())
    
# 'mutant' is a gene
def ff(x):
    return int('mutant' in x.lower())

# if lowercase vowel is not present, it is not an english word
def fg(x):
    bools = [(v in x) for v in vowels]
    return int( not any(bools)  )


    
# 'RNA" is likely in a gene
def fi(x):
    return int( 'RNA' in x )

# more than one uppercase letter
def fj(x):
    #remove_lower = x.translate(None, string.ascii_lowercase)
    #return int( len(remove_lower) >= 2 )
    return int(  sum([1 for c in x if c.isupper()]) > 2  and ('DNA' not in x ) and ('RNA' not in x) and ('PCR' not in x) )

# containts 'protein'
def fk(x):
    return int( 'protein' in x )
    
# combination of alphabet and numbers
def fl(x):
    remove_letter = x.translate(None, string.ascii_letters)
    return int( len(remove_letter) >= 1 and len(x) > len(remove_letter) )
   
# punctuations
def fm(x):
    return int( x==',' ) 
    
def fn(x):
    return int( x=='.' )

#'ase' is likely to be a gene
def fo(x):
    return int(  len(x) >= 3 and x[-3:] == 'ase'  )
    

# 'factor' is likely to be gene
def fs(x):
    return int( 'factor' in x.lower() )
# 'factor' following a gene is more likely to be gene
 
# a word that follows a gene
#def fu(x):
#    return int( z < 0 )
    

# period used not as EOF
def fx(x):
    return int( '.' in x and len(x) > 1 )

# '+' at the end of word
def fy(x):
    return int( len(x) > 1 and x[-1] == '+'  )


   
    
feature_list = [f for fname, f in sorted(globals().items()) if callable(f) and fname[0] == 'f' and (len(fname) == 2 or len(fname) == 3)]

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

for i in xrange(len(feature_list)):
    fn = feature_list[i]
    vfn = np.vectorize(fn)
    X[:,i] = vfn(words)
    
#%%
"""
Fit regression
"""

logreg = LogisticRegression(C=1e5)
logreg.fit(X,Y)

def toFeature(x):
    return [ fn(x) for fn in feature_list ]
    
logreg.predict( toFeature('WDX') )


#%%
"""
predict output
"""
input_dev = open('./data/test.tag', 'rb')
output = open('./output_test1.tag', 'w')
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
    ltags  = [ logreg.predict(toFeature(word))[0] for word in lwords ]
    
    ltagged = []
    for word, sign in zip(lwords, ltags):
        tag = signToTag(sign, switch)
        wtagged = word + '_' + tag
        ltagged.append(wtagged)
        
        if lastgene and sign > 0:
            switch = not switch
        
        lastgene = (sign<0)
        
    
    print " ".join(ltagged)
        


