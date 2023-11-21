# MLP's Trainig 

import pandas      as pd
import numpy       as np
import nnetwork    as nn
import data_param  as dpar

# Training by use miniBatch iSGD
def train_miniBatch(X,Y,W,V,Param):
    Costo=0
    Batch_size = Param[1]
    gW = []
    T = int(np.floor(len(X)/Batch_size))
    for i in range(T):
        
        xe,ye = X[Batch_size*i:(Batch_size*i)+Batch_size].T, Y[Batch_size*i:(Batch_size*i)+Batch_size].T
        
        Act = nn.forward(xe, W, Param)
        gW,C = nn.gradWs(Act,ye, W, Param)
        
        Costo =+ C/T
        W,V = nn.updWs(W, gW, V, Param, i, T)
        
    
    return W,Costo

# mlp's training 
def train_mlp(x,y,param):        
    W,V = nn.iniWs(x.shape[1], param)
    Batch_size = param[1]                    
    for Iter in range(1,param[0]):        
        xe,ye = nn.randpermute(x,y)
        
        W,Costo = train_miniBatch(xe,ye,W,V,param)
        
        if ((Iter %20)== 0):
            print('Iter={} Cost={:.5f}'.format(Iter,Costo[-1]))    
            
    return W, Costo


# Beginning ...
def main():
    param       = dpar.load_config()        
    x,y         = dpar.load_dtrain()   
    W,costo     = train_mlp(x,y,param)         
    dpar.save_ws_costo(W,costo)
       
if __name__ == '__main__':   
	 main()

