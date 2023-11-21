# Data and Parameters

import numpy  as np
import pandas as pd ###Caambiar esta verga
#
def load_config(ruta_archivo='config.csv'):

    with open(ruta_archivo, 'r') as archivo_csv:

        conf = [int(i) if '.' not in i else float(i)
                for i in archivo_csv if i != '\n']

    return conf
# training data load
def load_dtrain(path_csv_x='xtrain.csv',path_csv_y='ytrain.csv'):

    x = np.genfromtxt(path_csv_x, delimiter=',').T
    y = np.genfromtxt(path_csv_y, delimiter=',').T

    return x , y

#matrix of weights and costs
def save_ws_costo(W,Cost):
    np.savez('Ws.npz', *W)

    df = pd.DataFrame( Cost )
    df.to_csv('costo_avg.csv',index=False, header = False )

    return

#load pretrained weights
def load_ws():

    ws = np.load('Ws.npz')

    ws = [ws[i] for i in ws.files]

    return ws

#save metrics
def save_metric(cm,Fsc):

    cm = pd.DataFrame( cm )
    cm.to_csv('cmatrix.csv',index=False, header = False )
    
    Fsc = pd.DataFrame( Fsc )
    Fsc.to_csv('fscores.csv',index=False, header = False )

    return