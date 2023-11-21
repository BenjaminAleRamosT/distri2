# Neural Network: functions

import numpy  as np
    
# initialize weights
def iniWs(inshape, Param):
    W1 = randW(Param[3], inshape)
    
    if Param[2] == 2:
        W2 = randW(Param[4], Param[3])
        W3 = randW(2, Param[4])
        W = list((W1, W2))
    else:
        W3 = randW(2, Param[3])
        W = list((W1,W3))

    V = []
    for i in range(len(W)):
        V.append(np.zeros(W[i].shape))

    return W, V

# Rand values for W    
def randW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Random location for data
def randpermute(X, Y):
    
    indices_filas = np.arange(X.shape[0])
    
    np.random.shuffle(indices_filas)
    
    
    X = X[indices_filas, :]
    Y = Y[indices_filas, :]

    
    return X , Y

#Activation function
def act_functions(x, act=1, a_ELU=0.01, a_SELU=1.6732, lambd=1.0507):
    # Sigmoid

    if act == 1:
        return 1 / (1 + np.exp(-1*x))

    # tanh

    if act == 2:
        
        return 


    # Relu

    if act == 3:
        condition = x > 0
        return np.where(condition, x, np.zeros(x.shape))

    # ELU

    if act == 4:
        condition = x > 0
        return np.where(condition, x, a_ELU * np.expm1(x))

    # SELU

    if act == 5:
        condition = x > 0
        return lambd * np.where(condition, x, a_SELU * np.expm1(x))

    

    return x

def deriva_act(x, act=1, a_ELU=0.01, a_SELU=1.6732, lambd=1.0507):
    # Sigmoid

    if act == 1:
        # pasarle la sigmoid
        return np.multiply(act_functions(x, act=5), (1 - act_functions(x, act=5)))

    # tanh

    if act == 2:
        
        return 

    # Relu

    if act == 3:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), np.zeros(x.shape))

    
    # ELU

    if act == 4:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), a_ELU * np.exp(x))

    # SELU

    if act == 5:
        condition = x > 0
        return lambd * np.where(condition, np.ones(x.shape), a_SELU * np.exp(x))

    
    return x


#Feed-forward 
def forward(X, W, Param):        
    # cambiar activaciones por config
    act_encoder = Param[5]

    A = []
    z = []
    Act = []

    # data input
    z.append(X)
    A.append(X)

    # iter por la cantidad de pesos
    for i in range(len(W)):
        # print(W[i].shape, X.shape)
        X = np.dot(W[i], X.)
        z.append(X)
        if i == len(W)-1:
            X = act_functions(X, act=1)
        else:
            X = act_functions(X, act=Param[5])

        A.append(X)

    Act.append(A)
    Act.append(z)

    return Act
# Feed-Backward 
def gradWs(Act,Y, W, Param):
    
    act_encoder = Param[5]

    L = len(Act[0])-1

    N = Param[1]

    e = Act[0][L] - Y

    Cost = np.sum(np.sum(np.square(e), axis=0)/2)/N

    # grad decoder
    
    delta = np.multiply(e, deriva_act(Act[1][L], act=1))

    gW_l = np.dot(delta, Act[0][L-1].T)/N
    gW = []
    gW.append(gW_l)

    # grad encoder
    for l in reversed(range(1,L)):
        
        t1 = np.dot(W[l].T, delta)

        t2 = deriva_act(Act[1][l], act=Param[5])

        delta = np.multiply(t1, t2)

        t3 = Act[0][l-1].T

        gW_l = np.dot(delta, t3)
        gW.append(gW_l)

    gW.reverse()

    return gW, Cost        

# Update MLP's weigth using iSGD
def updWs(W, gW, V, Param, ite, T):
    
    u = Param[6]

    t = 1-(ite/T)
    beta = (0.9*t)/(0.1+(0.9*t))

    for i in range(len(W)):
        V[i] = (beta * V[i]) - (u*gW[i])
        W[i] = W[i] + V[i]

    return W, V
# Measure
def metricas(x,y):
    cm     = confusion_matrix(x,y)
    #z = z[0][-1]
    
    #z = np.asarray(z).squeeze()
    
    TP = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TN = cm[1,1]
        
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    Fsc = (( 2 * Precision * Recall ) / ( Precision + Recall ))
     
    return cm , Fsc
    
#Confusion matrix
def confusion_matrix(z, y):
    y,z = y.T,z.T
    m= y.shape[0]
    c = y.shape[1]
    
    y = np.argmax(y, axis=1)
    
    z = np.argmax(z, axis=1)
   
    cm = np.zeros((c,c))
    
    for i in range(m):
         cm[z[i] ,y[i]] +=1
    
    return cm

#

