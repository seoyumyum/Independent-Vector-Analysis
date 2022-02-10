import numpy as np
from scipy import signal
from scipy.linalg import sqrtm

def ConvBSS_IVA_NG(X, fs, N_fft=1024, N_hop=256, K=None, lr = 0.1, maxiter=1000, tol=1e-6):

    
    
    """  Fast algorithm for Frecuency Domain Blind source separation
            based on Independent Vector Analysis

        Parameters
        --------------------------------------------
        X : array containing mixtures, shape (# of mixture, # of samples) 

        fs : sampling frequency of mixture measurements in [Hz]

        N_fft : int, optional
                # of fft points (default = 1024)

        N_hop : int, optional
                Hop length of STFT (default = 256)

        K : int, optional
            # of sources. If None, same as the # of mixtures

        lr : float, optional
            Learning rate. (default = 0.1)

        max_iter: int, optional
                  Maximum number of iterations. (default = 1000)
        tol : float, optional 
              When the increment of likelihood is less than tol, the algorithm terminates (default = 1e-6)
        --------------------------------------------

        Returns
        --------------------------------------------
        y_t : Matrix containing separated sources, shape (# of sources, # of samples) 

        Y : STFT of y_t, shape(# of sources, # of freq bins, # of time bins)

        A : Matrix whose each column vector containing independent components (unscaled)
            shape (# of mixtures, # of sources, # of freq bins)
        --------------------------------------------



      - Original script in Matlab in Nov. 2, 2005 - Copyright: Taesu Kim
        Url: https://github.com/teradepth/iva/blob/master/matlab/ivabss.m

      - Citation: T. Kim, H. T. Attias, S. Lee and T. Lee, 
        "Blind Source Separation Exploiting Higher-Order Frequency Dependencies," 
        in IEEE Transactions on Audio, Speech, and Language Processing, 
        vol. 15, no. 1, pp. 70-79, Jan. 2007, doi: 10.1109/TASL.2006.872618.

      - Author (This python script): Hyungjoo Seo <seoyumyum@gmail.com>
        Date: 2/2/2022                                                      """

    
    M,_ = X.shape

    if K==None:
        K = M
    
    
    ## Perform short-time Fourier-Transform
    f, t, X_ft = signal.stft(X, fs, window = 'hamming', nperseg=N_fft, noverlap = N_fft-N_hop)

  
    epsi = 1e-6                                  ## For preventing overflow
    pObj = float("inf")                          ## Initialte with infinity 
    W = np.zeros((K,M,len(f)),dtype='complex')
    A = np.zeros((M,K,len(f)),dtype='complex')
    Wp = np.zeros((K,K,len(f)),dtype='complex')
    dWp = np.zeros(Wp.shape,dtype='complex')
    Q = np.zeros((K, M, len(f)),dtype='complex')
    Xp = np.zeros((K, len(f), len(t) ),dtype='complex')
    Y = np.zeros((K, len(f), len(t) ),dtype='complex')
    Ysq = np.zeros((K, len(t)),dtype='complex')
    Ysq1 = np. zeros((K, len(t)),dtype='complex')

    ### Whiten (PCA) at each frequency:
    for i in range(len(f)):
        Xmean, _ = center(X_ft[:,i,:])
        Xp[:,i,:], Q[:,:,i] = whiten(Xmean)
        Wp[:,:,i] = np.eye(K)



    ### Learning algorithm
    for iter in range(maxiter):
        dlw = 0
        for i in range(len(f)):
            Y[:,i,:] = Wp[:,:,i]@Xp[:,i,:]

        Ysq = np.sum(abs(Y)**2,axis=1)**0.5
        Ysq1 = (Ysq + epsi)**-1

        for i in range(len(f)):
            ## Calculate multivariate score function and gradients
            Phi = Ysq1*Y[:,i,:]
            dWp[:,:,i] = (np.eye(K) - Phi@Y[:,i,:].T.conj()/len(t))@Wp[:,:,i]
            dlw = dlw + np.log(abs(np.linalg.det(Wp[:,:,i])) + epsi)

        ## update unmixing matrices
        Wp = Wp + lr*dWp

        Obj = (sum(sum(Ysq))/len(t)-dlw)/(K*len(f))
        dObj = pObj-Obj
        pObj = Obj

        if iter%20 == 1:
            print(iter, 'iterations: Objective=', Obj, ', dObj=', dObj);


        if abs(dObj)/abs(Obj) < tol:
            print('Converged')
            break
        iter += 1


    ## Correct scaling of unmixing filter coefficients
    for i in range(len(f)):
        W[:,:,i] = Wp[:,:,i]@Q[:,:,i]
        A[:,:,i] = np.linalg.pinv(W[:,:,i])           ## This is optional.
        W[:,:,i] = np.diag(np.diag(A[:,:,i]))@W[:,:,i]

    ## Calculate outputs
    for i in range(len(f)):
        Y[:,i,:] = W[:,:,i]@X_ft[:,i,:]

    ## Recover signal to time domain
    y_t = np.zeros((K,len(X[0,:])))
    for i in range(K):
        _, y = signal.istft(Y[i,:,:], fs, nperseg=N_fft, noverlap = N_fft-N_hop)
        y_t[i,:]=y[:len(X[0,:])]

    return y_t, Y, A


def whiten(X):
    # Calculate the covariance matrix
    Xcov = np.cov(X, rowvar=True, bias=True)
    EigVal, EigVec = np.linalg.eigh(Xcov)
    SigmaInv = np.diag(1/EigVal**0.5)
    whiteM = EigVec@SigmaInv@np.transpose(EigVec).conj()            ## ZCA implementation
    #whiteM = SigmaInv@np.transpose(EigVec)            
    Xw = whiteM@X
    return Xw, whiteM

def whiten_complex(X, K):
    # Calculate the covariance matrix
    M = len(X)
    if M>K: 
        Xcov = np.cov(X, rowvar=True, bias=True)
        EigVal, EigVec = np.linalg.eigh(Xcov)
        k = np.argsort(EigVal)
        E = EigVal[k]
        bl_nv = np.sqrt(E[-K:]-E[:-K].mean())
        bl = 1/bl_inv
        whiteM = np.diag(bl).dot(EigVec[:,k[-K:]].conj().T)   
        Xw = whiteM@X
    else:
        W_inv      = sqrtm.linalg.sqrtm((X.dot(X.conj().T))/len(t))
        whiteM  = inv(W_inv)
        Xw = whiteM@X
    return Xw, whiteM


def center(X):
    mean = np.mean(X, axis=1, keepdims=True)
    centered =  X - mean 
    return centered, mean





































    