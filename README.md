# Natural-gradient-based Independent-Vector-Analysis (IVA) for convolutive mixtures in blind source separation problems written in Python

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
