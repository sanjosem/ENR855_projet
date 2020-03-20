class CollineModeleClass:
    '''Cette classe integre les fonctions d'acces et de calculs aux donnees fournies dans le devoir
    Pour l'initialiser, il faut fournir le chemin du fichier excel ainsi que le Reynolds de reference
    inputs:
    -------
      input_excel_colline: chemin absolu du fichier excel a charger
      Reynolds_modele: valeur du nombre de Reunolds des donnees fournies
      convert_to_m3s: bool (optionnel,defaut: False), si vrai Q11 brut sera suppose en l/s et convertit en m3/s
    '''
    def __init__(self,input_excel_colline,Reynolds_modele,convert_to_m3s=False):
        self.file = input_excel_colline
        self.ReM = Reynolds_modele
        self.loaded = False
        self.interpolated = False
        self.convert_Q11 = convert_to_m3s

    def raw_colline(self):
        '''Lit les donnees brutes fournies dans le fichier excel et les rend accessibles aux autres fonctions'''
        import pandas as pd
        import numpy as np
        
        m3s_to_ls = 1000.0
        
        colline = pd.read_excel(self.file,sheet_name=['Colline_Débit','Colline_Rendement'])

        colline['Colline_Rendement'].rename(columns={0: 'Gamma'},inplace=True)
        colline['Colline_Débit'].rename(columns={0: 'Gamma'},inplace=True)

        # Pour faciliter les traces les donnees sont reorganisees en tableaux a 2 dimensions (numpy)
        self.n11 = np.asarray(colline['Colline_Débit'].columns[1:],dtype='float')
        self.gamma = np.asarray(colline['Colline_Débit']['Gamma'],dtype='float')

        self.N11_map,self.G_map = np.meshgrid(self.n11,self.gamma)

        if self.convert_Q11:
            self.Q11_map = colline['Colline_Débit'].iloc[:,1:].values / m3s_to_ls
        else:
            self.Q11_map = colline['Colline_Débit'].iloc[:,1:].values            
            
        self.eta_map = colline['Colline_Rendement'].iloc[:,1:].values
        
        self.loaded = True

    def get_raw_colline(self):
        '''Retourne les donnees brutes fournies dans le fichier excel'''
        if not self.loaded:
            self.raw_colline()
        return self.gamma,self.n11,self.Q11_map,self.eta_map 
    

    def interpol_colline(self,spline_order=3,smoothing=0):
        import pandas as pd
        import numpy as np
        import scipy.interpolate as inpt

        # donnees utilisateurs (optionnelles)
        # Ordre de la spline (entree utilisateur ou valeur par defaut 3)
        self.spline_order = spline_order
        # Parametre de lissage
        self.smoothing = smoothing
        
        # Donnees brute
        gamma,n11,Q11_map,eta_map = self.get_raw_colline()
        
        # Fonction d'interpolation sur une grille reguliere
        # Approximation spline a deux variables sur le maillage regulier : $f(\gamma,n_{11}) = Q_{11}$
        # `spline_Q11` et `spline_eta` sont des instances de la classe `UnivariateSpline`. 
        # La fonction spline est alors evaluee en appelant la fonction `spline_Q11.ev`
        
        # On elargit le domaine de course des directrices (extrapolation)
        gamma_min = gamma.min()
        gamma_max = 40.0
        self.gamma_range = [gamma_min,gamma_max]
        n11_min = n11.min()
        n11_max = n11.max()
        self.n11_range = [n11_min,n11_max]
        
        spline_Q11 = inpt.RectBivariateSpline(gamma,n11,Q11_map,bbox=[gamma_min,gamma_max,n11_min,n11_max],
                                          kx=self.spline_order,ky=self.spline_order,s=self.smoothing)
        spline_eta = inpt.RectBivariateSpline(gamma,n11,eta_map,bbox=[gamma_min,gamma_max,n11_min,n11_max],
                                          kx=self.spline_order,ky=self.spline_order,s=self.smoothing)
        
        self.f_Q11 = spline_Q11.ev
        self.f_eta = spline_eta.ev
        self.interpolated = True

    def get_Q11_n11_gamma(self,D,Q,nrpm,H):
        '''Calcule Q11 et N11 et trouve l'ouverture de directrice qui fournit le debit a la vitesse de rotation specifiee grace a la fonction d'interpolation
        inputs:
        -------
            D: diametre de la roue
            Q: debit volumique (meme unite que Q11)
            nrpm: vitesse de rotation en tour/min (rpm)
            H: chute nominale en m
        outputs:
        --------
            Q11: parametre de similitude du systeme Q11-N11 (meme unite que Q)
            N11: parametre de similitude du systeme Q11-N11 (en rpm)
            gamma: ouverture de directrice (en degrees)
        '''
        from numpy import sqrt
        
        Q11 = Q /(D**2 * sqrt(H))
        N11 = nrpm*D/sqrt(H)
        gamma = self.lsq_get_gamma_from_Q11_N11(Q11,N11)
        
        return Q11,N11,gamma

    def sync_rot(self,n_rpm,Q,D,H):
        '''Trouve la vitesse de rotation synchrone qui donne le meilleur rendement
        inputs:
        -------
            n_rpm: vitesse de rotation en tour/min (rpm)
            Q: debit en m3/s
            D: diametre de la roue
            H: chute nominale en m
        outputs:
        --------
            n_rpm_sync: vitesse de rotation synchrone en tour/min (rpm)
        '''
        import numpy as np
        
        if not self.interpolated:
            # En utilisant les parametres par defaut de l'interpolation
            self.interpol_colline()
        
        # minorant et majorant
        # 3600 = 60 Hz convertit en 1/min
        n_int = np.asarray((np.ceil(3600/n_rpm),np.floor(3600/n_rpm)))
        n_rpm_sync = 3600/n_int
        Q11,N11,gamma = self.get_Q11_n11_gamma(D,Q,n_rpm_sync,H)
        eta_sync = self.f_eta(gamma,N11)
        # trouve l'indice qui correspond au meilleur rendement
        idx = np.argmax(eta_sync)
        print('Nombre de poles: {0:.0f}'.format(n_int[idx]))
        
        return n_rpm_sync[idx]
    
    def lsq_get_gamma_from_Q11_N11(self,Q11,N11):
        '''Trouve l'ouverture de directrice correspondant au point (Q11,N11) sur la colline interpolee grace a un algorithme des moindres carres
        inputs:
        -------
            Q11: parametre de similitude du systeme Q11-N11 (en l/s)
            N11: parametre de similitude du systeme Q11-N11 (en rpm)
        outputs:
        --------
            gamma: ouverture de directrice en degrees
        '''
        from numpy import abs
        from scipy.optimize import least_squares
        
        if not self.interpolated:
            # En utilisant les parametres par defaut de l'interpolation
            self.interpol_colline()
        
        def res_func(x):
            res = abs(self.f_Q11(x,N11) - Q11)
            return res
        init_guess = 20.
        gamma = least_squares(res_func,init_guess,bounds=self.gamma_range)
        
        return gamma.x

    def plot_colline(self,npts_gamma=50,npts_n11=50,nlevels=15):
        '''Trace la colline interpolee sur les plages specifiees
        inputs:
        -------
            npts_gamma: (default: 50) nombre de points entre les bornes de l'interpolation
            npts_n11: (default: 50) nombre de points entre les bornes de l'interpolation
            nlevels: (default: 15) nombre de contours
        outputs:
        --------
            fig: pointer vers l'objet figure
            ax: list des pointers vers les axes de la figure
        '''
        import matplotlib.pyplot as plt
        import matplotlib.ticker as tck
        import numpy as np
        
        # quelques parametres pour l'affichage des figures
        plt.rcParams['font.size'] = 16
        plt.rcParams['figure.figsize'] = (10,8)
        plt.rcParams['image.cmap'] = 'inferno'
        
        gamma_range=self.gamma_range
        n11_range=self.n11_range
        gamma_col = np.linspace(gamma_range[0],gamma_range[1],npts_gamma)
        n11_col = np.linspace(n11_range[0],n11_range[1],npts_n11)
        GM,NM = np.meshgrid(gamma_col,n11_col)
        Q11M = self.f_Q11(GM,NM)
        ETAM = self.f_eta(GM,NM)
        z=[]
        fig,ax = plt.subplots(1,2, figsize=(15,5))
        plt.subplots_adjust(hspace=0.1,wspace = 0.3,left = 0.17,right = 0.97,bottom = 0.1,top = 0.93)
        z.append(ax[0].contour(NM,GM,Q11M,nlevels))
        ax[0].set_title(r'$Q_{11} \ [m^3/s]$')
        fmt = tck.FormatStrFormatter("%.3f")
        fmt.create_dummy_axis()
        ax[0].clabel(z[0],z[0].levels, fmt=fmt, inline=True)
        levels = np.linspace(0.8,.94,nlevels)
        z.append(ax[1].contour(NM,GM,ETAM,levels))
        ax[1].set_title(r'$\eta_{M} \ [\%]$')
        fmt = tck.PercentFormatter(xmax=1.0,decimals=1)
        fmt.create_dummy_axis()
        ax[1].clabel(z[1],z[1].levels, fmt=fmt, inline=True)
        for axi,zi  in zip(ax,z):
            axi.set_xlabel(r'$n_{11} \ [rpm]$')
            axi.set_ylabel(r'$\gamma \ [\circ]$')
            axi.grid()  
            # plt.colorbar(zi,ax=axi);
        return fig,ax
