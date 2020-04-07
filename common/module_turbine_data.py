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
        self.max_interpolated = False
        self.loaded_geometry = False
        self.proto_geom = None

    def raw_colline(self):
        '''Lit les donnees brutes de colline fournies dans le fichier excel et les rend accessibles aux autres fonctions'''
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
        '''Retourne les donnees brutes fournies dans le fichier excel
        va recuperer dans la classe les donnees brutes issues du tableau excel:
        outputs:
        --------
          gamma: vecteur de valeurs de gamma
          n11: vecteur de valeurs de n11
          Q11_map: tableau (2D) du parametre de similitude Q11 en fonction des couples (gamma,n11) en m3/s
          eta_map: tableau (2D) du rendement modele eta en fonction des couples (gamma,n11) 
        '''
        if not self.loaded:
            self.raw_colline()
        return self.gamma,self.n11,self.Q11_map,self.eta_map         
    
    def interpol_colline(self,spline_order=3,smoothing=0):
        '''Cree les fonctions d'interpolation des donnees brutes fournies dans le fichier excel
        Cette routine equivaut a l'utilisation de interpo2 sous excel
        Une fois les noeuds et coefficients des fonction spline bivariee cree, celles-ci pourrant etre appelees 
        par self.f_Q11(gamma,n11) et self.f_eta(gamma,n11).
        inputs:
        --------
          spline_order (optionnel, defaut 3): ordre de la fonction spline pour l'interpolation bivariee
          smoothing (optionnel, defaut 0): parametre de lissage (0 = pas de lissage)
        '''
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
        gamma_max = gamma.max()
        self.gamma_range = [gamma_min,gamma_max]
        n11_min = n11.min()
        n11_max = n11.max()
        self.n11_range = [n11_min,n11_max]
        print([gamma_min,gamma_max])
        print([n11_min,n11_max])
        
        spline_Q11 = inpt.RectBivariateSpline(gamma,n11,Q11_map,bbox=[gamma_min,gamma_max,n11_min,n11_max],
                                          kx=self.spline_order,ky=self.spline_order,s=self.smoothing)
        spline_eta = inpt.RectBivariateSpline(gamma,n11,eta_map,bbox=[gamma_min,gamma_max,n11_min,n11_max],
                                          kx=self.spline_order,ky=self.spline_order,s=self.smoothing)
        
        self.f_Q11 = spline_Q11.ev
        self.f_eta = spline_eta.ev
        self.interpolated = True
        
    def interpol_max_colline(self,spline_order=3):
        '''Cree les fonctions pour reperer le maximum de rendment a tout les points n11
        Une fois les noeuds et coefficients des fonctions spline univariee cree, celles-ci pourront etre appelees 
        par self.f_eta_max(n11) et self.f_gd_max(n11) pour obtenir le rendement max et l'angle d'ouverture 
        directrice correspondant
        inputs:
        --------
          spline_order (optionnel, defaut 3): ordre de la fonction spline pour l'interpolation bivariee
        '''
        import numpy as np
        import scipy.interpolate as inpt
        
        if not self.interpolated:
            self.interpol_colline() # avec les parametres par defaut
            
        # Fonction d'interpolation du rendement max en fonction de n11
        eta_M_max = self.eta_map.max(axis=0)
        gamma_max = np.zeros_like(eta_M_max)
        iv = 0
        for n11v,etav in zip(self.n11,eta_M_max):
            gamma_max[iv] = self.lsq_get_gamma_from_N11_eta(n11v,etav)
            iv+=1
        idx_eta_M_max = self.eta_map.argmax(axis=0)
        raw_gamma_max = self.gamma[idx_eta_M_max]
            
        if spline_order==1:
            splnm = 'slinear'
        elif spline_order==2:
            splnm = 'quadratic'
        elif spline_order==3:
            splnm = 'cubic'
        else:
            print('spline_order not implemented !')
            return
            
        f_eta_max = inpt.interp1d(self.n11,eta_M_max,kind=splnm)
        
        self.f_eta_max = f_eta_max
        self.max_interpolated = True
        
        
    def prise_charge(self,n11,D,rho,g,H,nu,npts_intp=100,flag_plot=True,d_eta_maj_pm=None):
        '''La fonction prise_charge realise les differente etape suivante
        1. Cherche le n11 le plus proche correspondant a une vitesse synchrone et au diametre fourni.
        2. La majoration de rendement est calculee au point de rendement max pour la vitesse synchrone n11 obtenue à moins qu'elle ne soit specifiee en entree. 
        3. Pour cette vitesse synchrone, une prise de charge est realisee, c'est a dire que l'angle des directrices est balaye sur la plage des donnees. Pour ces ouvertures, le debit et le rendement modele majore (ou rendement prototype estime) sont obtenus par interpolation sur la colline de rendement. 
        4. La puissance prototype sur la prise de charge est egalement calculee. 
        5. Des traces sont fournis par defaut.
        
        inputs:
        -------
            n11: pour la prise de charge souhaitee
            D: diametre de la roue prototype
            rho: masse volumique en kg/m3
            g: acceleration de la gravite
            H: chute nominale en m
            nu: viscosite cinematique 
            npts_intp (optionnel): nombre de point du balayage de directrice (defaut: 100)
            flag_plot (optionnel): montre la prise de charge obtenue (defaut: True)
            d_eta_maj_pm (optionnel) : majoration du rendement prototype (default: None)
            
        outputs:
        --------
            n_rpm_sync: vitesse synchrone (en rpm)
            d_eta_maj_pm: majoration du rendement prototype (calculee au point de rendement maximum ou specifiee en input)
            gamma_pc: vecteur de variation de l'angle de directrice sur la prise de charge n11=n11_sync (en deg)
            Q_pc: variation du debit le long de la prise de charge n11=n11_sync (en m3/s)
            eta_P_pc: rendement prototype le long de la prise de charge n11=n11_sync
            P_mp: puissance prototype (en W)
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as tck
    
        n_rpm_sync,n11_sync = self.sync_rot_n11(n11,D,H)
        
        if d_eta_maj_pm is None:
            eta_M_opt = self.f_eta_max(n11_sync)
            print('eta_M_max = {0:.3f} %'.format(eta_M_opt*100)) 
            Re_p = (2*np.pi * n_rpm_sync / 60.) * D/2 * D / nu
            print('Re_p = {0:.3e}'.format(Re_p)) 
            print('Pertes relatives transposables à Re_M={0:.0e} avec V_ref=0.7 pour une turbine Francis'.format(self.ReM))
            delta_M = (1.0 - eta_M_opt) * 0.7
            print('delta_M={0:0.3f} %'.format(delta_M*100.))
            print('Majoration du rendement calculee')
            d_eta_maj_pm = delta_M * (1 - (self.ReM/Re_p)**0.16)
            print('d_eta_maj_pm={0:0.3f} %'.format(d_eta_maj_pm*100.))
        else:
            print('Majoration du rendement specifiee')
            print('d_eta_maj_pm={0:0.3f} %'.format(d_eta_maj_pm*100.))
        
        gamma_pc = np.linspace(self.gamma_range[0],self.gamma_range[1],npts_intp)
        eta_M_pc = self.f_eta(gamma_pc,n11_sync)
        keep = eta_M_pc>0.8
        
        gamma_pc_interest = gamma_pc[keep]
        Q11_pc = self.f_Q11(gamma_pc_interest,n11_sync)
        Q_pc = Q11_pc * np.sqrt(H) * D**2
        eta_M_pc = self.f_eta(gamma_pc_interest,n11_sync)
        eta_P_pc = eta_M_pc + d_eta_maj_pm
        P_mp = rho * g * H * Q_pc * eta_P_pc
        
        # Rendement max
        gd_max_pc = self.lsq_get_gamma_from_N11_eta(n11_sync,eta_M_pc.max())
        Q_eta_max = self.f_Q11(gd_max_pc,n11_sync) * np.sqrt(H) * D**2
        eta_P_max = self.f_eta(gd_max_pc,n11_sync) + d_eta_maj_pm
        P_mp_max = rho * g * H * Q_eta_max * eta_P_max        
        
        if flag_plot:
            n11_i = np.linspace(self.n11_range[0],self.n11_range[1],npts_intp)            
            
            label = [r'$n_{11} = $','{0:.2f}'.format(n11_sync),' [rpm]']
            fig,ax = self.plot_colline()
            ax[0].plot(n11_sync*np.ones_like(gamma_pc_interest),gamma_pc_interest,linewidth=2.0,color='blue')
            ax[0].text(1.01*n11_sync, gamma_pc_interest[-3],' '.join(label), color='blue')
            ax[1].plot(n11_sync*np.ones_like(gamma_pc_interest),gamma_pc_interest,linewidth=2.0,color='blue')
            ax[1].text(1.01*n11_sync, gamma_pc_interest[0]-4,' '.join(label), color='blue')
            ax[0].set_xlim(60,105)
            ax[1].set_xlim(60,105)
            
            fig2,ax2 = plt.subplots(1,2, figsize=(15,5))
            plt.subplots_adjust(hspace=0.1,wspace = 0.3,left = 0.17,right = 0.97,bottom = 0.1,top = 0.93)
            ax2[0].plot(gamma_pc_interest, Q_pc,label='prise de charge')
            ax2[0].plot(gd_max_pc,Q_eta_max,'+',color='green',ms=15,markeredgewidth=3.0,label='Sommet')
            ax2[0].set_ylabel(r'$Q_p \ [m^3/s]$')
            ax2[0].set_xlabel(r'$\gamma \ [\circ]$')
            ax2[1].plot(P_mp/1.0e6,eta_P_pc,label='prise de charge')
            ax2[1].plot(P_mp_max/1.0e6,eta_P_max,
                        marker='+',color='green',ms=15,markeredgewidth=3.0,label='Sommet',linestyle='none')
            ax2[0].grid()
            ax2[1].set_ylabel(r'$\eta_p \ [\%]$')
            ax2[1].set_xlabel(r'$P_{m,p}\ [MW]$')
            ax2[1].yaxis.set_major_formatter(tck.PercentFormatter(xmax=1.0))
            ax2[1].grid()
            
        if flag_plot:
            return n_rpm_sync, d_eta_maj_pm, gamma_pc_interest, Q_pc, eta_P_pc, P_mp, fig, ax, fig2, ax2
        else:
            return n_rpm_sync, d_eta_maj_pm, gamma_pc_interest, Q_pc, eta_P_pc, P_mp
        
                    

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
            D: diametre de la roue en m
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
        
    def sync_rot_n11(self,n11,D,H):
        '''Trouve la vitesse de rotation synchrone qui donne le meilleur rendement donnant un n11 proche de la valeur fournie
        inputs:
        -------
            n11: parametre de similitude vitesse de rotation en tour/min (rpm)
            D: diametre de la roue en m
            H: chute nominale en m
        outputs:
        --------
            n_rpm_sync: vitesse de rotation synchrone en tour/min (rpm)
            n11_sync: n11 ajutse pour la vitesse synchrone en tour/min (rpm)
        '''
        import numpy as np
        
        if not self.max_interpolated:
            # En utilisant les parametres par defaut de l'interpolation
            self.interpol_max_colline()
        n_rpm_guess = n11*np.sqrt(H)/D
        
        # minorant et majorant
        # 3600 = 60 Hz convertit en 1/min
        n_int = np.asarray((np.ceil(3600/n_rpm_guess),np.floor(3600/n_rpm_guess)))
        n_rpm_sync = 3600/n_int
        
        # choix de n_rpm_sync au rendement max
        n11_sync = n_rpm_sync * D / np.sqrt(H)
        eta_max = self.f_eta_max(n11_sync)
        idx_max = eta_max.argmax()
        print('Nombre de poles: {0:.0f}'.format(n_int[idx_max]))
        return n_rpm_sync[idx_max], n11_sync[idx_max]
    
    def lsq_get_gamma_from_Q11_N11(self,Q11,N11):
        '''Trouve l'ouverture de directrice correspondant au point (Q11,N11) sur la colline interpolee grace a un algorithme des moindres carres
        inputs:
        -------
            Q11: parametre de similitude du systeme Q11-N11 (en m/s)
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
        
    def lsq_get_gamma_from_N11_eta(self,N11,eta):
        '''Trouve l'ouverture de directrice correspondant fournissant le rendement eta au n11 donne sur la colline interpolee grace a un algorithme des moindres carres
        inputs:
        -------
            N11: parametre de similitude du systeme Q11-N11 (en rpm)
            eta: rendement au n11 fourni
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
            res = abs(self.f_eta(x,N11) - eta)
            return res
        init_guess = 20.
        gamma = least_squares(res_func,init_guess,bounds=self.gamma_range)
        
        return gamma.x
        
    def lsq_get_N11_from_gamma_eta(self,gamma,eta,n11_guess):
        '''Trouve la vitesse correspondante fournissant le rendement eta au gamma donne sur la colline interpolee grace a un algorithme des moindres carres
        inputs:
        -------
            gamma: ouverture de directrice (en deg)
            eta: rendement modele au gamma fourni
            n11_guess: n11 proche de la valeur souhaitee
            
        outputs:
        --------
            n11: parametre de similitude recherche
        '''
        from numpy import abs
        from scipy.optimize import least_squares
        
        if not self.interpolated:
            # En utilisant les parametres par defaut de l'interpolation
            self.interpol_colline()
        
        def res_func(x):
            res = abs(self.f_eta(gamma,x) - eta)
            return res
        n11 = least_squares(res_func,n11_guess,bounds=self.n11_range)
        
        return n11.x

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
        n11_col = np.linspace(n11_range[0],105,npts_n11)
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
        
    # Fonctions liees a la geometrie
    
    def raw_geometry(self):
        '''Lit les donnees brutes de geometrie fournies dans le fichier excel et les rend accessibles aux autres fonctions'''
        import pandas as pd
        
        geom_modele = pd.read_excel(self.file,sheet_name='Géométrie', usecols='A:D',skiprows=4,nrows=11,header=None)
        # correct column header
        geom_modele.columns = ['ID','Description','Valeur','Unite']
        # correct NaN value
        geom_modele.loc[geom_modele.index[-1],'ID']='S'
        
        self.modele_geometry = geom_modele
        self.loaded_geometry = True
        
    def get_raw_geometry(self):
        '''Retourne les donnees brutes geometriques chargees dans la classe
        outputs:
        --------
          geom_modele: tableau des donnees geometrique du modele
        '''
        if not self.loaded_geometry:
            self.raw_geometry()
        
        geom_modele =self.modele_geometry
        
        return geom_modele

    def scale_proto_geometry(self,D,units='m'):
        '''Cree les parametres geometriques du prototype en appliquant la similitude geometrique a partir de la geometrie du modele fourni
        inputs:
        -------
            D: diametre du prototype
            units: (default: 'm') unite du diametre prototype fournit 'm' ou 'mm' 
        '''
        
        if units not in ['mm','m']:
            print('Error with units')
            return
        geom_modele = self.get_raw_geometry()
        D_M = geom_modele.loc[geom_modele['ID']=='A',['Valeur','Unite']]
        if units == D_M.iloc[0]['Unite']:
            D_P = D
            Lstar = D_P/D_M.iloc[0]['Valeur']
        else: 
            assert D_M.iloc[0]['Unite']=='mm'
            D_P = D*1000
            Lstar = D_P/D_M.iloc[0]['Valeur']
        
        proto_geom = geom_modele.copy(deep=True)
        for iv,key in enumerate(proto_geom['ID']):
            if not key == 'S':
                proto_geom.loc[iv,'Valeur'] *= Lstar
            else:
                proto_geom.loc[iv,'Valeur'] *= Lstar**2
                
        self.proto_geom = proto_geom
        return proto_geom

    def calc_sigma(self,z0,p_atm,Bief_aval,p_va,Q,H,rho,g):
        '''Calcule le nombre de Thomas pour l'enfoncement choisi. 
        inputs:
        -------
            z0: altitude du fond de l'aspirateur
            p_atm: pression atmospherique au bief aval
            Bief_aval: altitude du bief aval
            p_va: pression de vapeur saturante
            H: chute nette
            Q: debit volumique en m3/s
            rho: masse volumique
            g: acceleration de la gravite
        output:
        -------
            sigma: nombre de Thomas
            volume_trou: volume de construction de l'unite
        '''
        from numpy import sqrt
        assert self.proto_geom is not None, "La geometrie du prototype doit etre definie avec la fonction scale_proto_geometry"
        proto_geom = self.proto_geom
        
        D  = float(proto_geom.loc[proto_geom['ID']=='A','Valeur']/1000)
        VD = float(proto_geom.loc[proto_geom['ID']=='D','Valeur']/1000)
        VC = float(proto_geom.loc[proto_geom['ID']=='C','Valeur']/1000)
        VJ = float(proto_geom.loc[proto_geom['ID']=='J','Valeur']/1000)
        VF = float(proto_geom.loc[proto_geom['ID']=='F','Valeur']/1000)
        VS = float(proto_geom.loc[proto_geom['ID']=='S','Valeur'])
        
        zref = z0+VD-VC
        print('Altitude de reference zref={0:.2f} m'.format(z0+VD-VC))
        
        hs = zref - Bief_aval
        print('Hauteur de sustentation hs={0:.2f} m'.format(hs))
        
        h_atm = p_atm / (rho*g)
        h_v = p_va / (rho*g)
        
        aire_aspi = VJ*VF
        v2 = Q/aire_aspi
        hf_aspi = v2**2/(2*g)
        
        sigma = (h_atm + hf_aspi - hs - h_v)/H
        Q11 = Q /(D**2 * sqrt(H))
        k11 = sigma/Q11**2
        
        volume_trou = VS * (Bief_aval-z0)
        
        return sigma, k11, volume_trou

    def target_k11(self,k11,p_atm,Bief_aval,p_va,Q,H,rho,g):
        '''Calcule l'enfoncement pour le nombre de Thomas choisi'
        inputs:
        -------
            k11: constante sigma/Q11^2 de design de la turbine 
            p_atm: pression atmospherique au bief aval
            Bief_aval: altitude du bief aval
            p_va: pression de vapeur saturante
            H: chute nette
            Q: debit volumique en m3/s
            rho: masse volumique
            g: acceleration de la gravite
        output:
        -------
            hs: hauteur de sustentation correspondant au nombre de Thomas vise
            z0: cote du fond de l'aspirateur
            volume_trou: volume de construction de l'unite
        '''
        from numpy import sqrt
        
        assert self.proto_geom is not None, "La geometrie du prototype doit etre definie avec la fonction scale_proto_geometry"
        proto_geom = self.proto_geom
        
        D  = float(proto_geom.loc[proto_geom['ID']=='A','Valeur']/1000)
        VD = float(proto_geom.loc[proto_geom['ID']=='D','Valeur']/1000)
        VC = float(proto_geom.loc[proto_geom['ID']=='C','Valeur']/1000)
        VJ = float(proto_geom.loc[proto_geom['ID']=='J','Valeur']/1000)
        VF = float(proto_geom.loc[proto_geom['ID']=='F','Valeur']/1000)
        VS = float(proto_geom.loc[proto_geom['ID']=='S','Valeur'])
        
        Q11 = Q /(D**2 * sqrt(H))
        sigma = k11 * Q11**2
        print('Nombre de Thomas vise: {0:.4f}'.format(sigma))
        
        h_atm = p_atm / (rho*g)
        h_v = p_va / (rho*g)
        
        aire_aspi = VJ*VF
        v2 = Q/aire_aspi
        hf_aspi = v2**2/(2*g)        
        
        hs = h_atm + hf_aspi - sigma * H - h_v
        print('Hauteur de sustentation necessaire hs={0:.3f} m'.format(hs))
        
        zref = hs + Bief_aval
                
        z0 = zref - VD+VC
        print('Altitude du fond aspirateur z0={0:.3f} m'.format(z0))
        
        volume_trou = VS * (Bief_aval-z0)
        
        return hs, z0, volume_trou

    def allievi_method_normal(self,Tstop,H_0,gamma_0,n_rpm_sync,D,d_eta_maj_pm,L_duct,A_duct,rho,
                              csound,g,method='SLSQP',tol=1.0e-6):
        import pandas as pd
        import numpy as np
        from scipy.optimize import least_squares,minimize
        
        Tstep = 2*L_duct/csound
        nstep = int(Tstop / Tstep) + 2
        
        transient = pd.DataFrame(columns=['t','gamma','H','n_rpm','Q','eta_P','P_m','C_m','C_r'])
        def calc_op(gamma_curr,H_curr,n_rpm_curr):
            om_curr =  (2*np.pi * n_rpm_curr/60)
            n11_curr = n_rpm_curr * D / np.sqrt(H_curr)
            Q11_curr = self.f_Q11(gamma_curr,n11_curr)
            Q_curr = Q11_curr*np.sqrt(H_curr)*D**2
            eta_curr = self.f_eta(gamma_curr,n11_curr) + d_eta_maj_pm
            Pm_curr = rho * g * Q_curr * H_curr * eta_curr
            C_m_curr = Pm_curr / om_curr
            return Q_curr,eta_curr,Pm_curr,C_m_curr
                                    
        def calc_Hcurr(Q_curr,Q_prev,H_0,H_prev):
            v_prev = Q_prev/A_duct
            v_curr = Q_curr/A_duct
            dH_curr = -csound/g * (v_curr-v_prev)
            H_curr = 2*H_0 + dH_curr - H_prev
            return H_curr
                            
        Q_0,eta_0,Pm_0,Cm_0 = calc_op(gamma_0,H_0,n_rpm_sync)
        
        transient.loc[0] = {'t': 0,'gamma': gamma_0,'H': H_0,'n_rpm': n_rpm_sync,'Q':  Q_0,
                                        'eta_P': eta_0,'P_m': Pm_0,'C_m': Cm_0,'C_r':0}


        for i in range(1,nstep):
            
            Q_prev = transient.loc[i-1,'Q']
            H_prev = transient.loc[i-1,'H']
            
            gamma_curr = gamma_0 * (1 - i*Tstep/Tstop)
            n_rpm_curr = n_rpm_sync
            
            def residuals(x):
                Q_curr,_,_,_ = calc_op(gamma_curr,x[0],n_rpm_curr)
                res_h = np.abs(x - calc_Hcurr(Q_curr,Q_prev,H_0,H_prev))
                return res_h
            
            hsol = minimize(residuals,H_prev,method=method,tol=tol)
            H_curr = hsol.x[0]
            Q_curr,eta_curr,Pm_curr,C_m_curr = calc_op(gamma_curr,H_curr,n_rpm_curr)
            
            transient.loc[i] = {'t': i*Tstep,'gamma': gamma_curr,'H': H_curr,'n_rpm': n_rpm_sync,'Q':  Q_curr,
                                            'eta_P': eta_curr,'P_m': Pm_curr,'C_m': C_m_curr,'C_r':C_m_curr}
            
        return transient
        
    def allievi_method_delestage(self,Tstop,H_0,gamma_0,n_rpm_sync,D,d_eta_maj_pm,I,L_duct,A_duct,rho,csound,g,
                                 method='SLSQP',tol=1.0e-6):
        import pandas as pd
        import numpy as np
        from scipy.optimize import minimize
        
        Tstep = 2*L_duct/csound
        nstep = int(Tstop / Tstep) + 2
        
        transient = pd.DataFrame(columns=['t','gamma','H','n_rpm','Q','eta_P','P_m','C_m','C_r'])
        
        def calc_op(gamma_curr,H_curr,n_rpm_curr):
            om_curr =  (2*np.pi * n_rpm_curr/60)
            n11_curr = n_rpm_curr * D / np.sqrt(H_curr)
            Q11_curr = self.f_Q11(gamma_curr,n11_curr)
            Q_curr = Q11_curr*np.sqrt(H_curr)*D**2
            eta_curr = self.f_eta(gamma_curr,n11_curr) + d_eta_maj_pm
            Pm_curr = rho * g * Q_curr * H_curr * eta_curr
            C_m_curr = Pm_curr / om_curr
            return Q_curr,eta_curr,Pm_curr,C_m_curr
                                    
        def calc_Hcurr(Q_curr,Q_prev,H_0,H_prev):
            v_prev = Q_prev/A_duct
            v_curr = Q_curr/A_duct
            dH_curr = -csound/g * (v_curr-v_prev)
            H_curr = 2*H_0 + dH_curr - H_prev
            return H_curr      
                  
        def calc_Ncurr(n_rpm_prev,C_m_prev,C_m_curr):
            # conservation du moment d'inertie
            # I d (om)/dt = Cm
            om_prev = (2*np.pi * n_rpm_prev/60)
            om_curr = om_prev + 0.5 * Tstep * (C_m_curr + C_m_prev) / I
            n_rpm_curr = om_curr * 60 / (2*np.pi) 
            return n_rpm_curr
            

                            
        Q_0,eta_0,Pm_0,Cm_0 = calc_op(gamma_0,H_0,n_rpm_sync)
        
        transient.loc[0] = {'t': 0,'gamma': gamma_0,'H': H_0,'n_rpm': n_rpm_sync,'Q':  Q_0,
                                        'eta_P': eta_0,'P_m': Pm_0,'C_m': Cm_0,'C_r':0}

        for i in range(1,nstep):
            
            Q_prev = transient.loc[i-1,'Q']
            H_prev = transient.loc[i-1,'H']
            n_rpm_prev = transient.loc[i-1,'n_rpm']
            C_m_prev = transient.loc[i-1,'C_m']
            
            gamma_curr = gamma_0 * (1 - i*Tstep/Tstop)
                        
            def residuals(x):
                Q_curr,_,_,C_m_curr = calc_op(gamma_curr,x[0],x[1])
                res = np.sqrt((x[0] - calc_Hcurr(Q_curr,Q_prev,H_0,H_prev))**2 
                              + (x[1] - calc_Ncurr(n_rpm_prev,C_m_prev,C_m_curr))**2 )
                return res
            
            if method in ['SLSQP','COBYLA']:
                def positive_constraint(x):
                    # y>0
                    _,_,P_m,_ = calc_op(gamma_curr,x[0],x[1])
                    return P_m
                
                cons=({'type':'ineq','fun': positive_constraint})

                Csol = minimize(residuals,[H_prev+5,n_rpm_prev+5],
                                constraints = cons, method=method,tol=tol)
            else:
                Csol = minimize(residuals,[H_prev+5,n_rpm_prev+5],method=method,tol=tol)
            H_curr = Csol.x[0]
            n_rpm_curr = Csol.x[1]
            Q_curr,eta_curr,Pm_curr,C_m_curr = calc_op(gamma_curr,H_curr,n_rpm_curr)
            
            transient.loc[i] = {'t': i*Tstep,'gamma': gamma_curr,'H': H_curr,'n_rpm': n_rpm_curr,'Q':  Q_curr,
                                            'eta_P': eta_curr,'P_m': Pm_curr,'C_m': C_m_curr,'C_r':0.}
            
        return transient
        
    def allievi_method_emballement(self,Tstop,H_0,gamma_0,n_rpm_sync,D,d_eta_maj_pm,I,L_duct,A_duct,rho,csound,g,
                                 method='SLSQP',tol=1.0e-6):
        import pandas as pd
        import numpy as np
        from scipy.optimize import minimize
        
        Tstep = 2*L_duct/csound
        nstep = int(Tstop / Tstep) + 30
        
        transient = pd.DataFrame(columns=['t','gamma','H','n_rpm','Q','eta_P','P_m','C_m','C_r'])
        
        def calc_op(gamma_curr,H_curr,n_rpm_curr):
            om_curr =  (2*np.pi * n_rpm_curr/60)
            n11_curr = n_rpm_curr * D / np.sqrt(H_curr)
            Q11_curr = self.f_Q11(gamma_curr,n11_curr)
            Q_curr = Q11_curr*np.sqrt(H_curr)*D**2
            eta_curr = self.f_eta(gamma_curr,n11_curr) + d_eta_maj_pm
            Pm_curr = rho * g * Q_curr * H_curr * eta_curr
            C_m_curr = Pm_curr / om_curr
            return Q_curr,eta_curr,Pm_curr,C_m_curr
                                    
        def calc_Hcurr(Q_curr,Q_prev,H_0,H_prev):
            v_prev = Q_prev/A_duct
            v_curr = Q_curr/A_duct
            dH_curr = -csound/g * (v_curr-v_prev)
            H_curr = 2*H_0 + dH_curr - H_prev
            return H_curr      
                  
        def calc_Ncurr(n_rpm_prev,C_m_prev,C_m_curr,C_r_prev,C_r_curr):
            # conservation du moment d'inertie
            # I d (om)/dt = Cm
            om_prev = (2*np.pi * n_rpm_prev/60)
            om_curr = om_prev + 0.5 * Tstep * (C_m_curr-C_r_curr + C_m_prev-C_r_prev) / I
            n_rpm_curr = om_curr * 60 / (2*np.pi) 
            return n_rpm_curr
            
        Q_0,eta_0,Pm_0,Cm_0 = calc_op(gamma_0,H_0,n_rpm_sync)
        
        transient.loc[0] = {'t': 0,'gamma': gamma_0,'H': H_0,'n_rpm': n_rpm_sync,'Q':  Q_0,
                                        'eta_P': eta_0,'P_m': Pm_0,'C_m': Cm_0,'C_r':0}

        for i in range(1,nstep):
            
            Q_prev = transient.loc[i-1,'Q']
            H_prev = transient.loc[i-1,'H']
            n_rpm_prev = transient.loc[i-1,'n_rpm']
            C_m_prev = transient.loc[i-1,'C_m']
            C_r_prev = transient.loc[i-1,'C_r']
            gamma_curr = gamma_0
            
            C_r_curr = C_r_prev * (1 - i*Tstep/Tstop)
                        
            def residuals(x):
                Q_curr,_,_,C_m_curr = calc_op(gamma_curr,x[0],x[1])
                res = np.sqrt((x[0] - calc_Hcurr(Q_curr,Q_prev,H_0,H_prev))**2 
                              + (x[1] - calc_Ncurr(n_rpm_prev,C_m_prev,C_m_curr,C_r_prev,C_r_curr))**2 )
                return res
            
            if method in ['SLSQP','COBYLA']:
                def positive_constraint(x):
                    # y>0
                    _,_,P_m,_ = calc_op(gamma_curr,x[0],x[1])
                    return P_m
                
                cons=({'type':'ineq','fun': positive_constraint})

                Csol = minimize(residuals,[H_prev+5,n_rpm_prev+5],
                                constraints = cons, method=method,tol=tol)
            else:
                Csol = minimize(residuals,[H_prev+5,n_rpm_prev+5],method=method,tol=tol)
            H_curr = Csol.x[0]
            n_rpm_curr = Csol.x[1]
            Q_curr,eta_curr,Pm_curr,C_m_curr = calc_op(gamma_curr,H_curr,n_rpm_curr)
            
            transient.loc[i] = {'t': i*Tstep,'gamma': gamma_curr,'H': H_curr,'n_rpm': n_rpm_curr,'Q':  Q_curr,
                                            'eta_P': eta_curr,'P_m': Pm_curr,'C_m': C_m_curr,'C_r':0.}
            
        return transient
        
def plot_transient(tr_hist,title_txt):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    
    # quelques parametres pour l'affichage des figures
    plt.rcParams['font.size']  = 14
    plt.rcParams['figure.dpi'] = 150
    
    fig,ax = plt.subplots(3,2, figsize=(15,10), sharex=True)
    plt.subplots_adjust(hspace=0.02,wspace = 0.3,left = 0.17,right = 0.97,bottom = 0.1,top = 0.93)

    fig.suptitle(title_txt)

    ax[0,0].plot(tr_hist['t'],tr_hist['gamma'],linestyle='--',marker='o')
    ax[0,0].set_ylabel(r'$\gamma$ [deg]');
    ax[0,0].grid()

    ax[1,0].plot(tr_hist['t'],tr_hist['Q'],linestyle='--',marker='o')
    ax[1,0].set_ylabel(r'$Q$ [m3/s]');
    ax[1,0].grid()

    ax[2,0].plot(tr_hist['t'],tr_hist['H'],linestyle='--',marker='o')
    ax[2,0].set_ylabel(r'$H$ [m]');
    ax[2,0].set_xlabel(r'$t$ [s]');
    ax[2,0].grid()

    ax[0,1].plot(tr_hist['t'],tr_hist['n_rpm'],linestyle='--',marker='o')
    ax[0,1].set_ylabel(r'$N$ [RPM]');
    ax[0,1].grid()

    ax[1,1].plot(tr_hist['t'],tr_hist['eta_P'],linestyle='--',marker='o')
    ax[1,1].set_ylabel(r'$\eta_P$ [%]');
    ax[1,1].yaxis.set_major_formatter(tck.PercentFormatter(xmax=1.0))
    ax[1,1].grid()

    ax[2,1].plot(tr_hist['t'],tr_hist['P_m']/1.0e6,linestyle='--',marker='o')
    ax[2,1].set_ylabel(r'$P_{m,P}$ [MW]');
    ax[2,1].set_xlabel(r'$t$ [s]');
    ax[2,1].grid()
