from astropy import constants as const, units as u
import math
import numpy as np
from scipy.special import kn as K, iv as I
from uncertainties import unumpy as unp, ufloat as uf


class TotalHalo():
    
    def __init__(self,M_b,delta_M_b,M_d,delta_M_d,a_b,delta_a_b,
                 R_d,delta_R_d,M_hi,delta_M_hi,M_h=None,delta_M_h=None,
                 R_h=None,delta_R_h=None,scale=1.5,scale_error=0.2,
                 fix_halo=False):
        
        self.M_b = M_b.to(u.Msun)
        self.delta_M_b = delta_M_b.to(u.Msun)
        self.M_d = M_d.to(u.Msun)
        self.delta_M_d = delta_M_d.to(u.Msun)
        self.a_b = a_b.to(u.kpc)
        self.delta_a_b = delta_a_b.to(u.kpc)
        self.R_d = R_d.to(u.kpc)
        self.delta_R_d = delta_R_d.to(u.kpc)
        self.M_hi = M_hi.to(u.Msun)
        self.delta_M_hi = delta_M_hi.to(u.Msun)
        
        self.M_h = M_h.to(u.Msun) if M_h is not None else None
        self.delta_M_h = (delta_M_h.to(u.Msun) if delta_M_h is not None 
                          else None)
        self.R_h = R_h.to(u.kpc) if R_h is not None else None
        self.delta_R_h = (delta_R_h.to(u.kpc) if delta_R_h is not None
                          else None)
        
        self.scaler = uf(scale,scale_error)
        
        self.M_bulge = unp.uarray(self.M_b.value,self.delta_M_b.value)
        self.M_disc = unp.uarray(self.M_d.value,self.delta_M_d.value)
        self.R_bulge = (unp.uarray(self.a_b.value,self.delta_a_b.value)
                       / self.scaler)
        self.R_disc = (unp.uarray(self.R_d.value,self.delta_R_d.value)
                      / self.scaler)
        self.M_hi = unp.uarray(self.M_hi.value,self.delta_M_hi.value)
        
        self.fix_halo = fix_halo
        
        H0 = 70 * u.km/u.s/u.Mpc
        rho_crit = (3 * H0**2) / (8*math.pi*const.G)
        self.rho_crit = (rho_crit.to(u.kg/u.m**3))
        
    def nda_to_unp(self,data):
        return unp.uarray(data.data,data.uncertainty.array)
      
    def unp_to_nda(self,data,unit=None):
        datas = [data[i].nominal_value for i in range(len(data))]
        uncertainties = [data[i].std_dev for i in range(len(data))]
        return nda(datas,sdu(uncertainties),unit=unit) 
    
    # --- halo terms, from Dutton+10, Dutton+Maccio 14 ---
    def M_halo(self,alpha=-0.5,alpha_upper=-0.475,alpha_lower=-0.575,
               beta=0,logx0=10.4,gamma=1,
               logy0=1.61,logy0_lower=1.49,logy0_upper=1.75):
        
        exists = ('halo_mass' in self.__dict__)
        if exists is False:
            if self.M_h == None:
                stellar_mass = self.M_bulge + self.M_disc
        
                alpha = uf(alpha,np.max([alpha_upper-alpha,
                                         alpha-alpha_lower]))
                beta = uf(beta,0)
                x0 = uf(10**logx0,0)
                gamma = uf(gamma,0)
                y0 = uf(10**logy0,np.max([10**logy0_upper-10**logy0,
                                          10**logy0-10**logy0_lower]))
    
                a = y0 * (stellar_mass/x0)**alpha
                b = 1/2 + 1/2*((stellar_mass/x0)**gamma)
                c = (beta-alpha)/gamma
                y = a * (b**c)
                self.halo_mass = y * stellar_mass
            else:
                self.halo_mass = unp.uarray(self.M_h.value,
                                            self.delta_M_h.value)
            
        return self.halo_mass
    
    def R_halo(self):
        exists = 'R200' in self.__dict__
        exists = False
        # unit converter---
        cf = (1*u.Msun)**(1/3) * (1*u.kg)**(-1/3) * (1*u.m)
        cf = cf.to(u.kpc).value
        #-------------------
        if exists is False:
            if self.R_h == None:
                h = 0.7
                K = 3 / (4*np.pi*200*self.rho_crit.value)
                M200 = self.M_halo()
                logc200 = (0.905 
                         - 0.101 * unp.log10(M200/(10**12 * h**(-1))))
            
                c200 = 10**logc200
                self.R200 = (K*(M200))**(1/3) * cf
                self.a_h = self.R200 / c200
            else:
                self.a_h = unp.uarray(self.R_h.value,self.delta_R_h.value)
                self.R200 = 10 * self.a_h

        return self.a_h, self.R200

    def M_halo_hernquist(self):
        # use a modified halo mass so that the halo follows the NFW profile 
        # closely.
        exists = 'hernquist_halo_mass' in self.__dict__
        if exists is False:
            if self.fix_halo is False:
                a_h = self.R_halo()[0]
                rho0 = self.burkert_rho0()
                self.hernquist_halo_mass = 4 * math.pi * a_h**3 * rho0
            else:
                self.hernquist_halo_mass = self.M_halo()
        return self.hernquist_halo_mass
    
    def burkert_rho0(self):
        exists = 'rho0' in self.__dict__
        if exists is False:
            a_h, r200 = self.R_halo()
            M_h = self.M_halo()
      
            self.rho0 = (M_h/((2*np.pi*a_h**3) * 
                         (unp.log((r200+a_h)/a_h) 
                        + 0.5 * unp.log((r200**2+a_h**2)/a_h**2) 
                        - unp.arctan(r200/a_h))))
        
        return self.rho0
    
    # --- Disc scale lengths: including the HI + stellar component ---
    def hi_scalelength(self,m=0.86,m_error=0.04,c=0.79,c_error=0.02,
                        k=0.19,k_error=0.03): # Wang+14, Lelli+16
        exists = 'R_hi' in self.__dict__
        if exists is False:
            m_param = uf(m,m_error)
            c_param = uf(c,c_error)
            k_param = uf(k,k_error)

            r_d = self.R_disc
            log_r_d = unp.log10(r_d)
            logr_hi = m_param * log_r_d + c_param
            r_hi = 10**logr_hi
            self.R_hi = k_param * r_hi
        return self.R_hi
    
    def baryonic_scalelength(self): 
        exists = 'R_disc_hi' in self.__dict__
        if exists is False:
            r_stars = self.R_disc
            r_hi = self.hi_scalelength()
        
            rho_stars = (self.M_disc / (2*math.pi*r_stars**2))
            rho_hi = (self.M_hi / (2*math.pi*r_hi**2))
            rho_tot = rho_stars + rho_hi
        
            self.R_disc_hi = ((self.M_disc*r_stars + self.M_hi*r_hi) 
                                  / (self.M_disc + self.M_hi))

        return self.R_disc_hi
    
    # --- Masses w/i 2.2 scale disc scalelengths ---
    def disc_mass_22(self):
        self.M_disc_22 = 0.645 * (self.M_disc + self.M_hi)
        return self.M_disc_22
      
    def bulge_mass_22(self):
        M_b = self.M_bulge
        a_b = self.R_bulge
        R22 = 2.2 * self.baryonic_scalelength()
        self.M_bulge_22 = M_b * (R22**2 / (R22 + a_b)**2)
        return self.M_bulge_22
      
    def hernquist_mass_22(self):
        a_h = self.R_halo()[0]
        M_h = self.M_halo_hernquist()
        R22 = 2.2 * self.baryonic_scalelength()
        self.M_hernquist_22 = M_h * (R22**2 / (R22 + a_h)**2)
        return self.M_hernquist_22
    
    def burkert_mass_22(self):
        a_h = self.R_halo()[0]
        rho0 = self.burkert_rho0()
        R22 = 2.2 * self.baryonic_scalelength()
        a = 2 * math.pi * rho0 * a_h**3
        b = (unp.log((R22+a_h)/a_h)
            + 0.5 * unp.log((R22**2 + a_h **2)/a_h**2)
            - unp.arctan(R22/a_h))
        self.M_burkert_22 = a * b
        return self.M_burkert_22
    
    # --- Spiral arm numbers ---
    def bulge_arm_number(self,y=1,X=1.5):
        M_b = self.M_bulge
        a_b = self.R_bulge
        M_d = self.M_disc + self.M_hi
        R_d = self.baryonic_scalelength()
        
        a = unp.exp(2*y)/X
        b = (M_b/M_d) * ((2*y + (3*a_b/R_d)) / (2*y + (a_b/R_d))**3)
        self.m_bulge = a * b
        return self.m_bulge
      
    def disc_arm_number(self,y=1,X=1.5):
        R_d = self.baryonic_scalelength()
            
        a = unp.exp(2*y)/X
        d = (y**2/2) * ((3*(I(1,y)*K(0,y)) - 3*(I(0,y)*K(1,y)) 
                         + (I(1,y)*K(2,y)) - (I(2,y)*K(1,y))))
        e = (4*y) * (I(0,y)*K(0,y) - (I(1,y)*K(1,y)))
        
        self.m_disc = np.array(a * (d + e))
        return self.m_disc
      
    def hernquist_arm_number(self,y=1,X=1.5):
        M_d = self.M_disc + self.M_hi
        R_d = self.baryonic_scalelength()
        M_h = self.M_halo_hernquist()
        a_h = self.R_halo()[0]
        
        a = unp.exp(2*y)/X
        c = (M_h/M_d) * ((2*y + (3*a_h/R_d)) / (2*y + (a_h/R_d))**3)
        self.m_hernquist = a * c
        return self.m_hernquist
      
    def burkert_arm_number(self,y=1,X=1.5):
        M_d = self.M_disc + self.M_hi
        R_d = self.baryonic_scalelength()
        rho0 = self.burkert_rho0()
        a_h = self.R_halo()[0]
        a = unp.exp(2*y)/X
        R = y * 2 * R_d
        c = (2*np.pi*rho0*R**3/M_d) * (1/(4*y**2)) * ((R**2/(R**2+a_h**2))
                                                 + (R/(a_h*(R**2/a_h**2 + 1)))
                                                 + unp.log(1+R/a_h)
                                                 + unp.log(1+(R/a_h)**2)
                                                 - unp.arctan(R/a_h))

        self.m_burkert = a * c
        return self.m_burkert

    # --- Pitch angles ---
    # kappa2 = r(domega2/dr) + 4omega2
    def total_shear(self,y=1,halo='hernquist'):
        r = 2 * y * self.baryonic_scalelength()
        
        a_b = self.R_bulge
        M_b = self.M_bulge
        kappa2_b = M_b * (4/(r*(r+a_b)**2) - (3*r + a_b)/(r*(r+a_b)**3))
        omega2_b = M_b * (1/(r*(r+a_b)**2))
        
        R_d = self.R_disc
        M_d = self.M_disc
        C = M_d / (2 * R_d)
        kappa2_d = C * ((r/(4*R_d) * (3*I(1,y)*K(0,y) - 3*I(0,y)*K(1,y) 
                                      + I(1,y)*K(2,y) - I(2,y)*K(1,y)))
                        + 4*(I(0,y)*K(0,y) - I(1,y)*K(1,y)))

        omega2_d = C * (I(0,y)*K(0,y) - I(1,y)*K(1,y))
        
        if halo == 'hernquist':
            a_h = self.R_halo()[0]
            M_h = self.M_halo_hernquist()
            kappa2_h = M_h * (4/(r*(r+a_h)**2) - (3*r + a_h)/(r*(r+a_h)**3))
            omega2_h = M_h * (1/(r*(r+a_h)**2))
        
        else:
            a_h = self.R_halo()[0]
            rho0 = self.burkert_rho0()
            C = 2* math.pi * rho0 * a_h**3
            kappa2_h = (C/r**3) * (r**2/(r**2 + a_h**2) 
                                - r/(a_h*((r/a_h)**2 + 1)) + r/(r+a_h)
                                + unp.log((r+a_h)/a_h) 
                                + (1/2)*unp.log((r**2+a_h**2)/a_h**2)
                                - unp.arctan(r/a_h))
            omega2_h = (C/r**3) * (unp.log((r+a_h)/a_h) 
                                + (1/2)*unp.log((r**2+a_h**2)/a_h**2)
                                - unp.arctan(r/a_h))
        
        omega2 = omega2_b + omega2_d + omega2_h
        kappa2 = kappa2_b + kappa2_d + kappa2_h
        
        self.Gamma = 2 - kappa2/(2 * omega2)
        return self.Gamma
      
    def pitch_angle(self,y=1,halo='hernquist',
                    relation='michikoshi'):
        Gamma = self.total_shear(y,halo)
        #tanpsi = 1.932 - 5.186*(0.5*Gamma) + 4.704*(0.5*Gamma)**2
        if relation == 'michikoshi':
            tanpsi = 2/7 * unp.sqrt(4-2*Gamma)/Gamma
            self.psi = unp.arctan(tanpsi) * 360/(2*math.pi)
        elif relation == 'fuchs':
            tanpsi = 1.932 - 5.186*(0.5*Gamma) + 4.704*(0.5*Gamma)**2
            self.psi = unp.arctan(tanpsi) * 360/(2*math.pi)
        elif relation == 'seigar':
            m = uf(-36.62,2.77)
            c = uf(64.25,2.87)
            self.psi = m * Gamma + c
        
        return self.psi