import astropy.constants as const
from astropy.nddata import NDDataArray as nda, StdDevUncertainty as sdu
from astropy.table import Table, column
import astropy.units as u
import math
import numpy as np
from scipy.special import kn as K, iv as I
from uncertainties import ufloat as uf, unumpy as unp

class TotalHalo():
    
    def __init__(self,M_b,delta_M_b,M_d,delta_M_d,a_b,delta_a_b,
                 R_d,delta_R_d,M_hi,delta_M_hi,
                 M_halo=None,delta_M_halo=None,
                 R_halo=None,delta_R_halo=None,scale=1.5,scale_error=0.2):
        
        self.M_b = M_b.to(u.Msun)
        self.delta_M_b = delta_M_b.to(u.Msun)
        self.M_d = M_d.to(u.Msun)
        self.delta_M_d = delta_M_d.to(u.Msun)
        self.a_b = a_b.to(u.kpc)
        self.delta_a_b = delta_a_b.to(u.kpc)
        self.R_d = R_d.to(u.kpc)
        self.delta_R_d = delta_R_d.to(u.kpc)
        self.M_hi = M_hi.to(u.Msun) if M_hi is not None else None
        self.delta_M_hi = delta_M_hi.to(u.Msun) if delta_M_hi is not None else None
        self.M_halo = M_halo
        self.delta_M_halo= delta_M_halo
        self.R_halo = R_halo
        self.delta_R_halo = delta_R_halo
        self.scale = scale
        self.scale_error = scale_error
        
        H0 = 70 * u.km/u.s/u.Mpc
        rho_crit = (3 * H0**2) / (8*math.pi*const.G)
        self.rho_crit = rho_crit.to(u.kg/u.m**3)
        
    def nda_to_unp(self,data):
        return unp.uarray(data.data,data.uncertainty.array)
      
    def unp_to_nda(self,data,unit=None):
        datas = [data[i].nominal_value for i in range(len(data))]
        uncertainties = [data[i].std_dev for i in range(len(data))]
        return nda(datas,sdu(uncertainties),unit=unit)                
        
    def bulge_mass(self):
        exists = 'bulge_mass' in self.__dict__
        if exists is False:
            self.bulge_mass_ = nda(self.M_b,sdu(self.delta_M_b),unit=u.Msun)
        return self.bulge_mass_
    
    def disc_mass(self,hi=True):
        M_disc = nda(self.M_d,sdu(self.delta_M_d),unit=u.Msun)
        if (hi is False) | (self.M_hi is None):
            return M_disc
        else:
            M_hi = nda(self.M_hi,sdu(self.delta_M_hi),unit=u.Msun)
            M_tot = M_disc.add(M_hi)
            return M_tot
    
    def stellar_mass(self,hi=True):
        return self.disc_mass(hi).add(self.bulge_mass())
        
    def halo_mass(self,alpha=-0.5,alpha_upper=-0.475,alpha_lower=-0.575,
                  beta=0,logx0=10.4,gamma=1,
                  logy0=1.61,logy0_lower=1.49,logy0_upper=1.75):
        
        exists = 'halo_mass' in self.__dict__
        if exists is False:
            if self.M_halo is not None:
                self.halo_mass_ = nda(self.M_halo,sdu(self.delta_M_halo),
                                      unit=u.Msun)
            else:
                stellar_mass = unp.uarray(self.stellar_mass(hi=False).data,
                                self.stellar_mass(hi=False).uncertainty.array)
        
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
                y = a * b**c
                halo_mass_ = y * stellar_mass
                self.halo_mass_ = self.unp_to_nda(halo_mass_,u.Msun)
            
        return self.halo_mass_
    
    # --- Now for the sizes ---
    def bulge_scale_length(self):
        exists = 'bulge_scale_length_' in self.__dict__
        if exists is False:
            scale_factor = nda(self.scale,sdu(self.scale_error))
            r_b = nda(self.a_b,sdu(self.delta_a_b),unit=u.kpc)
            self.bulge_scale_length_ = r_b.divide(scale_factor)
        return self.bulge_scale_length_
        
    def halo_scale_length(self):
        exists = self.R_halo is not None
        if exists is True:
            self.R_halo_ = nda(self.R_halo,sdu(self.delta_R_halo))
            self.R200_ = self.R_halo_.multiply(10)
        else:
            h = 0.7
            K = 3 / (4*np.pi*200*self.rho_crit)
            M200 = self.halo_mass()
            M200_value = M200.data * (u.Msun)
            M200_error = M200.uncertainty.array * (u.Msun)
            logc200 = (0.905 
                   - 0.101 * np.log10(M200_value/(10**12 * h**(-1) * u.Msun)))
            c200 = 10**logc200
            r200 = (K*(M200_value))**(1/3)
            r200 = r200.to(u.kpc)
            r200_error = (1/3) * r200 * M200_error/M200_value
            self.R200_ = nda(r200,sdu(r200_error),unit=u.kpc)
            self.R_halo_ = self.R200_.divide(c200)
        return self.R_halo_, self.R200_
    
    def hi_scale_length(self,m=0.86,m_error=0.04,c=0.79,c_error=0.02,
                        k=0.19,k_error=0.03): # Wang+14, Lelli+16
        exists = 'hi_scale_length_' in self.__dict__
        if exists is False:
            m_param = uf(m,m_error)
            c_param = uf(c,c_error)
            k_param = uf(k,k_error)

            r_d = self.nda_to_unp(self.stellar_disc_scale_length())
            log_r_d = unp.log10(r_d)
            logr_hi = m_param * log_r_d + c_param
            r_hi = 10**logr_hi
            r_s = k_param * r_hi
            self.hi_scale_length_ = self.unp_to_nda(r_s,u.kpc)
        return self.hi_scale_length_
    
    def stellar_disc_scale_length(self):
        exists = 'stellar_disc_scale_length_' in self.__dict__
        if exists is False:
            scale_factor = nda(self.scale,sdu(self.scale_error))
            r_d = nda(self.R_d,sdu(self.delta_R_d),unit=u.kpc)
            self.stellar_disc_scale_length_ = r_d.divide(scale_factor)
        return self.stellar_disc_scale_length_
    
    def disc_scale_length(self): 
        exists = 'disc_scale_length_' in self.__dict__
        if exists is False:
            if self.M_hi is None:
                self.disc_scale_length_ = self.stellar_disc_scale_length()
            else:
                r_d = self.nda_to_unp(self.stellar_disc_scale_length())
                r_g = self.nda_to_unp(self.hi_scale_length())
        
                rho_d = (unp.uarray(self.M_hi.value,self.delta_M_hi.value)/
                         (2*math.pi*r_d**2))
                rho_g = (self.nda_to_unp(self.disc_mass(hi=False))
                      / (2*math.pi*r_g**2))
        
                r_s = 1 / (unp.log((rho_d/(rho_g+rho_d)*unp.exp(1/r_g)) 
                                 + (rho_g/(rho_g+rho_d)*unp.exp(1/r_d))))
                self.disc_scale_length_ = self.unp_to_nda(r_s,unit=u.kpc)
        return self.disc_scale_length_
    
    def m_predicted(self,y=np.linspace(0.25,2.5,100),R=None,X=1.5,
                    halo='hernquist'):
        
        M_B = self.nda_to_unp(self.bulge_mass())
        M_D = self.nda_to_unp(self.disc_mass())
        M_H = self.nda_to_unp(self.halo_mass())
        a_b = self.nda_to_unp(self.bulge_scale_length())
        R_d = self.nda_to_unp(self.disc_scale_length())
        a_h = self.nda_to_unp(self.halo_scale_length()[0])
        
        if R is not None:
            y = (R/(2*R_d)).value
        else:
            R = y * 2 * R_d
        a = unp.exp(2*y)/X
        b = (M_B/M_D) * ((2*y + (3*a_b/R_d)) / (2*y + (a_b/R_d))**3)
        if halo is 'hernquist':
            c = (M_H/M_D) * ((2*y + (3*a_h/R_d)) / (2*y + (a_h/R_d))**3)
        else:
            rho0 = self.burkert_rho0()
            c = (2*np.pi*rho0*R**3/M_D) * (1/(4*y**2)) * ((R**2/(R**2+a_h**2))
                                                 + (R/(a_h*(R**2/a_h**2 + 1)))
                                                 + unp.log(1+R/a_h)
                                                 + unp.log(1+(R/a_h)**2)
                                                 - unp.arctan(R/a_h))
            
        d = (y**2/2) * ((3*(I(1,y)*K(0,y)) - 3*(I(0,y)*K(1,y)) 
                         + (I(1,y)*K(2,y)) - (I(2,y)*K(1,y))))
        e = (4*y) * (I(0,y)*K(0,y) - (I(1,y)*K(1,y)))
        m = a*(b + c + d + e)
        
        m_table = Table()
        m_table['m'] = [m[i].nominal_value for i in range(len(m))]
        m_table['error'] = [m[i].std_dev for i in range(len(m))]
        m_table['R'] = [R[i].nominal_value for i in range(len(R))]
        m_table['y'] = y
        return m_table
      
    def burkert_rho0(self):
        exists = 'rho0' in self.__dict__
        if exists is False:
            a_h = self.nda_to_unp(self.halo_scale_length()[0])
            M_H = self.nda_to_unp(self.halo_mass())
            r200 = self.nda_to_unp(self.halo_scale_length()[1])
      
            self.rho0 = M_H/((2*np.pi*a_h**3) * (unp.log((r200+a_h)/a_h) 
                                      + 0.5 * unp.log((r200**2+a_h**2)/a_h**2) 
                                      - unp.arctan(r200/a_h)))
        return self.rho0
      
    def M_disc_22(self,hi=False):
        M_disc = self.disc_mass(hi)
        self.disc_mass_22_ = M_disc.multiply(0.645)
        return self.disc_mass_22_
      
    def M_bulge_22(self):
        exists = 'bulge_mass_22_' in self.__dict__
        if exists is False:
            R22 = self.nda_to_unp(self.disc_scale_length()) * 2.2
            a_b = self.nda_to_unp(self.bulge_scale_length())
            M_b = self.nda_to_unp(self.bulge_mass())
            M_22 = M_b * (R22**2 / (R22 + a_b)**2)
            self.bulge_mass_22_ = self.unp_to_nda(M_22,u.Msun)
        return self.bulge_mass_22_
      
    def M_halo_22_hernquist(self):
        exists = 'halo_mass_22_hernquist_' in self.__dict__
        if exists is False:
            R22 = self.nda_to_unp(self.disc_scale_length()) * 2.2
            a_h = self.nda_to_unp(self.halo_scale_length()[0])
            M_h = self.nda_to_unp(self.halo_mass())

            M_22 = M_h * (R22**2 / (R22 + a_h)**2)
            self.halo_mass_22_hernquist_ = self.unp_to_nda(M_22,u.Msun)
        return self.halo_mass_22_hernquist_
        
    def M_halo_22_burkert(self):
        exists = 'halo_mass_22_burkert_' in self.__dict__
        if exists is False:
            R22 = self.nda_to_unp(self.disc_scale_length()) * 2.2
            a_h = self.nda_to_unp(self.halo_scale_length()[0])
            rho0 = self.burkert_rho0()
            a = 2 * math.pi * rho0 * a_h**3
            b = (unp.log((R22+a_h)/a_h)
               + 0.5 * unp.log((R22**2 + a_h **2)/a_h**2)
               - unp.arctan(R22/a_h))
            M_22 = a * b
            self.halo_mass_22_burkert_ = self.unp_to_nda(M_22)
        return self.halo_mass_22_burkert_
      
    def kappa_omega_disc(self,y=1.1,R=None):
        exists = 'disc_kappa2_' in self.__dict__
        if exists is False:
            R_d = self.nda_to_unp(self.disc_scale_length())
            M_d = self.nda_to_unp(self.disc_mass(True))
            if R is not None:
                y = (R/(2*R_d)).value
            else:
                R = y * 2 * R_d
                
            C = M_d / (2 * R_d)
        
            kappa2 = C * ((R/(4*R_d) * (3*I(1,y)*K(0,y) - 3*I(0,y)*K(1,y) 
                                      + I(1,y)*K(2,y) - I(2,y)*K(1,y)))
                        + 4*(I(0,y)*K(0,y) - I(1,y)*K(1,y)))

            omega2 = C * (I(0,y)*K(0,y) - I(1,y)*K(1,y))
            self.disc_kappa2_ = self.unp_to_nda(kappa2)
            self.disc_omega2_ = self.unp_to_nda(omega2)
        return self.disc_kappa2_, self.disc_omega2_

    def kappa_omega_bulge(self,y=1.1,R=None):
        exists = 'bulge_kappa2_' in self.__dict__
        if exists is False:
            a_b = self.nda_to_unp(self.bulge_scale_length())
            R_d = self.nda_to_unp(self.disc_scale_length())
            if R is not None:
                y = (R/(2*R_d)).value
            else:
                R = y * 2 * R_d
            
            C = self.nda_to_unp(self.bulge_mass())
            kappa2 = C * (4/(R*(R+a_b)**2) - (3*R + a_b)/(R*(R+a_b)**3))
            omega2 = C * (1/(R*(R+a_b)**2))
            self.bulge_kappa2_ = self.unp_to_nda(kappa2)
            self.bulge_omega2_ = self.unp_to_nda(omega2)
        return self.bulge_kappa2_, self.bulge_omega2_
      
    def kappa_omega_halo_hernquist(self,y=1.1,R=None):#y=np.linspace(0.25,2.5,100),):
        exists = 'hernquist_halo_kappa2_' in self.__dict__
        if exists is False:
            a_h = self.nda_to_unp(self.halo_scale_length()[0])
            R_d = self.nda_to_unp(self.disc_scale_length())
            if R is not None:
                y = (R/(2*R_d)).value
            else:
                R = y * 2 * R_d
                
            C = self.nda_to_unp(self.halo_mass())
            kappa2 = C * (4/(R*(R+a_h)**2) - (3*R + a_h)/(R*(R+a_h)**3))
            omega2 = C * (1/(R*(R+a_h)**2))
            self.hernquist_halo_kappa2_ = self.unp_to_nda(kappa2)
            self.hernquist_halo_omega2_ = self.unp_to_nda(omega2)
        return self.hernquist_halo_kappa2_, self.hernquist_halo_omega2_
      
    def kappa_omega_halo_burkert(self,y=1.1,R=None):#=np.linspace(0.25,2.5,100),):
        exists = 'burkert_halo_kappa2_' in self.__dict__
        if exists is False:
            a_h = self.nda_to_unp(self.halo_scale_length()[0])
            R_d = self.nda_to_unp(self.disc_scale_length())
            if R is not None:
                y = (R/(2*R_d)).value
            else:
                R = y * 2 * R_d
            rho0 = self.burkert_rho0()    
            C = 2* math.pi * rho0 * a_h**3
            
            kappa2 = (C/R**3) * (R**2/(R**2 + a_h**2) 
                               - R/(a_h*((R/a_h)**2 + 1)) + R/(R+a_h)
                               + unp.log((R+a_h)/a_h) 
                               + (1/2)*unp.log((R**2+a_h**2)/a_h**2)
                               - unp.arctan(R/a_h))
            omega2 = (C/R**3) * (unp.log((R+a_h)/a_h) 
                              + (1/2)*unp.log((R**2+a_h**2)/a_h**2)
                              - unp.arctan(R/a_h))
    
            self.burkert_halo_kappa2_ = self.unp_to_nda(kappa2)
            self.burkert_halo_omega2_ = self.unp_to_nda(omega2)

        return self.burkert_halo_kappa2_, self.burkert_halo_omega2_
     
    def total_shear(self,halo='hernquist'):
        R22 = unp.uarray(self.R_d.data, self.delta_R_d.data) * 2.2
        kappa2_disc, omega2_disc = (
                                  self.nda_to_unp(self.kappa_omega_disc()[0]),
                                  self.nda_to_unp(self.kappa_omega_disc()[1]))

        kappa2_bulge, omega2_bulge = (
                                 self.nda_to_unp(self.kappa_omega_bulge()[0]),
                                 self.nda_to_unp(self.kappa_omega_bulge()[1]))

        if halo is 'hernquist':
            kappa2_halo, omega2_halo = (
                        self.nda_to_unp(self.kappa_omega_halo_hernquist()[0]),
                        self.nda_to_unp(self.kappa_omega_halo_hernquist()[1]))
        else:
            kappa2_halo, omega2_halo = (
                        self.nda_to_unp(self.kappa_omega_halo_burkert()[0]),
                        self.nda_to_unp(self.kappa_omega_halo_burkert()[1]))
    
        kappa2 = kappa2_disc + kappa2_bulge + kappa2_halo
        omega2 = omega2_disc + omega2_bulge + omega2_halo
        Gamma = 2 - (kappa2/(2*omega2))
        self.Gamma_ = self.unp_to_nda(Gamma)
        return self.Gamma_

      
    def psi_predicted_michikoshi(self,halo='hernquist'):
        Gamma = self.nda_to_unp(self.total_shear(halo))
        tanpsi = 2/7 * unp.sqrt(4-2*Gamma)/Gamma
        #tanpsi = 1.932 - 5.186*(0.5*Gamma) + 4.704*(0.5*Gamma)**2
        psi = unp.arctan(tanpsi) * 360/(2*math.pi)
        self.psi_michikoshi_ = self.unp_to_nda(psi,u.degree)
        return self.psi_michikoshi_
      
    def psi_predicted_seigar(self,halo='hernquist'):
        Gamma = self.nda_to_unp(self.total_shear(halo))
        m = unp.uarray(-36.62,2.77)
        c = unp.uarray(64.25,2.87)
        psi = m * Gamma + c
        self.psi_seigar_ = self.unp_to_nda(psi,u.degree)
        return self.psi_seigar_
        
def get_halo_table(halos):

    halo_table = Table()

    halo_table['M_disc_stars'] = halos.disc_mass(hi=False).data
    halo_table['delta_M_disc_stars'] = (
                                  halos.disc_mass(hi=False).uncertainty.array)

    halo_table['M_disc_total'] = halos.disc_mass(hi=True).data
    halo_table['delta_M_disc_total'] = (
                                   halos.disc_mass(hi=True).uncertainty.array)

    halo_table['R_disc_stars'] = halos.stellar_disc_scale_length().data
    halo_table['delta_R_disc_stars'] = (
                          halos.stellar_disc_scale_length().uncertainty.array)

    halo_table['R_disc_total'] = halos.disc_scale_length().data
    halo_table['delta_R_disc_total'] = (
                                  halos.disc_scale_length().uncertainty.array)

    halo_table['M_bulge'] = halos.bulge_mass().data
    halo_table['delta_M_bulge'] = halos.bulge_mass().uncertainty.array

    halo_table['R_bulge'] = halos.bulge_scale_length().data
    halo_table['delta_R_bulge'] = halos.bulge_scale_length().uncertainty.array

    halo_table['M_halo'] = halos.halo_mass().data
    halo_table['delta_M_halo'] = halos.halo_mass().uncertainty.array

    halo_table['R_halo'] = halos.halo_scale_length()[0].data
    halo_table['delta_R_halo'] = (
                               halos.halo_scale_length()[0].uncertainty.array)

    r_d = [2,2.2]
    for r in r_d:
        m_hernquist_pred = halos.m_predicted(y=r/2,halo='hernquist')
        m_burkert_pred = halos.m_predicted(y=r/2,halo='burkert')
        halo_table['m_hernquist_{}R_d'.format(r)] = m_hernquist_pred['m']
        halo_table['delta_m_hernquist_{}R_d'.format(r)] = m_hernquist_pred['error']
        halo_table['m_burkert_{}R_d'.format(r)] = m_burkert_pred['m']
        halo_table['delta_m_burkert_{}R_d'.format(r)] = m_burkert_pred['error']
        
    halo_table['M_disc_stars_2.2'] = halos.M_disc_22(False).data
    halo_table['delta_M_disc_stars_2.2'] = (
                                     halos.M_disc_22(False).uncertainty.array)
    
    halo_table['M_disc_total_2.2'] = halos.M_disc_22(True).data
    halo_table['delta_M_disc_total_2.2'] = (
                                      halos.M_disc_22(True).uncertainty.array)
    
    halo_table['M_bulge_2.2'] = halos.M_bulge_22().data
    halo_table['delta_M_bulge_2.2'] = halos.M_bulge_22().uncertainty.array
    
    halo_table['M_halo_2.2_hernquist'] = halos.M_halo_22_hernquist().data
    halo_table['delta_M_halo_2.2_hernquist'] = (
                                halos.M_halo_22_hernquist().uncertainty.array)
    
    halo_table['M_halo_2.2_burkert'] = halos.M_halo_22_burkert().data
    halo_table['delta_M_halo_2.2_burkert'] = (
                                  halos.M_halo_22_burkert().uncertainty.array)
    
    halo_table['Gamma_hernquist'] = halos.total_shear('hernquist').data
    halo_table['delta_Gamma_hernquist'] = (
                             halos.total_shear('hernquist').uncertainty.array)
    
    halo_table['Gamma_burkert'] = halos.total_shear('burkert').data
    halo_table['delta_Gamma_burkert'] = (
                               halos.total_shear('burkert').uncertainty.array)

    halo_table['psi_hernquist_michikoshi'] = (
                             halos.psi_predicted_michikoshi('hernquist').data)
    halo_table['delta_psi_hernquist_michikoshi'] = (
                halos.psi_predicted_michikoshi('hernquist').uncertainty.array)
    
    halo_table['psi_burkert_michikoshi'] = (
                               halos.psi_predicted_michikoshi('burkert').data)
    halo_table['delta_psi_burkert_michikoshi'] = (
                  halos.psi_predicted_michikoshi('burkert').uncertainty.array)
    
    halo_table['psi_hernquist_seigar'] = (
                                 halos.psi_predicted_seigar('hernquist').data)
    halo_table['delta_psi_hernquist_seigar'] = (
                    halos.psi_predicted_seigar('hernquist').uncertainty.array)
    
    halo_table['psi_burkert_seigar'] = (
                                   halos.psi_predicted_seigar('burkert').data)
    halo_table['delta_psi_burkert_seigar'] = (
                      halos.psi_predicted_seigar('burkert').uncertainty.array)
    
    return halo_table
        