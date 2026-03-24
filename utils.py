import pywt
import numpy as np
from numpy.fft import fft, ifft


def getWaveletTransforms(n,wavelet_type = "db2",level = 5, weight=1):
    mode = "periodization"

    
    coeffs_tpl = pywt.wavedec(data=np.zeros(n), wavelet=wavelet_type, mode=mode, level=level)
    coeffs_1d, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs_tpl)
    coeffs_tpl_rec = pywt.unravel_coeffs(coeffs_1d, coeff_slices, coeff_shapes)

    scaling_vec = np.zeros_like(coeffs_1d)
    for i,slice in enumerate(coeff_slices):
        if i==0:
            scaling_vec[slice] += weight**i
        else: 
            scaling_vec[slice['d']] += weight**i
        
    

    def py_W(x):
        alpha = pywt.wavedec(data=x, wavelet=wavelet_type, mode=mode, level=level)
        alpha, _, _ = pywt.ravel_coeffs(alpha)
        return alpha

    def py_Ws(alpha):
        coeffs = pywt.unravel_coeffs(alpha, coeff_slices, coeff_shapes,output_format='wavedec')
        rec = pywt.waverec(coeffs, wavelet=wavelet_type, mode=mode)
        return rec
    
    return py_W, py_Ws,scaling_vec



def getWaveletTransforms_2D(n,m,wavelet_type = "db2",level = 5, weight=1):
    mode = "periodization"

    
    coeffs_tpl = pywt.wavedecn(data=np.zeros((n, m)), wavelet=wavelet_type, mode=mode, level=level)
    coeffs_1d, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs_tpl)
    coeffs_tpl_rec = pywt.unravel_coeffs(coeffs_1d, coeff_slices, coeff_shapes)

    scaling_vec = np.zeros_like(coeffs_1d)
    
    for i,slice in enumerate(coeff_slices):
        if i==0:
            scaling_vec[slice] += weight**i
        else: 
            scaling_vec[slice['ad']] += weight**i
            scaling_vec[slice['da']] += weight**i
            scaling_vec[slice['dd']] += weight**i
    


    def py_W(im):
        alpha = pywt.wavedecn(data=im, wavelet=wavelet_type, mode=mode, level=level)
        alpha, _, _ = pywt.ravel_coeffs(alpha)
        return alpha

    def py_Ws(alpha):
        coeffs = pywt.unravel_coeffs(alpha, coeff_slices, coeff_shapes)
        im = pywt.waverecn(coeffs, wavelet=wavelet_type, mode=mode)
        return im
    
    return py_W, py_Ws, scaling_vec



# define filter
def GaussianFilter(s,n): 
    x = np.hstack((np.arange(0,n//2), np.arange(-n//2,0)))
    h = np.exp( (-x**2)/(2*s**2) )
    h = h/sum(h)
    return h


def GaussianFilter_2d(s,n,m): 
    x = np.hstack((np.arange(0,n//2), np.arange(-n//2,0)))
    y = np.hstack((np.arange(0,m//2), np.arange(-m//2,0)))
    [X,Y] = np.meshgrid(y,x)
    h = np.exp( (-X**2-Y**2)/(2*s**2) )
    h = h/sum(sum(h))
    return h


def rFISTA(proxF, dG, gamma, xinit,niter,mfunc):
    tol = 1e-16

    x = xinit
    z = x
    t=1
    fval = []
    for k in range(niter):
        xkm = x
        ykm = z

        x =  proxF( z - gamma*dG(z), gamma )
        tnew = (1+np.sqrt(1+4*t**2))/2

        z = x + (t-1)/(tnew)*(x-xkm)
        t = tnew
        if np.sum((ykm-x)*(x-xkm))>0:
            z=x;
        fval.append(mfunc(x))

        if np.linalg.norm(xkm-x)<tol:
            break
    return x, fval
        
def ISTA(proxF, dG, gamma, xinit,niter,mfunc):
    x = xinit
    
    fval = []
    for k in range(niter):
        x =  proxF( x - gamma*dG(x), gamma )
        fval.append(mfunc(x))

    return x, fval
