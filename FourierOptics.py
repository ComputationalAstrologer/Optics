#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:35:10 2018
   by Richard A. Frazin
   
This is a class of Fourier Optics functions, in which all optical elements are treated as
  thin phase screens.  The Fresnel propagation algorithm, based purely on spatial domain
  convolution, appears to be original.  Note that it takes advantage of FFT acceleration.
   
"""

import numpy as np
from scipy.signal import convolve as conv
from scipy.interpolate import interp1d, interp2d

# It works on a square computation grid, specified by a 1D spatial
#  coordinate vector x.
#params is a dict of stuff you need, including:
# params['wavelength'] - lambda (recommend micron units)
# params['max_chirp_step_deg'] (degrees) controls Fresnel propogation accuracy is controlled by 
# params['max_lens_step_deg'] (degrees) may require increasing resolution for lens phase screen
# params['max_pyramid_phase_step'] may require increasing the resolutin for the pyramid screen
# params['interp_style'] is passed to scipy.interpolate.interp2d (used for up-sampling field)
# params['pyramid_slope_deg'] (degrees) is the slope of the pyramid relative to the horizontal

class FourierOptics():
    def __init__(self, params):
        self.params = params
        return

    #1D Fresenel prop using convolution in the spatial domain
    # g - vector of complex-valued field in the inital plane
    # x - vector of coordinates in initial plane, corresponding to bin center locaitons
    # diam_out - diameter of region to be calculated in output plane
    # z - propagation distance
    # index_of_refaction - isotropic index of refraction of medium
    # set_dx = [True | False | dx (microns)]
    #   True  - forces full sampling of chirp according to self.params['max_chirp_step_deg']
    #           Note: this can lead to unacceptably large arrays.
    #   False - zeroes the kernel beyond where self.params['max_chirp_step_deg'] 
    #            exceedes the dx given in the x input array.  Note: the output dx will differ
    #            slightly from dx in the x input array.
    #...dx - sets the resolution to dx (units microns).  Note: the output dx will differ 
    #         slightly from this value
    # return_derivs - True to return deriv. of output field WRT z
    def ConvFresnel1D(self, g, x, diam_out, z, index_of_refraction=1,
                      set_dx=True, return_derivs=False):
        if g.shape != x.shape:
            raise Exception("Input field and grid must have same dimensions.")
        lam = self.params['wavelength']/index_of_refraction
        dx, diam = self.GetDxAndDiam(x)
        dx_new = dx  # this will probably change
        dPhiTol_deg = self.params['max_chirp_step_deg']
        dx_chirp = (dPhiTol_deg/180)*lam*z/(diam + diam_out)  # sampling criterion for chirp (factors of pi cancel)
        if isinstance(set_dx, bool):  # this step is needed so that 1 and 1.0 are not treated as True
            if set_dx == False: pass
            else:  # use chirp sampling criterion
                if dx_chirp < dx:
                    dx_new = dx_chirp
        else:  # take dx_new to be value of set_dx
            if not isinstance(set_dx, float):
                raise Exception("ConvFresnel1D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("ConvFresnel1D: numerical value of set_dx must be > 0.")
            dx_new = set_dx

        if not np.isclose(dx, dx_new):  # interpolate g onto a grid with spacing of (approx) dx_new
            [g, x] = self.ResampleField1D(g, x, dx_new)
            dx = x[1] - x[0]

        # make the kernel grid (s) match x as closely as possible
        ns = int(np.round(diam + diam_out)/dx)  # number of points on extended kernel
        s = np.linspace(-diam/2 - diam_out/2 + dx/2, diam/2 + diam_out/2 - dx/2, ns) # spatial grid of extended kernel
        indices_out = np.where(np.abs(s) < diam_out/2)[0] # get the part of s within the output grid

        #Calculate Fresnel convoltion kernel, (Goodman 4-16)
        #  Note: the factor p = 1/(lam*z) is applied later
        #  Also note: the factor -1j*np.exp(2j*np.pi*z/lam) causes unwanted oscillations with z
        kern = np.exp(1j*np.pi*s*s/(lam*z))  # Fresnel kernel
        if dx > dx_chirp:  # Where does |s| exceed the max step for this dx?
            s_max = lam*z*self.params['max_chirp_step_deg']/(360*dx)
            null_ind = np.where(np.abs(s) > s_max)[0]
            kern[null_ind] = 0
        h = conv(kern, g, mode='same', method='fft')  # h is on the s spatial grid

        p = 1/(lam*z)
        if not return_derivs:
            return([p*h[indices_out], s[indices_out]])
        #dpdz = (-1j/(lam*z*z) + 2*np.pi/(lam*lam*z))*np.exp(2j*np.pi*z/lam)  # includes unwanted oscillations
        dpdz = -1/(lam*z*z)
        dkerndz = -1j*np.pi*s*s*kern/(lam*z*z)
        dhdz = conv(dkerndz, g, mode='same', method='fft')
        s = s[indices_out]
        h = h[indices_out]
        dhdz = dhdz[indices_out]
        return([p*h, dpdz*h + p*dhdz, s])


    #This is an alternative to ConvFresnel1D.  Instead of using the quadaratic (paraxial)
    #  approximation of r, the full Huygens-Fresnel integral over the aperture is
    #  treated directly (see Intro to Fourier Optics eq. 4-9).  Note that this cannot
    #  be treated with a convolution integral.  This is designed to work when the paraxial
    #  approximation breaks down.
    def HuygensFresnel1D(self, g, x, diam_out, z, index_of_refraction=1,
                      set_dx=True):
        if g.shape != x.shape:
            raise Exception("Input field and grid must have same dimensions.")
        lam = self.params['wavelength']/index_of_refraction 
        dx, diam = self.GetDxAndDiam(x)
        dx_new = dx  # this will probably change
        xnew = None
        dPhiTol_deg = self.params['max_chirp_step_deg']
        dx_chirp = (dPhiTol_deg/45.)*lam*np.sqrt(z*z + (0.5*diam + 0.5*diam_out)**2)/(diam + diam_out)  # sampling criterion
        if isinstance(set_dx, bool):  # this step is needed so that 1 and 1.0 are not treated as True
            if set_dx == False:
                xnew = x
                nxnew = len(x)
            else:  # use chirp sampling criterion
                if dx_chirp < dx:
                    dx_new = dx_chirp
        else:  # take dx_new to be value of set_dx
            if not isinstance(set_dx, float):
                raise Exception("HuygensFresnel1D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("HuygensFresnel1D: numerical value of set_dx must be > 0.")
            dx_new = set_dx

        if xnew is None:
            nxnew = np.round(diam_out/dx_new).astype('int')
            xnew = np.linspace(-diam_out/2., diam_out/2., nxnew)
        f = np.zeros((nxnew,)).astype('complex')  #new field
        for k1 in range(nxnew):
            for k2 in range(len(x)):
                deltax = xnew[k1] - x[k2]
                if np.abs(deltax) < lam/10.: deltax = lam/10.  # small negative values are also mapped to lam/10
                rr = z*z + deltax**2
                r = np.sqrt(rr)
                #the simple form below leads to unwanted periodicity (it acts like DFT)
                #f[k1] += g[k2]*(-1j*z/(rr*lam))*np.exp(2j*np.pi*(r-z)/lam)  # subtract z from r to remove piston phase
                #instead, it is better to integrate over the source pixel
                q = (-z/np.pi)*(r/deltax)*np.sin( np.pi*deltax*dx/(lam*r) )
                f[k1] += q*(g[k2]/rr)*np.exp( 2j*np.pi(r - z)/lam )
        return([f,xnew])


    #This is an alternative to ConvFresnel2D.  Instead of using the quadaratic (paraxial)
    #  approximation of r, the full Huygens-Fresnel integral over the aperture is
    #  treated directly (see Intro to Fourier Optics eq. 4-9).  Note that this cannot
    #  be treated with a convolution integral.  This is designed to work when the paraxial
    #  approximation breaks down.
    #scr_thresh is an amplitude threshold for skipping a pixel of the source
    def HuygensFresnel2D(self, g, x, diam_out, z, index_of_refraction=1,
                      set_dx=True, return_derivs=False, src_thresh=1.e-4):
        if g.shape[0] != x.shape[0]:
            raise Exception("HuygensFresnel2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("HuygensFresnel2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("HuygensFresnel2D: input field array must be square.")

        lam = self.params['wavelength']/index_of_refraction
        dx, diam = self.GetDxAndDiam(x)
        dx_new = dx  # this will probably change
        xnew = None
        dPhiTol_deg = self.params['max_chirp_step_deg']
        dx_chirp = (dPhiTol_deg/45.)*lam*np.sqrt(z*z + (0.5*diam + 0.5*diam_out)**2)/(diam + diam_out)  # sampling criterion
        if isinstance(set_dx, bool):  # this step is needed so that 1 and 1.0 are not treated as True
            if set_dx == False:
                xnew = x
                nxnew = len(x)
            else:  # use chirp sampling criterion
                if dx < dx_chirp:
                    dx_new = dx_chirp
        else:  # take dx_new to be value of set_dx
            if not isinstance(set_dx, float):
                raise Exception("HuygensFresnel2D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("HuygensFresnel2D: numerical value of set_dx must be > 0.")
            dx_new = set_dx

        thresh = np.max(np.abs(g))*src_thresh
        if xnew is None:
            nxnew = np.round(diam_out/dx_new).astype('int')
            xnew = np.linspace(-diam_out/2., diam_out/2., nxnew)
        f = np.zeros((nxnew, nxnew)).astype('complex')  #new field
        for k2 in range(len(x)):
         for l2 in range(len(x)):
          if np.abs(g[k2,l2]) < thresh:  continue  # skip empty pixels
          for k1 in range(nxnew):
           for l1 in range(nxnew):
             if (x[k1]**2 + x[l1]**2) > diam_out:  continue
             deltax = xnew[k1] - x[k2]
             if np.abs(deltax) < lam/10.: deltax = lam/10.  # small negative values are also mapped to lam/10
             deltay = xnew[l1] - x[l2]
             if np.abs(deltay) < lam/10.: deltay = lam/10.
             rr = z*z + deltax**2 + deltay**2
             r = np.sqrt(rr)
             qx = (r/deltax)*np.sin( np.pi*deltax*dx/(lam*r) )
             qy = (r/deltay)*np.sin( np.pi*deltay*dx/(lam*r) )
             f[k1,l1] += (1j*lam/np.pi/np.pi)*qx*qy*(g[k2,l2]/rr)*np.exp( 2j*np.pi*(r - z)/lam )
        return([f,xnew])

    #2D Fresenel prop using convolution in the spatial domain
    # g - matrix of complex-valued field in the inital plane
    # x - vector of 1D coordinates in initial plane, corresponding to bin center locaitons
    #        it is assumed that the x and y directions have the same sampling.
    # diam_out - diameter of region to be calculated in output plane
    #    Note that the field is set to 0 outside of disk defined by r = diam_out/2.  This
    #      is because the chirp sampling criteria could be violated outside of this disk.
    # z - propagation distance
    # index_of_refaction - isotropic index of refraction of medium
    # set_dx = [True | False | dx (microns)]
    #   True  - forces full sampling of chirp according to self.params['max_chirp_step_deg']
    #           Note: this can lead to unacceptably large arrays.
    #   False - zeroes the kernel beyond where self.params['max_chirp_step_deg'] 
    #            is exceeded in the x input array.
    #...dx - sets the resolution to dx (units microns).  Note: the output may differ
    #         slightly from this value
    # return_derivs - True to return deriv. of output field WRT z (drops piston term)
    def ConvFresnel2D(self, g, x, diam_out, z, index_of_refraction=1,
                      set_dx=True, return_derivs=False):
        if g.shape[0] != x.shape[0]:
            raise Exception("ConvFresnel2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ConvFresnel2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ConvFresnel2D: input field array must be square.")

        lam = self.params['wavelength']/index_of_refraction
        dx, diam = self.GetDxAndDiam(x)
        dx_new = dx  # this will probably change
        dPhiTol_deg = self.params['max_chirp_step_deg']
        dx_chirp = (dPhiTol_deg/180)*lam*z/(diam + diam_out)  # sampling criterion for chirp (factors of pi cancel)
        if isinstance(set_dx, bool):  # this step is needed so that 1 and 1.0 are not treated as True
            if set_dx == False:  pass
            else:  # use chirp sampling criterion
                if dx < dx_chirp:
                    dx_new = dx_chirp
        else:  # take dx_new to be value of set_dx
            if not isinstance(set_dx, float):
                raise Exception("ConvFresnel2D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("ConvFresnel2D: numerical value of set_dx must be > 0.")
            dx_new = set_dx

        if not np.isclose(dx, dx_new):  # interpolate g onto a grid with spacing of approx dx_new
            [g, x] = self.ResampleField2D(g, x, dx_new, kind=self.params['interp_style'])
            dx = x[1] - x[0]

        # make the kernel grid (s) match x as closely as possible
        ns = int(np.round(diam + diam_out)/dx)  # number of points on extended kernel
        s = np.linspace(-diam/2 - diam_out/2 + dx/2, diam/2 + diam_out/2 - dx/2, ns) # spatial grid of extended kernel
        ind = np.where(np.abs(s) < diam_out/2)[0] # get the part of s within the 1D output grid
        [sx, sy] = np.meshgrid(s, s, indexing='xy')
        i_out = np.where(np.sqrt(sx*sx + sy*sy) > diam_out/2)

        #Calculate Fresnel convoltion kernel, (Goodman 4-16)
        #  Note: the factor p = 1/(lam*z) is applied later
        #  Also note: the factor -1j*np.exp(2j*np.pi*z/lam) causes unwanted oscillations with z
        kern = np.exp(1j*np.pi*(sx*sx + sy*sy)/(lam*z))  # Fresnel kernel
        if dx > dx_chirp:  # Where does |s| exceed the max step for this dx?
            s_max = lam*z*self.params['max_chirp_step_deg']/(360*dx)
            null_ind = np.where(np.sqrt(sx*sx + sy*sy) > s_max)
            kern[null_ind[0], null_ind[1]] = 0
        h = conv(kern, g, mode='same', method='fft')  # h is on the s spatial grid
        h[i_out[0], i_out[1]] = 0.  # zero the field outside the desired region
        h = h[ind[0]:ind[-1] + 1, ind[0]:ind[-1] + 1]
        p = 1/(lam*z)
        if not return_derivs:
            return([p*h, s[ind]])
        #dpdz = (-1j/(lam*z*z) + 2*np.pi/(lam*lam*z))*np.exp(2j*np.pi*z/lam)  # includes unwanted piston term
        dpdz = -1/(lam*z*z)
        dkerndz = -1j*np.pi*(sx*sx + sy*sy)*kern/(lam*z*z)
        dhdz = conv(dkerndz, g, mode='same', method='fft')
        dhdz[i_out[0], i_out[1]] = 0.
        dhdz = dhdz[ind[0]:ind[-1] + 1, ind[0]:ind[-1] + 1]
        return([p*h, dpdz*h + p*dhdz, s[ind]])

    #Apply thin lens phase transformation.
    # g - electric field impinging on lens
    # x - spatial coordinate
    # center - center of lens relative to zero of x
    # fl - lens focal length (same units as x and wavelength)
    def ApplyThinLens1D(self, g, x, center, fl, return_derivs=False):
        if g.shape != x.shape:
            raise Exception("Input field and spatial grid must have same dimensions.")
        [dx, diam] = self.GetDxAndDiam(x)
        lam = self.params['wavelength']
        max_step = self.params['max_lens_step_deg']*np.pi/180
        dx_tol = max_step*lam*fl/(2*np.pi*(diam/2 + np.abs(center)))
        if dx > dx_tol:  # interpolate onto higher resolution grid
            [g, xnew] = self.ResampleField1D(g, x, dx_tol)
        else:
            xnew = x
        h = g*np.exp(-1j*np.pi*(xnew - center)*(xnew - center)/(fl*lam))
        if not return_derivs:
            return([h, xnew])
        dhdc = 2j*np.pi*(xnew - center)*h/(lam*fl)
        return([h, dhdc, xnew])

    # similar to 1D version, except center must be of length 2
    # set_dx is bool or the desired value of dx
    def ApplyThinLens2D(self, g, x, center, fl, set_dx=True, return_derivs=False):
        if g.shape[0] != x.shape[0]:
            raise Exception("ApplyThinLens2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ApplyThinLens2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ApplyThinLens2D: input field array must be square.")
        if len(center) != 2:
            raise Exception("ApplyThinLens2D: center parameter must have len=2.")

        [dx, diam] = self.GetDxAndDiam(x)
        lam = self.params['wavelength']
        max_step = self.params['max_lens_step_deg']*np.pi/180
        dx_tol = max_step*lam*fl/(2*np.pi*(diam/2 + np.sqrt(center[0]**2 + center[1]**2)))
        if isinstance(set_dx, bool):
            if set_dx: # interpolate onto new grid
                [g, x] = self.ResampleField2D(g, x, dx_tol, kind=self.params['interp_style'])
            else:  # set_dx is False
                if (dx <= dx_tol): pass
                else: 
                    print ("ApplyThinLens2D: dx > dx_tol.  dx_tol = ", dx_tol, ", dx = ", dx)
                    raise Exception()
        else:  # set_dx not boolean
            if (set_dx <= dx_tol):
                [g, x] = self.ResampleField2D(g, x, set_dx, kind=self.params['interp_style'])
            else:
                print ("ApplyThinLens2D: set_dx > dx_tol.  dx_tol = ", dx_tol, ", set_dx = ", set_dx)
                raise Exception()

        [sx, sy] = np.meshgrid(x, x, indexing='xy')
        sx -= center[0]
        sy -= center[1]
        h = g*np.exp(-1j*np.pi*(sx*sx + sy*sy)/(fl*lam))
        if not return_derivs:
            return([h, x])
        dhdc0 = 2j*np.pi*sx*h/(lam*fl)
        dhdc1 = 2j*np.pi*sy*h/(lam*fl)
        return([h, [dhdc0, dhdc1], x])

    #This treats a planar interface as a phase screen.  The light is going
    #  in the +z direction, from medium A with index=nA, to medium B,
    #  with index=nB.
    #normal is a 2-component normal vector describing the orientation of
    #  the interface [i.e., (x,z).dot(normal) = 0]
    def ApplyPlanarPhaseScreen1D(self, g, x, normal, nA=1, nB=1.5):
        if g.shape != x.shape:
            raise Exception("ApplyPlanarPhaseScreen1D: input field and grid must have same sampling.")
        if len(normal) != 2:
            raise Exception("ApplyPlanarPhaseScreen1D: normal must have 2 components.")
        lam = self.params['wavelength']
        normal = np.array(normal)
        normal /= np.sum(normal*normal)
        assert normal[1] != 0
        if np.abs(normal[1]) > 0.999999999:  # return if the normal is in the z direction
            return([g, x])

        dx, diam = self.GetDxAndDiam(x)
        dz = - dx*normal[0]/normal[1]
        dph = 360*np.abs(dz*(nA - nB)/lam) 
        if dph > self.params['max_plane_step_deg']:
            dxnew = dx*self.params['max_plane_step_deg']/dph
            g, x = self.ResampleField1D(g, x, dxnew)

        z = -x*normal[0]/normal[1]  # Since phase increases with z, if z > 0, nB > nA, the effect on phase is negative
        ph = 2*np.pi*(nA - nB)*z/lam
        g = g*np.exp(1j*ph)
        return([g, x])

    #This treats a planar interface as a phase screen.  The light is going
    #  in the +z direction, from medium A with index=nA, to medium B,
    #  with index=nB.
    #normal is a 3-component normal vector describing the orientation of
    #  the interface [i.e., (x,y,z).dot(normal) = 0]
    def ApplyPlanarPhaseScreen2D(self, g, x, normal, nA=1, nB=1.5):
        if g.shape[0] != x.shape[0]:
            raise Exception("ApplyPlanarPhaseScreen2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ApplyPlanarPhaseScreen2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ApplyPlanarPhaseScreen2D: input field array must be square.")
        if len(normal) != 3:
            raise Exception("ApplyPlanarPhaseScreen2D: normal must have 3 components.")
        lam = self.params['wavelength']
        normal /= np.sum(normal*normal)
        assert normal[2] != 0
        if np.abs(normal[2]) > 0.9999999999:  # return if the normal is in the z direction
            return([g, x])

        dx, diam = self.GetDxAndDiam(x)
        dz = dx*np.sqrt(normal[0]*normal[0] + normal[1]*normal[1])/normal[2]
        dph = 360*np.abs(dz*(nA - nB)/lam) 
        if dph > self.params['max_plane_step_deg']:
            dxnew = dx*self.params['max_plane_step_deg']/dph
            g, x = self.ResampleField2D(g, x, dxnew, kind=self.params['interp_style'])

        # Since phase increases with z, if z > 0, nB > nA, the effect on phase is negative
        xs, ys = np.meshgrid(x, x, indexing='xy')
        z = - (xs*normal[0] + ys*normal[1])/normal[2]
        ph = 2*np.pi*(nA - nB)*z/lam
        g = g*np.exp(1j*ph)
        return([g, x])

    def ApplySnellAtPlane1D(self, g, x, normal, nA=1, nB=1.5):
        if g.shape != x.shape:
            raise Exception("ApplySnellAtPlane1D: input field and grid must have same sampling.")
        if len(normal) != 2:
            raise Exception("ApplySnellAtPlane1D: normal must have 2 components.")
        norm = np.array(normal)
        norm = norm/np.sum(norm*norm)
        assert normal[1] != 0
        norm = norm*np.sign(norm[1])  # make sure normal z compontent is +
        #make a coord system out of normal anti-normal
        norm = norm/np.sqrt(np.sum(norm*norm))
        anti = np.array([- norm[1]**2, norm[0]*norm[1]])
        anti = anti/np.sqrt(np.sum(anti*anti))
        #the transformation into (norm, anti) coords is governed by tmat
        #I think tmat has to be an involutory matrix so that tmat=itmat
        tmat = np.array([[anti[0], anti[1]], [norm[0],norm[1]]])
        itmat = np.linalg.inv(tmat)  # probably don't need this

        #note kA and kB are travel directions, not the wave vector "k"
        #kA is the wave direction in medium A, in (norm, anti) coords
        #kB                                 B, in (norm, anit) coords
        #also note that we expect norm and the travel direction to be mostly
        #  in the +z direction, so take note of how kA is calculated.
        #  kA[1, k] is the component parallel to the normal ("~z")
        #  kA[0, k] is the component perpendicular to the normal ("~x").
        [alpha, gamma] = self.GetDirectionCosines1D(g, x, index_of_refraction=nA)
        alphaB = 0*alpha  # these are direction cosines after the interface
        gammaB = 0*gamma
        kA = np.zeros(2)
        kB = np.zeros(2)
        cr = np.zeros(2)
        for k in range(len(x)):
            kA[:] = tmat.dot(np.array([alpha[k], gamma[k]]))
            kA *= np.sign(kA[1])  # must have a + normal component
            #sin_theta is the sin(theta) in Snell's law
            sin_theta = np.sqrt(1 - kA[1]**2)
            #now use Snell's law
            sin_theta = nB*np.abs(sin_theta)/nA
            kB[1] =  np.sqrt(1 - sin_theta**2)
            kB[0] = np.sign(kA[0])*np.abs(sin_theta)
            #now put kB into laboratory coordinate system
            cr[:] = itmat.dot(kB)
            gammaB[k] = cr[1]
            alphaB[k] = cr[0]

        #Now that we have alphaB, we know the phase gradient on the other side of the plane
        phi = np.zeros(len(x))  #phase of wave on other side
        mid = int(len(x)/2)
        dx, diam = self.GetDxAndDiam(x)
        kmag = nB*2*np.pi/self.params['wavelength']  # magnitude of wave vector k
        #intergrate gradient to get phi, note that this preserves the phase as the mid-point
        #  in order to reduce nonlinearities in application of Snell's law
        phi_mid = np.angle(g)[mid]  # phase at midpoint
        for k in np.arange(mid+1, len(x)):
            phi[k] = phi_mid + phi[k - 1] + dx*kmag*alphaB[k]
        for k in np.linspace(mid-1, 0, mid).astype('int'):
            phi[k] = phi_mid + phi[k+1] - dx*kmag*alphaB[k]

        return(np.abs(g)*np.exp(1j*phi))

    #This applies the pyramid's phase ramp, applies re-gridding if phase-step is too big
    # g - input field
    # x - input spatial grid
    # center - location of pyramid center with respect to x (same units as x)
    # return_derivs - duh
    # no_apex - if True, apex is removed and the beam is uniformly tilted
    def ApplyPyramidPhaseRamp1D(self, g, x, center=0, no_apex=False, return_derivs=False):
        if g.shape != x.shape:
            raise Exception("Input field and spatial grid must have same dimensions.")
        if no_apex:
            assert not return_derivs
        lam = self.params['wavelength']
        pslope = self.params['pyramid_slope_deg']*np.pi/180
        q = self.params['indref'] - 1  # index of refraction - 1
        ramp = 2*np.pi*q*np.tan(pslope)*np.abs(x - center)/lam  # z = -|x - center|*tan(theta)
        if no_apex:
            ramp = 2*np.pi*q*np.tan(pslope)*(x - center)/lam
        phase_diff = np.abs(ramp[1] - ramp[0])*180/np.pi
        if phase_diff > self.params['max_pyramid_phase_step']:  # if too big, we must decrase dx
            dx = x[1] - x[0]
            dx_new = dx*self.params['max_pyramid_phase_step']/phase_diff
            [g, xnew] = self.ResampleField1D(g, x, dx_new)
            x = xnew
            ramp = 2*np.pi*q*np.tan(pslope)*np.abs(x - center)/lam
            if no_apex:
                ramp = 2*np.pi*q*np.tan(pslope)*(x - center)/lam

        phasor = np.exp(1j*ramp)
        if not return_derivs:
            return([g*phasor, x])
        pos = np.where(x - center >= 0)[0]
        neg = np.where(x - center <  0)[0]
        xpos = x[pos]
        xneg = x[neg]
        aa = - np.ones(xpos.shape)*2*np.pi*q*np.tan(pslope)/lam  # deriv WRT center for positive part
        bb =   np.ones(xneg.shape)*2*np.pi*q*np.tan(pslope)/lam  # negative part
        d_phasor = 1j*phasor*np.hstack([bb,aa])
        return([g*phasor, g*d_phasor, x])


    #rotate (degrees) is the offset of the rotation of the pyramid (nominal is 45 deg)
    #   Zero rotation correspons to pyramid edges at 45 degrees to the x-y coords
    #tip (degrees) corresponds to rotating the pyramid about the apex in x-z plane
    #tilt (degress)                                                      y-z
    #   tip and tilt are relative to the pyramid axes
    #returns:
    # g - field (perhaps resampled) with phase screen applied
    # x - 1D coordinate vector (consistent with resampling)
    # ee - 3x3 matrix of pyramid axes.  Columns are (x, y, z) axes
    def ApplyPyramidPhaseRamp2D(self, g, x, center=[0,0], rotate=0, tip=0, tilt=0, return_derivs=False):
        if g.shape[0] != x.shape[0]:
            raise Exception("ApplyPyramidPhaseRamp2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ApplyPyramidPhaseRamp2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ApplyPyramidPhaseRamp2D: input field array must be square.")
        if len(center) != 2:
            raise Exception("ApplyPyramidPhaseRamp2D: center parameter must have len=2.")

        lam = self.params['wavelength']
        indref = self.params['indref']
        tslope = np.tan(self.params['pyramid_slope_deg']*np.pi/180)
        dx, diam = self.GetDxAndDiam(x)
        phase_step = 360*(indref-1)*dx*tslope/lam
        if phase_step > self.params['max_pyramid_phase_step']:
            dxnew = self.params['max_pyramid_phase_step']*lam/(360*tslope)
            [g, x] = self.ResampleField2D(g, x, dxnew, kind=self.params['interp_style'])

        [height, ee] = self.PyramidHeight2D(x, center, rotate, tip, tilt)

        ramp = np.exp(-2j*np.pi*height*(indref - 1)/lam)
        g *= ramp
        if not return_derivs:
            return([g, x, ee])
        # Unfortunately, I think finite differences is the best option for these derivatives.
        #    Make sure the steps are not to small for the delta x

    #see self.NSidedPyramidHeight for the other parameters
    #reflective (bool) - if True phase change is multiplied by -2 and self.params['indref'] doesn't matter
    #set_dx [True, False, scalar], allowing the field to be resampled according to self.params['max_pyramid phase step'],
    #  or, it is a scalar value specifying dx.  This is useful for finite differencing.
    #Field is only resampled if the initial grid does not have enough resolution to resolve the phase step
    def ApplyNSidedPyrPhase2D(self, g, x, SlopeDeviations=None, FaceCenterAngleDeviations=None, N=4,
                              NominalSlope=1., rot0=None, reflective=True, set_dx=False):
        if not isinstance(set_dx, bool):
            assert np.isscalar(set_dx) and set_dx > 0.
        if g.shape[0] != x.shape[0]:
            print('g.shape = ' , g.shape, ', x.shape = ', x.shape)
            raise Exception("ApplyNSidedPyrPhase2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ApplyNSidedPyrPhase2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ApplyNSidedPyrPhase2D: input field array must be square.")
        assert np.isscalar(NominalSlope) and NominalSlope > 0.
        if SlopeDeviations is None:
            SlopeDeviations = np.zeros((N,))
        SlopeDeviations = np.array(SlopeDeviations)
        assert SlopeDeviations.shape == (N,)
        lam = self.params['wavelength']
        tslope = np.tan(np.max(NominalSlope + SlopeDeviations)*np.pi/180)
        dx, diam = self.GetDxAndDiam(x)
        if reflective:
            phase_step = 2*360*dx*tslope/lam
        else:
            indref = self.params['indref']
            phase_step = 360*(indref-1)*dx*tslope/lam

        if isinstance(set_dx, bool):
            if set_dx:  # resample field according to criterion below
                if phase_step > self.params['max_pyramid_phase_step']:
                    dxnew = self.params['max_pyramid_phase_step']*lam/(360*tslope)
                    [g, x] = self.ResampleField2D(g, x, dxnew, kind=self.params['interp_style'])
            else:  pass # don't resample field
        else:
            if not np.isclose(set_dx, dx):
                [g, x] = self.ResampleField2D(g, x, set_dx, kind=self.params['interp_style'])

        height = self.NSidedPyramidHeight(x, SlopeDeviations=SlopeDeviations, FaceCenterAngleDeviations=FaceCenterAngleDeviations,
                                          N=N, NominalSlope=NominalSlope, rot0=rot0)
        if reflective:
            ramp = np.exp(4j*np.pi*height/lam)
        else:
            ramp = np.exp(-2j*np.pi*height(indref-1)/lam)
        return((ramp*g, x))


#SlopeDeviations (degrees) is the vector of deviations of the angles relative to the z=0 plane of each face (len=N)
    #Nominal slope (degrees, > 0) is the nominal slope of the pyramid faces relative to the z=0 plane
    #N is the number of sides of the pyramid
    #FaceCenterAngleDeviations (degrees) is the deviation in azimuthal angles (about the z-axis) of the face centers.
    #rot0 (degrees) is constant that gets added to all N FaceCenterAngles
    def NSidedPyramidHeight(self, x, SlopeDeviations=None, FaceCenterAngleDeviations=None, N=4, NominalSlope=1., rot0=None):
        if rot0 is None: rot0 = 180./N
        if FaceCenterAngleDeviations is None:
            FaceCenterAngleDeviations = np.zeros((N,))
        else:
            FaceCenterAngleDeviations = np.array(FaceCenterAngleDeviations)
            assert FaceCenterAngleDeviations.shape == (N,)
        assert np.isscalar(NominalSlope) and NominalSlope > 0.
        if SlopeDeviations is None:
            SlopeDeviations = np.zeros((N,))
        SlopeDeviations = np.array(SlopeDeviations)
        assert SlopeDeviations.shape == (N,)
        FaceCenterAngles = np.linspace(0., 360*(N-1)/N, N)
        FaceCenterAngles += FaceCenterAngleDeviations + rot0
        AlphaVec = (NominalSlope*np.ones((N,)) + SlopeDeviations)*np.pi/180.

        normvec = np.zeros((3,N))  # each col is normal vector
        for k in range(N):
            normvec[1,k] = np.sin(AlphaVec[k])
            normvec[2,k] = np.cos(AlphaVec[k])
            normvec[:,k] = self.RotationMatrix(FaceCenterAngles[k], axis='z').dot(normvec[:,k])

        [px, py] = np.meshgrid(x , x , indexing='xy')
        z = np.zeros((x.shape[0], x.shape[0], N))  # contains the heights of all N planes
        for k in range(N):
            z[:, :, k] = (normvec[0,k]*px + normvec[1,k]*py)/normvec[2,k]
        zs = np.min(z, axis=2)
        return(zs)



    #tip and tilt are with respect the pyramid axes
    #this also keeps track of the pyramid axis: ex, ey, ez.  This makes it easier to remove
    #  the non-physical angular offset introduced by (tip, tilt) != 0
    #Returns height profile and coordinate axes of pyramid
    def PyramidHeight2D(self, x, center, rotate, tip, tilt):
        #calculate the height of the pyramid face.  Rotate according to (rotate, tip, tilt) params
        tan_slope = np.tan(self.params['pyramid_slope_deg']*np.pi/180.)
        roof = self.params['pyramid_roofsize']/2
        nx = x.shape[0]
        #These transformations put the optical system coords into pyramid coords
        [px, py] = np.meshgrid(x - center[0], x - center[1], indexing='xy')
        xs = np.zeros(px.shape)
        ys = np.zeros(px.shape)
        zs = np.zeros(px.shape)  # face height (taken to be >= 0)
        rotate = (rotate + 45)*np.pi/180
        for k in range(nx):
            for l in range(nx):
                sx = px[k,l]*np.cos( rotate) + py[k,l]*np.sin(rotate)
                sy = px[k,l]*np.sin(-rotate) + py[k,l]*np.cos(rotate)
                th = np.arctan2(sy, sx)
                xs[k,l] = sx
                ys[k,l] = sy
                if (th >= -np.pi/4) and (th < np.pi/4):  # +x direction
                    if np.abs(sx) >= roof:
                        zs[k,l] = (np.abs(sx) - roof)*tan_slope
                elif (th >= np.pi/4) and (th < 3*np.pi/4):  # +y direction
                    if np.abs(sy) >= roof:
                        zs[k,l] = (np.abs(sy) - roof)*tan_slope
                elif (th >= 3*np.pi/4) or (th < -3*np.pi/4):  # -x direction
                    if np.abs(sx) >= roof:
                        zs[k,l] = (np.abs(sx) - roof)*tan_slope
                else:  # -y direction
                    if np.abs(sy) >= roof:
                        zs[k,l] = (np.abs(sy) - roof)*tan_slope
        if tip != 0:  # to apply tip, rotate about y-axis.  for tilt, rotate about x-axis
            rymat = self.RotationMatrix( tip, 'y')
        else:
            rymat = np.eye(3)
        if tilt != 0:
            rxmat = self.RotationMatrix( tilt, 'x')
        else:
            rxmat = np.eye(3)
        if tip !=0 or tilt != 0:
            rxymat = rxmat.dot(rymat)
            v = np.zeros(3); u = np.zeros(3)
            for k in range(nx):
                for l in range(nx):
                    v[0] = xs[k,l]; v[1] = ys[k,l]; v[2] = zs[k,l]
                    u = rxymat.dot(v)
                    xs[k,l] = u[0]; ys[k,l] = u[1]; zs[k,l] = u[2]
        else:
            rxymat = np.eye(3)

        mat = self.RotationMatrix(-rotate*180/np.pi ,'z').dot(np.linalg.inv(rxymat))
        ee = mat.dot(np.eye(3))  # pyramid axes are columns of matrix ee

        return([zs, ee])


    #Evaluate the derivative of the phase w.r.t. x to get the direction cosine in 1D.
    #  note: phase = (2pi/lambda)(alpha*x + sqrt(1 - alpha^2)*z ),
    #  so d(phase)/dx = (2pi/lambda)*alpha
    #returns:
    #   alpha - array of direction cosines w.r.t. x-axis
    #   gamma                                     z  (note forward propagation only)
    def GetDirectionCosines1D(self, g, x, index_of_refraction=1):
        if g.shape != x.shape:
            raise Exception("GetDirectionCosines1D: input field and x must have the same shape.")
        dx, diam = self.GetDxAndDiam(x)
        kmag = 2*np.pi*index_of_refraction/self.params['wavelength']  # magnitude of k

        ph = np.unwrap(np.angle(g))
        ph -= ph[int(len(x)/2)]
        dph = ph[1:] - ph[:-1]
        dphdx = np.hstack((dph[0], dph))/dx
        alpha = dphdx/kmag
        gamma = np.sqrt(1 - alpha*alpha)
        return([alpha, gamma])

    #Evaluate the derivative of the phase w.r.t. (x,y) to get the direction cosines in 2D.
    #  note: phase = (2pi/lambda)(alpha*x + beta*y + sqrt(1 - alpha^2 - beta^2)*z ),
    #  so d(phase)/dx = (2pi/lambda)*alpha, d(phase)/dy = (2pi/lambda)*beta
    #returns:
    #   alpha - array of direction cosines w.r.t. x-axis
    #   beta                                      y
    #   gamma                                     z  (note forward propagation only)
    def GetDirectionCosines2D(self, g, x, index_of_refraction=1):
        if g.shape[0] != x.shape[0]:
            raise Exception("GetDirectionCosines2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("GetDirectionCosines2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("GetDirectionCosines2D: input field array must be square.")

        dx, diam = self.GetDxAndDiam(x)
        kmag = 2*np.pi*index_of_refraction/self.params['wavelength']  # magnitude of k
        nx = len(x)

        ph = np.angle(g)
        ph -= ph[int(nx/2), int(nx/2)]
        line = np.unwrap(ph[:, int(nx/2)])
        ph[:, int(nx/2)] = line
        ph = np.unwrap(ph, axis=1)

        dphdx = ph[:, 1:] - ph[:,:-1]
        dphdx = np.vstack((dphdx[:,0], dphdx.T)).T
        alpha = dphdx/(dx*kmag)
        dphdy = ph[1:, :] - ph[:-1, :]
        dphdy = np.vstack((dphdy[0,:], dphdy))
        beta = dphdy/(dx*kmag)

        dph = ph[1:] - ph[:-1]
        dphdx = np.hstack((dph[0], dph))/dx
        alpha = dphdx/kmag
        gamma = np.sqrt(1 - alpha*alpha - beta*beta)
        return([alpha, beta, gamma])

#   This returns a tip_tilt phasor (square array) in a pupil plane.
#   Np - the number of pixels across the pupil
#   angle (degrees) - the modulation angle
#   amp (lambda/D units) - modulation radius
    def TipTiltPhasor(self, Np, angle_deg, amp=3.):
        s = np.linspace(-0.5, 0.5, Np)
        (xx, yy) = np.meshgrid(s,s)
        ph = np.zeros((Np, Np))
        v = np.zeros((2,))
        unit = np.array([np.cos(angle_deg*np.pi/180), np.sin(angle_deg*np.pi/180)])
        for k in range(Np):
            for l in range(Np):
                v[0] = xx[l,k]
                v[1] = yy[l,k]
                ph[l,k] = 2*np.pi*amp*v.dot(unit)
        return( np.exp(1j*ph) )


    #Resample the field to achieve a target spatial resolution given by dx_new
    #  Returns resampled field and new x
    #g - input field
    #x - initial 1D spatial grid
    #dx_new (microns) - desired spatial resolution
    #kind, fill_value - keywords passed to scipy.interpolate.interp2d
    def ResampleField2D(self, g, x, dx_new, kind='cubic', fill_value=None):
        if g.shape[0] != x.shape[0]:
            raise Exception("ResampleField2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ResampleField2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ResampleField2D: input field array must be square.")

        dx, diam = self.GetDxAndDiam(x)
        nx = int(np.round(diam/dx_new))
        dxnew = diam/nx
        xnew = np.linspace(-diam/2 + dxnew/2, diam/2 - dxnew/2, nx)
        #interp2d doesn't like complex numbers.  So stupid.
        interp_real = interp2d(x, x, np.real(g), kind=kind, fill_value=fill_value)
        interp_imag = interp2d(x, x, np.imag(g), kind=kind, fill_value=fill_value)
        g = interp_real(xnew, xnew) + 1j*interp_imag(xnew, xnew)
        return([g, xnew])

    #Resample the field to achieve a target spatial resolution given by dx_new
    #  Returns resampled field and new x
    #g - input field
    #x - initial 1D spatial grid
    #dx_new (microns) - desired spatial resolution
    #kind, fill_value - keywords passed to scipy.interpolate.interp1d
    def ResampleField1D(self, g, x, dx_new, kind='quadratic', fill_value='extrapolate'):
        if g.shape != x.shape:
            raise Exception("ResampleField1D: input field and x must have the same shape.")
        dx, diam = self.GetDxAndDiam(x)
        nx = int(np.round(diam/dx_new))
        dxnew = diam/nx
        xnew = np.linspace(-diam/2 + dxnew/2, diam/2 - dxnew/2, nx)
        interp = interp1d(x, g, kind=kind, fill_value=fill_value)
        g = interp(xnew)
        return([g, xnew])

    #3D rotation matrix for positive (?) rotations about 'axis'
    #t is the rotation angle in degrees.
    #axis must be 'x', 'y', or 'z'
    def RotationMatrix(self, t, axis='x'):
        assert axis == 'x' or axis == 'y' or axis == 'z'
        s = t*np.pi/180
        if axis == 'x':
            mat = np.array([[1, 0, 0], [0, np.cos(s), np.sin(s)],[0, np.sin(-s), np.cos(s)]])
        elif axis == 'y':
            mat = np.array([[np.cos(s), 0, np.sin(s)],[0, 1, 0],[np.sin(-s), 0, np.cos(s)]])
        else:  # 'z'
            mat = np.array([[np.cos(s), np.sin(s) , 0], [np.sin(-s), np.cos(s),0], [0, 0, 1]])
        return(mat)

    # assuming a regular spaced grid with values at bin centers,
    #  get spacing and diameter from spatial grid x
    def GetDxAndDiam(self, x):  # assumes x values are bin centers
        nx = x.shape[0]
        dx = x[1] - x[0]
        diam = (x[-1] - x[0])*(1 + 1/(nx - 1))
        assert diam > 0
        assert dx > 0
        return([dx, diam])