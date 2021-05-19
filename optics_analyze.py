"""
Miscellaneous functions used to analyze 
and modify optical results. 

Grace E. Chesmore
May 2021
"""

import numpy as np


# Phase unwrapping
def twodunwrapx(array):
    phase_offset = np.arange(-1000, 1000) * 2 * np.pi

    end_i = np.shape(array)[0] - 1
    end_j = np.shape(array)[1] - 1

    i = int(end_i / 2)
    while i < end_i:
        j = int(end_j / 2)
        while j < (np.shape(array))[1] - 1:
            current_val = [array[i, j]]
            next_val = array[i, j + 1] + phase_offset
            diff = np.abs(next_val - current_val)
            best = np.where(diff == np.min(diff))
            array[i, j + 1] = next_val[best][0]

            j += 1
        i += 1

    i = int(end_i / 2)
    while i > 0:
        j = int(end_j / 2)
        while j > 0:
            current_val = [array[i, j]]
            next_val = array[i, j - 1] + phase_offset
            diff = np.abs(next_val - current_val)
            best = np.where(diff == np.min(diff))
            array[i, j - 1] = next_val[best][0]

            j -= 1
        i -= 1

    i = int(end_i / 2)
    while i < end_i - 1:
        j = int(end_j / 2)
        while j > 0:
            current_val = [array[i, j]]
            next_val = array[i, j - 1] + phase_offset
            diff = np.abs(next_val - current_val)
            best = np.where(diff == np.min(diff))
            array[i, j - 1] = next_val[best][0]

            j -= 1
        i += 1

    i = int(end_i / 2)
    while i > 0:
        j = int(end_j / 2)
        while j < end_j - 1:
            current_val = [array[i, j]]
            next_val = array[i, j + 1] + phase_offset
            diff = np.abs(next_val - current_val)
            best = np.where(diff == np.min(diff))
            array[i, j + 1] = next_val[best][0]

            j += 1
        i -= 1

    return array


# Unwraps the phase two times. The first time the phase is
# transposed and unwrapped, then it is transposed again
# (returned to normal state) and unwrapped.
def twodunwrap(array):
    xunwraped = twodunwrapx(np.transpose(array))
    unwrapped = twodunwrapx(np.transpose(xunwraped))
    return unwrapped


# Unwraps the phase (calling on other unwrap functions) and
# normalizes to the center of the phase measurement.
def do_unwrap(phi):
    unwraped_phi = twodunwrap(phi)
    #    print(unwraped_phi[int(len(unwraped_phi)/2),int(len(unwraped_phi)/2)])
    # unwraped_phi = unwraped_phi - unwraped_phi[0,0]
    unwraped_phi = (
        unwraped_phi
        - unwraped_phi[int(len(unwraped_phi) / 2), int(len(unwraped_phi) / 2)]
    )
    return unwraped_phi


# Rotate coordinates in azimuth and elevation.
def rotate_azel(xyz, az, el):

    out = np.zeros(np.shape(xyz))

    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    # Rotate in elevation (note we assume z is along the elevation direction)
    xt = np.cos(el) * x + np.sin(el) * z
    yt = y
    zt = (-1.0) * np.sin(el) * x + np.cos(el) * z

    # Rotate in azimuth
    out[0] = xt
    out[1] = np.cos(az) * yt + np.sin(az) * zt
    out[2] = (-1.0) * np.sin(az) * yt + np.cos(az) * zt

    return out


# Given two arrays, calculate the 2D power spectrum
# for a given ell range, pixel size, and pixel number.
def calculate_2d_spectrum(Map1, Map2, delta_ell, ell_max, pix_size, N):
    "calcualtes the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"
    N = int(N)

    # Make a 2d ell coordinate system
    ones = np.ones(N)
    inds = (np.arange(N) + 0.5 - N / 2.0) / (N - 1.0)
    kY = np.outer(ones, inds) / (pix_size / 60.0 * np.pi / 180.0)
    kX = np.transpose(kY)
    K = np.sqrt(kX ** 2.0 + kY ** 2.0)
    ell_scale_factor = 2.0 * np.pi
    ell2d = K * ell_scale_factor

    # Make an array to hold the power spectrum results
    N_bins = int(ell_max / delta_ell)
    ell_array = np.arange(N_bins)
    CL_array = np.zeros(N_bins)
    input_maps = (np.conj(Map1) * Map1) / np.sum(abs(np.conj(Map1) * Map1))

    # 2d fourier transform of the map
    FMap1 = np.fft.fft2(np.fft.fftshift(input_maps))
    FMap2 = np.fft.fft2(np.fft.fftshift(input_maps))
    PSMap = np.fft.fftshift(np.real(np.conj(FMap1) * FMap2))

    # Fill out the spectra
    i = 0
    while i < N_bins:
        ell_array[i] = (i + 0.5) * delta_ell
        inds_in_bin = (
            (ell2d >= (i * delta_ell)) * (ell2d < ((i + 1) * delta_ell))
        ).nonzero()
        CL_array[i] = np.mean(PSMap[inds_in_bin])
        i = i + 1
    # Return the power spectrum and ell bins

    return (ell_array, CL_array)


# Elevation offset of holography measurements
def el_offset(x):
    slope = (-0.0204345 - 0.00719988) / (400)
    return x * slope


# Azimuth offset of holography measurements
def az_offset(x):
    slope = (0.01367175) / (200)
    return x * slope


# Shifts for holography measurements
def sh_z(z):
    return z * ((0.33 + 0.33) / 1200)


def sh_x(z):
    return z * ((0.36 + 0.36) / 1200)


# Ruze equation quantifying gain loss due
# to surface defects on antenna.
def ruze(eps, lam):
    return np.exp(-2 * (4 * np.pi * eps / lam) ** 2)


# Computes the RMS of z for a given area(x,y)
def rms(x, y, z):
    aperture_r = 2.75  # apodized beam radius [m]
    rr = np.where(
        ((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2) <= aperture_r ** 2
    )  # throw out outliers
    z_rms = z[rr]
    return np.sqrt(np.sum(z_rms ** 2) / len(z_rms))
