"""
Telescope geometry definitions.

Grace E. Chesmore
May 2021
"""

import numpy as np


class initialize_telescope_geometry:

    F_2 = 7000
    th_1 = np.arctan(1 / 2)  # Primary mirror tilt angle
    th_2 = np.arctan(1 / 3)  # Secondary mirror tilt angle
    th2 = (-np.pi / 2) - th_2
    th_fwhp = 44 * np.pi / 180  # Full width half power [rad]
    N_scan = 100  # Pixels in 1D of grid
    de_ang = 1 / 60 * np.pi / 180  # Far-field angle increment, arcsec = 1/60 degree
    lambda_ = (30.0 / 100.0) * 0.01  # Source wavelength [m]
    k = 2 * np.pi / lambda_  # Wavenumber [1/m]

    # Receiver feed position [um]
    rx_x = 0
    rx_y = 0
    rx_z = 0

    # Phase reference [m]
    x_phref = 0
    y_phref = -7.2
    z_phref = 0

    # Center of rotation [m]
    x_rotc = 0
    y_rotc = -7.2
    z_rotc = 0

    # Source position (tower) [m]
    x_tow = 0
    y_tow = -7.2
    z_tow = 1e7

    # Azimuth and Elevation center [rad]
    az0 = 0
    el0 = np.arctan(-y_tow / z_tow)

    # Aperture plane [m]
    x_ap = 3
    y_ap = -7.2
    z_ap = 4.0


# Matrix Coefficients defining mirror surfaces
# Primary Mirror
a1 = np.zeros((7, 7))
a1[0, :] = [0, 0, -57.74022, 1.5373825, 1.154294, -0.441762, 0.0906601]
a1[1, :] = [0, 0, 0, 0, 0, 0, 0]
a1[2, :] = [-72.17349, 1.8691899, 2.8859421, -1.026471, 0.2610568, 0, 0]
a1[3, :] = [0, 0, 0, 0, 0, 0, 0]
a1[4, :] = [1.8083973, -0.603195, 0.2177414, 0, 0, 0, 0]
a1[5, :] = [0, 0, 0, 0, 0, 0, 0]
a1[6, :] = [0.0394559, 0, 0, 0, 0, 0, 0]
# Secondary Mirror
a2 = np.zeros((8, 8))
a2[0, :] = [0, 0, 103.90461, 6.6513025, 2.8405781, -0.7819705, -0.0400483, 0.0896645]
a2[1, :] = [0, 0, 0, 0, 0, 0, 0, 0]
a2[2, :] = [115.44758, 7.3024355, 5.7640389, -1.578144, -0.0354326, 0.2781226, 0, 0]
a2[3, :] = [0, 0, 0, 0, 0, 0, 0, 0]
a2[4, :] = [2.9130983, -0.8104051, -0.0185283, 0.2626023, 0, 0, 0, 0]
a2[5, :] = [0, 0, 0, 0, 0, 0, 0, 0]
a2[6, :] = [-0.0250794, 0.0709672, 0, 0, 0, 0, 0, 0]
a2[7, :] = [0, 0, 0, 0, 0, 0, 0, 0]

R_N = 3000  # [mm]

# These functions define the mirror surfaces,
# and the normal vectors on the surfaces.
def z1(x, y):
    amp = 0
    for ii in range(7):
        for jj in range(7):
            amp += a1[ii, jj] * ((x / R_N) ** ii) * ((y / R_N) ** jj)
    return amp


def z2(x, y):
    amp = 0
    for ii in range(8):
        for jj in range(8):
            amp += a2[ii, jj] * ((x / R_N) ** ii) * ((y / R_N) ** jj)
    return amp


def d_z1(x, y):
    amp_x = 0
    amp_y = 0
    for ii in range(7):
        for jj in range(7):
            amp_x += (
                a1[ii, jj] * (ii / R_N) * ((x / R_N) ** (ii - 1)) * ((y / R_N) ** jj)
            )
            amp_y += (
                a1[ii, jj] * ((x / R_N) ** ii) * (jj / R_N) * ((y / R_N) ** (jj - 1))
            )
    return amp_x, amp_y


def d_z2(x, y):
    amp_x = 0
    amp_y = 0
    for ii in range(8):
        for jj in range(8):
            amp_x += (
                a2[ii, jj] * (ii / R_N) * ((x / R_N) ** (ii - 1)) * ((y / R_N) ** jj)
            )
            amp_y += (
                a2[ii, jj] * ((x / R_N) ** ii) * (jj / R_N) * ((y / R_N) ** (jj - 1))
            )
    return amp_x, amp_y


# Coordinate transfer functions. Transferring
# coordinates between telescope reference frame
# and mirror reference frame, and vice versa.
def m1_into_tele(x, y, z):
    th1 = initialize_telescope_geometry.th_1
    xx = x * np.cos(np.pi) + z * np.sin(np.pi)
    yy = y
    zz = -x * np.sin(np.pi) + z * np.cos(np.pi)

    x_rot1 = xx
    y_rot1 = yy * np.cos(th1) - zz * np.sin(th1) - 7200
    z_rot1 = (yy * np.sin(th1) + zz * np.cos(th1)) - 3600
    return x_rot1, y_rot1, z_rot1


def m2_into_tele(x, y, z):
    th2 = initialize_telescope_geometry.th2
    x_temp = x * np.cos(np.pi) - y * np.sin(np.pi)
    y_temp = x * np.sin(np.pi) + y * np.cos(np.pi)
    z_temp = z

    x_rot2 = x_temp
    y_rot2 = (y_temp * np.cos(th2) - z_temp * np.sin(th2)) - 4800 - 7200
    z_rot2 = y_temp * np.sin(th2) + z_temp * np.cos(th2)
    return x_rot2, y_rot2, z_rot2


def tele_into_m1(x, y, z):
    th1 = initialize_telescope_geometry.th_1
    z += 3600
    y += 7200
    x_temp = x
    y_temp = y * np.cos(-th1) - z * np.sin(-th1)
    z_temp = y * np.sin(-th1) + z * np.cos(-th1)

    x2 = x_temp * np.cos(np.pi) + z_temp * np.sin(np.pi)
    y2 = y_temp
    z2 = -x_temp * np.sin(np.pi) + z_temp * np.cos(np.pi)

    return x2, y2, z2


def tele_into_m2(x, y, z):
    th2 = initialize_telescope_geometry.th2
    y += 4800 + 7200
    x_temp = x
    y_temp = y * np.cos(-th2) - z * np.sin(-th2)
    z_temp = y * np.sin(-th2) + z * np.cos(-th2)

    x2 = x_temp * np.cos(-np.pi) - y_temp * np.sin(-np.pi)
    y2 = x_temp * np.sin(-np.pi) + y_temp * np.cos(-np.pi)
    z2 = z_temp
    return x2, y2, z2


def tele_geo_init(x, y, z, el, az):
    tele_geo = initialize_telescope_geometry()
    tele_geo.rx_x = x
    tele_geo.rx_y = y
    tele_geo.rx_z = z
    tele_geo.el0 += el
    tele_geo.az0 += az
    return tele_geo
