"""
Aperture field analysis.

Grace E. Chesmore
May 2021
"""

import ap_field as af
import far_field as ff
import matplotlib.pyplot as plt
import numpy as np
import optics_analyze as oa
import pan_fitting as pf
import pan_mod as pm
import scipy
from scipy.spatial import distance

imu = np.complex(0, 1)

# Optional function to remove gradient
# term from aperture fields. We don't use
# in this work, but provide as an optional
# function.
def remove_grad(x, y, phi, fftd):

    x = x[np.where(np.isnan(phi) == False)]
    y = y[np.where(np.isnan(phi) == False)]
    fftd = fftd[np.where(np.isnan(phi) == False)]
    phi = phi[np.where(np.isnan(phi) == False)]

    def grad(x, y, a, b):
        return a * x + b * y

    def rms_grad(p, x, y):
        a = p[0]
        b = p[1]

        phi_grad = grad(x, y, a, b)
        return np.sum(np.sqrt((phi - phi_grad) ** 2))

    p0 = [0.1, 0.1]
    p = scipy.optimize.minimize(rms_grad, p0, args=(x, y))
    phi_model = grad(x, y, p.x[0], p.x[1])

    return x, y, phi, fftd


# Applying reference plane and receiver feed
# corrections.
def apply_ref_pl_and_rx_corrections(beam_in, az, el, xyz_tow, xyz_ref, xyz_ap, k):

    pl_ap = np.zeros(len(az))
    pl_rp = np.zeros(len(az))

    i = int(0)
    while i < len(az):
        # Rotated location of aperture
        loc_ap = oa.rotate_azel(xyz_ap, az[i], el[i])
        pl_ap[i] = distance.euclidean(loc_ap, xyz_tow)

        # Rotated location of reference receiver
        loc_rp = oa.rotate_azel(xyz_ref, az[i], el[i])
        pl_rp[i] = distance.euclidean(loc_rp, xyz_tow)
        i += 1

    pl_ap = np.reshape(pl_ap, (int(np.sqrt(len(az))), int(np.sqrt(len(az)))))
    pl_rp = np.reshape(pl_rp, (int(np.sqrt(len(az))), int(np.sqrt(len(az)))))

    # Convert the path length to a phase correction

    ap_phase_cor = np.exp(-imu * pl_ap * k)
    rp_phase_cor = np.exp(imu * pl_rp * k)

    # Apply the correction
    out = beam_in * ap_phase_cor * rp_phase_cor

    return out


# Main function which takes in a far-field measurement
# and analyzes, turning measurement into aperture field.
def analyze_holography(
    dat, geo_struct, plotting, reference_motion_correction, epsilon_terms, shift
):
    pan_mod2 = pm.panel_model_from_adjuster_offsets(2, 0, 0, 0)  # mm
    pan_mod1 = pm.panel_model_from_adjuster_offsets(1, 0, 0, 0)  # mm

    # constants
    lambda_ = geo_struct.lambda_
    k = 2.0 * np.pi / lambda_
    diam = 6  # diameter of mirrors
    # Position of tower
    rx = np.array([geo_struct.rx_x, geo_struct.rx_y, geo_struct.rx_z])
    xyz_tow = np.array([geo_struct.x_tow, geo_struct.y_tow, geo_struct.z_tow])

    # Position of reference receiver
    xyz_ref = np.array([geo_struct.x_phref, geo_struct.y_phref, geo_struct.z_phref])

    # Position of aperture
    xyz_ap = np.array([geo_struct.x_ap, geo_struct.y_ap, geo_struct.z_ap])

    # Scale correction due to parralax
    plx_cor_x = 1.0 + (xyz_ap[2] / np.sqrt(np.sum(xyz_tow ** 2)))
    plx_cor_y = 1.0 + (xyz_ap[2] / np.sqrt(np.sum(xyz_tow ** 2)))

    ## Break out the data  (which is complex)
    nx = len(dat[:, 0])
    ny = len(dat[:, 1])
    ndat = len(dat[0, :])
    azi = dat[:, 0]
    az = azi
    eli = dat[:, 1]
    el = eli
    beam = dat[:, 2] + imu * dat[:, 3]

    AZ = np.reshape(az, (int(np.sqrt(len(el))), int(np.sqrt(len(el)))))
    EL = np.reshape(el, (int(np.sqrt(len(el))), int(np.sqrt(len(el)))))
    beam = np.reshape(beam, (int(np.sqrt(len(el))), int(np.sqrt(len(el)))))

    geo_struct.de_ang = abs(AZ[0, 0] - AZ[1, 0])
    geo_struct.N_scan = int(len(AZ) / 2)

    if plotting == 1:
        plt.figure(figsize=(3, 3))
        plt.title("Az vs. El")
        plt.plot(az, el, ".")
        plt.axis("equal")
        plt.xlim(np.min(el), np.max(el))
        plt.xlim(np.min(az), np.max(az))
        plt.axis("equal")
        plt.xlabel("Elevation [rad]")
        plt.ylabel("Azimuthal [rad]")
        plt.show()

        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        plt.title("Amplitude [dB]")
        plt.pcolormesh(
            AZ, EL, 20 * np.log10(abs(beam) / np.max(abs(beam))), shading="auto"
        )
        plt.colorbar()
        plt.axis("equal")
        plt.xlabel("Elevation [rad]")
        plt.ylabel("Azimuthal [rad]")
        plt.xlim(np.min(AZ), np.max(AZ))
        plt.ylim(np.min(EL), np.max(EL))

        plt.subplot(1, 2, 2)
        plt.title("Phase [rad]")
        Z = np.arctan2(np.imag(beam), np.real(beam))
        plt.pcolormesh(AZ, EL, Z, shading="auto")
        plt.colorbar()
        plt.axis("equal")
        plt.xlabel("Elevation [rad]")
        plt.ylabel("Azimuthal [rad]")
        plt.xlim(np.min(AZ), np.max(AZ))
        plt.ylim(np.min(EL), np.max(EL))
        plt.show()

    # Geometric corrections for motion of telescope
    # and reference receiver.
    if reference_motion_correction != 0:
        beam_ref_rx_cor = apply_ref_pl_and_rx_corrections(
            beam, az, el, xyz_tow, xyz_ref, xyz_ap, k
        )
        if plotting == 1:
            plt.figure(figsize=(3, 3))
            plt.title("Phase after RX Correction")
            Z = np.arctan2(np.imag(beam_ref_rx_cor), np.real(beam_ref_rx_cor))
            plt.pcolormesh(AZ, EL, Z, shading="auto")
            plt.xlabel("Elevation [rad]")
            plt.ylabel("Azimuthal [rad]")
            plt.colorbar()
            plt.axis("equal")
            plt.show()
    else:
        beam_ref_rx_cor = beam

    # Get Aperture Fields: FFT beam to aperture fields
    fftd = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(beam_ref_rx_cor)))

    if epsilon_terms != 0:

        u_fun = AZ - np.mean(AZ)
        v_fun = EL - np.mean(EL)

        ## u_integral
        temp = np.zeros(np.shape(AZ), dtype=complex)
        temp = beam_ref_rx_cor * u_fun
        fftd_temp = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(temp)))
        fftd_temp = np.transpose(fftd_temp)
        fftd_temp = np.flip(fftd_temp, axis=0)
        u_integral = fftd_temp

        ## v_integral
        temp = np.zeros(np.shape(AZ), dtype=complex)
        temp = beam_ref_rx_cor * v_fun
        fftd_temp = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(temp)))
        fftd_temp = np.transpose(fftd_temp)
        fftd_temp = np.flip(fftd_temp, axis=0)
        v_integral = fftd_temp

        ## uu_integral
        temp = np.zeros(np.shape(AZ), dtype=complex)
        temp = beam_ref_rx_cor * u_fun * u_fun
        fftd_temp = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(temp)))
        fftd_temp = np.transpose(fftd_temp)
        fftd_temp = np.flip(fftd_temp, axis=0)
        uu_integral = fftd_temp

        ## vv_integral
        temp = np.zeros(np.shape(AZ), dtype=complex)
        temp = beam_ref_rx_cor * v_fun * v_fun
        fftd_temp = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(temp)))
        fftd_temp = np.transpose(fftd_temp)
        fftd_temp = np.flip(fftd_temp, axis=0)
        vv_integral = fftd_temp

        ## uv_integral
        temp = np.zeros(np.shape(AZ), dtype=complex)
        temp = beam_ref_rx_cor * u_fun * v_fun
        fftd_temp = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(temp)))
        fftd_temp = np.transpose(fftd_temp)
        fftd_temp = np.flip(fftd_temp, axis=0)
        uv_integral = fftd_temp

    # Get spatial coordinates

    delta_th = abs(np.max(az) - np.min(az)) / len(az)  # increment in azimuthal angle
    alpha = (lambda_ / diam) / delta_th  # increment in x
    delta_x = alpha * diam / nx  # spatial coordinates conversion

    delta_th = abs(np.max(el) - np.min(el)) / len(el)  # increment in azimuthal angle
    beta = (lambda_ / diam) / delta_th
    delta_y = beta * diam / ny

    x = np.linspace(0, int(np.sqrt(len(az))), int(np.sqrt(len(az)))) * delta_x
    y = np.linspace(0, int(np.sqrt(len(az))), int(np.sqrt(len(az)))) * delta_y
    y = y - np.mean(y)
    x = x - np.mean(x)

    x = x / plx_cor_x
    y = y / plx_cor_y
    x, y = np.meshgrid(x, y)

    tow_di = np.sqrt((xyz_tow[0]) ** 2.0 + (xyz_tow[1]) ** 2.0 + (xyz_tow[2]) ** 2.0)
    tow_th = np.arctan(xyz_tow[0] / xyz_tow[2])

    phi = oa.do_unwrap(np.arctan2(np.imag(fftd), np.real(fftd)))

    x_temp = x * np.cos(np.pi / 2) - y * np.sin(np.pi / 2)
    y_temp = x * np.sin(np.pi / 2) + y * np.cos(np.pi / 2)

    x = x_temp
    y = y_temp

    # Optional for including epsilon terms
    if epsilon_terms != 0:
        print("Using epsilon terms.")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(
            x, y, 20 * np.log10(abs(fftd) / np.max(abs(fftd))), shading="auto"
        )
        plt.colorbar()
        plt.axis("equal")
        plt.subplot(1, 2, 2)
        plt.pcolormesh(
            x, y, oa.do_unwrap(np.arctan2(np.imag(fftd), np.real(fftd))), shading="auto"
        )
        plt.colorbar()
        plt.axis("equal")
        evaluated_epsilon_correcton = (
            -imu
            * k
            * (
                (u_integral / (2.0 * tow_di ** 2.0) * x * (x ** 2.0 + y ** 2.0))
                + (v_integral / (2.0 * tow_di ** 2.0) * y * (x ** 2.0 + y ** 2.0))
                - (uu_integral / (2.0 * tow_di) * x ** 2.0)
                - (vv_integral / (2.0 * tow_di) * y ** 2.0)
                - (uv_integral / (2.0 * tow_di) * x * y)
            )
        )

        plt.figure(figsize=(12, 7))

        plt.subplot(2, 3, 1)
        plt.title(r"$ -\kappa (u_{int}/ 2 d_{tow}^2) x(x^2 + y^2)$")
        z = -imu * k * (u_integral / (2.0 * tow_di ** 2.0) * x * (x ** 2.0 + y ** 2.0))
        plt.pcolormesh(x, y, np.arctan2(np.imag(z), np.real(z)), shading="auto")
        plt.colorbar()
        plt.axis("equal")

        plt.subplot(2, 3, 2)
        plt.title(r"$ -\kappa (v_{int}/ 2 d_{tow}^2) y(x^2 + y^2)$")
        z = -imu * k * (v_integral / (2.0 * tow_di ** 2.0) * y * (x ** 2.0 + y ** 2.0))
        plt.pcolormesh(x, y, np.arctan2(np.imag(z), np.real(z)), shading="auto")
        plt.colorbar()
        plt.axis("equal")

        plt.subplot(2, 3, 3)
        plt.title(r"$ -\kappa (uu_{int}/ 2 d_{tow}) x^2 $")
        z = -imu * k * (uu_integral / (2.0 * tow_di) * x ** 2.0)
        plt.pcolormesh(x, y, np.arctan2(np.imag(z), np.real(z)), shading="auto")
        plt.colorbar()
        plt.axis("equal")

        plt.subplot(2, 3, 4)
        plt.title(r"$ -\kappa (vv_{int}/ 2 d_{tow}) y^2$")
        z = -imu * k * (vv_integral / (2.0 * tow_di) * y ** 2.0)
        plt.pcolormesh(x, y, np.arctan2(np.imag(z), np.real(z)), shading="auto")
        plt.colorbar()
        plt.axis("equal")

        plt.subplot(2, 3, 5)
        plt.title(r"$ -\kappa (uv_{int}/ 2 d_{tow}) x y$")
        z = -imu * k * (uv_integral / (2.0 * tow_di) * x * y)
        plt.pcolormesh(x, y, np.arctan2(np.imag(z), np.real(z)), shading="auto")
        plt.colorbar()
        plt.axis("equal")

        plt.subplot(2, 3, 6)
        plt.title(r"Sum of all terms")
        z = evaluated_epsilon_correcton
        plt.pcolormesh(x, y, np.arctan2(np.imag(z), np.real(z)), shading="auto")
        plt.colorbar()
        plt.axis("equal")

        plt.show()

        fftd += evaluated_epsilon_correcton

    # This class is an optional output
    # for the user to return a series of
    # specific parameters (if desired).
    class guessed_geometery:
        x0 = 0
        y0 = 0
        rx_x = 0
        rx_y = 0
        rx_z = 0
        tower_dist = 1e6

        k = 2.0 * np.pi / geo_struct.lambda_

    th = np.linspace(-np.pi / 2 - 0.3, -np.pi / 2 + 0.3, 150)
    ph = np.linspace(np.pi / 2 - 0.3, np.pi / 2 + 0.3, 150)

    rxmirror = af.ray_mirror_pts(rx, geo_struct, th, ph)
    out = af.aperature_fields_from_panel_model(
        pan_mod1, pan_mod2, rx, geo_struct, th, ph, rxmirror
    )

    if shift[0] == "y":
        x_temp, y_temp, phi = pf.pannel_mask(x, y + shift[1], phi, out, 0)
        x_temp, y_temp, fftd = pf.pannel_mask(x, y + shift[1], fftd, out, 0)
    elif shift[0] == "x":
        x_temp, y_temp, phi = pf.pannel_mask(x + shift[1], y, phi, out, 0)
        x_temp, y_temp, fftd = pf.pannel_mask(x + shift[1], y, fftd, out, 0)
    else:
        x_temp, y_temp, phi = pf.pannel_mask(x + shift[1], y + shift[2], phi, out, 0)
        x_temp, y_temp, fftd = pf.pannel_mask(x + shift[1], y + shift[2], fftd, out, 0)

    x, y, phi, fftd = remove_grad(x_temp, y_temp, phi, fftd)
    x, y, phi, fftd = remove_grad(x, y, phi, fftd)
    return x, y, phi / 2, fftd, guessed_geometery


# Function takes a measurement with specified
# adjuster offsets
def take_measurement(adj_off1, adj_off2, rep, tele_geo, rxmirror):

    # Define telescope geometry and adjuster positions on each mirror:
    rx = np.array([tele_geo.rx_x, tele_geo.rx_y, tele_geo.rx_z])

    # Define panels on M1 and M2. Here you can define the
    # magnitude of the adjuster offsets on each mirror:
    pan_mod2 = pm.panel_model_from_adjuster_offsets(
        2, adj_off2 * 1e3, 1, 0
    )  # Panel Model on M2
    pan_mod1 = pm.panel_model_from_adjuster_offsets(
        1, adj_off1 * 1e3, 1, 0
    )  # Panel Model on M1

    # Set offsets of Receiver Feed (RX):
    # Define FOV of RX. In other words, define directions of
    # outgoing rays from the RX.
    th = np.linspace(-np.pi / 2 - 0.28, -np.pi / 2 + 0.28, tele_geo.N_scan)
    ph = np.linspace(np.pi / 2 - 0.28, np.pi / 2 + 0.28, tele_geo.N_scan)

    # Define the path of the rays from the RX to the aperture plane
    rxmirror = af.ray_mirror_pts(rx, tele_geo, th, ph)
    out = af.aperature_fields_from_panel_model(
        pan_mod1, pan_mod2, rx, tele_geo, th, ph, rxmirror
    )

    beam = ff.far_field_sim(out, tele_geo, rx)

    meas_file = "/data/chesmore/sim_out/sim_err_rep" + str(rep) + "_.txt"
    np.savetxt(
        meas_file,
        np.c_[
            np.real(beam[0, :]),
            np.real(beam[1, :]),
            np.real(beam[2, :]),
            np.imag(beam[2, :]),
        ],
    )

    return meas_file
