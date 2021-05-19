"""
Miscellaneous aperture field analysis functions.

Grace E. Chesmore
2021
"""

import ap_fitting as afit
import far_field as ff
import matplotlib.pyplot as plt
import numpy as np
import pan_mod as pm
from pan_mod import *
from scipy import optimize
from tele_geo import *

y_cent_m1 = -7201.003729431267

adj_pos_m1, adj_pos_m2 = pm.get_single_vert_adj_positions()


def ray_pts(P_rx, tele_geo):

    # Step 1:  grid the plane of rays shooting out of receiver feed
    theta = np.linspace(-np.pi / 2 - 0.3, -np.pi / 2 + 0.3, 300)
    phi = np.linspace(np.pi / 2 - 0.3, np.pi / 2 + 0.3, 300)
    theta, phi = np.meshgrid(theta, phi)
    theta = np.ravel(theta)
    phi = np.ravel(phi)

    th2 = tele_geo.th2
    z_ap = tele_geo.z_ap * 1e3
    horn_fwhp = tele_geo.th_fwhp
    N_linear = tele_geo.N_scan
    focal = (
        tele_geo.F_2
    )  # distance from focal plane center to center of secondary mirror

    n_pts = len(theta)
    out = np.zeros((6, n_pts))

    for ii in range(n_pts):

        # Define the direction of the outgoing ray:
        th = theta[ii]
        ph = phi[ii]
        r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

        alpha = r_hat[0]
        beta = r_hat[1]
        gamma = r_hat[2]

        # Receiver feed position [mm] (in telescope reference frame):
        x_0 = P_rx[0]
        y_0 = P_rx[1]
        z_0 = P_rx[2]

        def root_z2(t):

            x = x_0 + alpha * t
            y = y_0 + beta * t
            z = z_0 + gamma * t

            xm2, ym2, zm2 = tele_into_m2(
                x, y, z
            )  # Convert ray's endpoint into M2 coordinates
            z_m2 = z2(xm2, ym2)  # Z of mirror in M2 coordinates
            # Add perturbed mirror here:
            root = zm2 - z_m2
            return root

        t_m2 = optimize.brentq(root_z2, focal + 2000, focal + 13000)

        # Location of where ray hits M2
        x_m2 = x_0 + alpha * t_m2
        y_m2 = y_0 + beta * t_m2
        z_m2 = z_0 + gamma * t_m2

        # Using x and y in M2 coordiantes, find the z err:
        P_m2 = np.array([x_m2, y_m2, z_m2])

        ###### in M2 coordinates ##########################
        x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m2(x_m2, y_m2, z_m2)  # P_m2 temp
        x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m2(x_0, y_0, z_0)  # P_rx temp
        norm = d_z2(x_m2_temp, y_m2_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2))
        tan_rx_m2 = vec_rx_m2 / dist_rx_m2

        # Outgoing ray
        tan_og = tan_rx_m2 - 2 * np.dot(np.sum(np.dot(tan_rx_m2, N_hat)), N_hat)

        # Transform back to telescope cordinates
        N_hat_t = np.zeros(3)

        N_hat_x_temp = N_hat[0] * np.cos(np.pi) - N_hat[1] * np.sin(np.pi)
        N_hat_y_temp = N_hat[0] * np.sin(np.pi) + N_hat[1] * np.cos(np.pi)
        N_hat_z_temp = N_hat[2]

        N_hat_t[0] = N_hat_x_temp
        N_hat_t[1] = N_hat_y_temp * np.cos(th2) - N_hat_z_temp * np.sin(th2)
        N_hat_t[2] = N_hat_y_temp * np.sin(th2) + N_hat_z_temp * np.cos(th2)

        tan_rx_m2_t = np.zeros(3)

        tan_rx_m2_x_temp = tan_rx_m2[0] * np.cos(np.pi) - tan_rx_m2[1] * np.sin(np.pi)
        tan_rx_m2_y_temp = tan_rx_m2[0] * np.sin(np.pi) + tan_rx_m2[1] * np.cos(np.pi)
        tan_rx_m2_z_temp = tan_rx_m2[2]

        tan_rx_m2_t[0] = tan_rx_m2_x_temp
        tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(th2) - tan_rx_m2_z_temp * np.sin(th2)
        tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(th2) + tan_rx_m2_z_temp * np.cos(th2)

        tan_og_t = np.zeros(3)
        tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
        tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
        tan_og_z_temp = tan_og[2]

        tan_og_t[0] = tan_og_x_temp
        tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
        tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)
        ##################################################

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_z1(t):
            x = P_m2[0] + alpha * t
            y = P_m2[1] + beta * t
            z = P_m2[2] + gamma * t
            xm1, ym1, zm1 = tele_into_m1(
                x, y, z
            )  # take ray end coordinates and convert to M1 coordinates
            z_m1 = z1(xm1, ym1)  # Z of mirror 1 in M1 coordinates
            root = zm1 - z_m1
            return root

        t_m1 = optimize.brentq(root_z1, 1000, 18000)

        # Location of where ray hits M1
        x_m1 = P_m2[0] + alpha * t_m1
        y_m1 = P_m2[1] + beta * t_m1
        z_m1 = P_m2[2] + gamma * t_m1
        P_m1 = np.array([x_m1, y_m1, z_m1])

        # Write out
        out[0, ii] = x_m2
        out[1, ii] = y_m2
        out[2, ii] = z_m2

        out[3, ii] = x_m1
        out[4, ii] = y_m1
        out[5, ii] = z_m1

    return out


x_m2 = []
y_m2 = []
row_m2 = []
col_m2 = []


def panel_pts(panel_model1, panel_model2, P_rx, tele_geo):
    th2 = tele_geo.th2
    z_ap = tele_geo.z_ap * 1e3
    horn_fwhp = tele_geo.th_fwhp
    focal = tele_geo.F_2
    # Step 1:  grid the plane of rays shooting out of receiver feed
    N_linear = tele_geo.N_scan
    col_m2 = adj_pos_m2[0]
    row_m2 = adj_pos_m2[1]
    x_adj_m2 = adj_pos_m2[4]
    y_adj_m2 = adj_pos_m2[3]

    col_m1 = adj_pos_m1[0]
    row_m1 = adj_pos_m1[1]
    x_adj_m1 = adj_pos_m1[2]
    y_adj_m1 = adj_pos_m1[3]

    rxmirror = ray_pts(P_rx, tele_geo)
    theta = np.linspace(-np.pi / 2 - 0.3, -np.pi / 2 + 0.3, 300)
    phi = np.linspace(np.pi / 2 - 0.3, np.pi / 2 + 0.3, 300)
    theta, phi = np.meshgrid(theta, phi)
    theta = np.ravel(theta)
    phi = np.ravel(phi)

    x_panm_m2 = np.reshape(
        rxmirror[0, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )
    y_panm_m2 = np.reshape(
        rxmirror[2, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )

    x_panm_m1 = np.reshape(
        rxmirror[3, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )
    y_panm_m1 = np.reshape(
        rxmirror[4, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )

    pan_id_m2 = identify_panel(x_panm_m2, y_panm_m2, x_adj_m2, y_adj_m2, col_m2, row_m2)
    pan_id_m1 = identify_panel(
        x_panm_m1, y_panm_m1 - y_cent_m1, x_adj_m1, y_adj_m1, col_m1, row_m1
    )

    row_panm_m2 = np.ravel(pan_id_m2[0, :, :])
    col_panm_m2 = np.ravel(pan_id_m2[1, :, :])
    row_panm_m1 = np.ravel(pan_id_m1[0, :, :])
    col_panm_m1 = np.ravel(pan_id_m1[1, :, :])

    # Step 2: calculate the position + local surface normal for the dish
    n_pts = len(theta)
    out = np.zeros((17, n_pts))

    for ii in range(n_pts):
        i_row_m2 = row_panm_m2[ii]
        i_col_m2 = col_panm_m2[ii]
        i_panm = np.where(
            (panel_model2[0, :] == i_row_m2) & (panel_model2[1, :] == i_col_m2)
        )

        if len(i_panm[0]) != 0:

            # Break out the a,b,c,d,e,f & x0 / y0 & theta_rot ==>alpha, beta parametres
            th = theta[ii]
            ph = phi[ii]
            r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
            alpha = r_hat[0]
            beta = r_hat[1]
            gamma = r_hat[2]

            # Receiver feed position [mm] (in telescope reference frame):
            x_0 = P_rx[0]
            y_0 = P_rx[1]
            z_0 = P_rx[2]

            a = panel_model2[2, i_panm]
            b = panel_model2[3, i_panm]
            c = panel_model2[4, i_panm]
            d = panel_model2[5, i_panm]
            e = panel_model2[6, i_panm]
            f = panel_model2[7, i_panm]
            x0 = panel_model2[8, i_panm]
            y0 = panel_model2[9, i_panm]

            def root_z2(t):
                x = x_0 + alpha * t
                y = y_0 + beta * t
                z = z_0 + gamma * t
                xm2, ym2, zm2 = tele_into_m2(
                    x, y, z
                )  # Convert ray's endpoint into M2 coordinates

                if P_rx[2] != 0:
                    z /= np.cos(np.arctan(1 / 3))
                xm2_err, ym2_err, zm2_err = tele_into_m2(
                    x, y, z
                )  # Convert ray's endpoint into M2 coordinates

                x_temp = xm2_err * np.cos(np.pi) + zm2_err * np.sin(np.pi)
                y_temp = ym2_err
                z_temp = -xm2_err * np.sin(np.pi) + zm2_err * np.cos(np.pi)

                xpc = x_temp - x0
                ypc = y_temp - y0

                z_err = (
                    a
                    + b * xpc
                    + c * (ypc)
                    + d * (xpc ** 2 + ypc ** 2)
                    + e * (xpc * ypc)
                )
                z_err = z_err[0][0]

                z_m2 = z2(xm2, ym2)  # Z of mirror in M2 coordinates

                root = zm2 - (z_m2 + z_err)
                return root

            t_m2 = optimize.brentq(root_z2, focal + 4000, focal + 12000)

            # Location of where ray hits M2
            x_m2 = x_0 + alpha * t_m2
            y_m2 = y_0 + beta * t_m2
            z_m2 = z_0 + gamma * t_m2

            # Using x and y in M2 coordiantes, find the z err:

            P_m2 = np.array([x_m2, y_m2, z_m2])

            ###### in M2 coordinates ##########################
            x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m2(
                x_m2, y_m2, z_m2
            )  # P_m2 temp
            x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m2(x_0, y_0, z_0)  # P_rx temp
            norm = d_z2(x_m2_temp, y_m2_temp)
            norm_temp = np.array([-norm[0], -norm[1], 1])
            N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
            vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array(
                [x_rx_temp, y_rx_temp, z_rx_temp]
            )
            dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2))
            tan_rx_m2 = vec_rx_m2 / dist_rx_m2

            # Outgoing ray
            tan_og = tan_rx_m2 - 2 * np.dot(np.sum(np.dot(tan_rx_m2, N_hat)), N_hat)

            # Transform back to telescope cordinates
            N_hat_t = np.zeros(3)

            N_hat_x_temp = N_hat[0] * np.cos(np.pi) - N_hat[1] * np.sin(np.pi)
            N_hat_y_temp = N_hat[0] * np.sin(np.pi) + N_hat[1] * np.cos(np.pi)
            N_hat_z_temp = N_hat[2]

            N_hat_t[0] = N_hat_x_temp
            N_hat_t[1] = N_hat_y_temp * np.cos(th2) - N_hat_z_temp * np.sin(th2)
            N_hat_t[2] = N_hat_y_temp * np.sin(th2) + N_hat_z_temp * np.cos(th2)

            tan_rx_m2_t = np.zeros(3)

            tan_rx_m2_x_temp = tan_rx_m2[0] * np.cos(np.pi) - tan_rx_m2[1] * np.sin(
                np.pi
            )
            tan_rx_m2_y_temp = tan_rx_m2[0] * np.sin(np.pi) + tan_rx_m2[1] * np.cos(
                np.pi
            )
            tan_rx_m2_z_temp = tan_rx_m2[2]

            tan_rx_m2_t[0] = tan_rx_m2_x_temp
            tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(th2) - tan_rx_m2_z_temp * np.sin(
                th2
            )
            tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(th2) + tan_rx_m2_z_temp * np.cos(
                th2
            )

            tan_og_t = np.zeros(3)
            tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
            tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
            tan_og_z_temp = tan_og[2]

            tan_og_t[0] = tan_og_x_temp
            tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
            tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)
            ##################################################

            alpha = tan_og_t[0]
            beta = tan_og_t[1]
            gamma = tan_og_t[2]

            i_row_m1 = row_panm_m1[ii]
            i_col_m1 = col_panm_m1[ii]
            i_panm = np.where(
                (panel_model1[0, :] == i_row_m1) & (panel_model1[1, :] == i_col_m1)
            )
            if len(i_panm[0]) != 0:
                a = panel_model1[2, i_panm]
                b = panel_model1[3, i_panm]
                c = panel_model1[4, i_panm]
                d = panel_model1[5, i_panm]
                e = panel_model1[6, i_panm]
                f = panel_model1[7, i_panm]
                x0 = panel_model1[8, i_panm]
                y0 = panel_model1[9, i_panm]

                def root_z1(t):
                    x = P_m2[0] + alpha * t
                    y = P_m2[1] + beta * t
                    z = P_m2[2] + gamma * t
                    xm1, ym1, zm1 = tele_into_m1(
                        x, y, z
                    )  # take ray end coordinates and convert to M1 coordinates

                    xm1_err, ym1_err, zm1_err = tele_into_m1(x, y, z)

                    x_temp = xm1_err * np.cos(np.pi) + zm1_err * np.sin(np.pi)
                    y_temp = ym1_err
                    z_temp = -xm1_err * np.sin(np.pi) + zm1_err * np.cos(np.pi)

                    xpc = x_temp - x0
                    ypc = y_temp - y0

                    z_err = (
                        a
                        + b * xpc
                        + c * (ypc)
                        + d * (xpc ** 2 + ypc ** 2)
                        + e * (xpc * ypc)
                    )
                    z_err = z_err[0][0]
                    z_m1 = z1(xm1, ym1)  # Z of mirror 1 in M1 coordinates
                    root = zm1 - (z_m1 + z_err)
                    return root

                t_m1 = optimize.brentq(root_z1, 1000, 18000)

                # Location of where ray hits M1
                x_m1 = P_m2[0] + alpha * t_m1
                y_m1 = P_m2[1] + beta * t_m1
                z_m1 = P_m2[2] + gamma * t_m1
                P_m1 = np.array([x_m1, y_m1, z_m1])

                ###### in M1 cordinates ##########################
                x_m1_temp, y_m1_temp, z_m1_temp = tele_into_m1(
                    x_m1, y_m1, z_m1
                )  # P_m2 temp
                x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m1(
                    P_m2[0], P_m2[1], P_m2[2]
                )  # P_rx temp
                norm = d_z1(x_m1_temp, y_m1_temp)
                norm_temp = np.array(
                    [-norm[0], -norm[1], 1]
                )  # this is where I think i'm messing up
                N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
                vec_m2_m1 = np.array([x_m1_temp, y_m1_temp, z_m1_temp]) - np.array(
                    [x_m2_temp, y_m2_temp, z_m2_temp]
                )
                dist_m2_m1 = np.sqrt(np.sum(vec_m2_m1 ** 2))
                tan_m2_m1 = vec_m2_m1 / dist_m2_m1

                # Outgoing ray
                tan_og = tan_m2_m1 - 2 * np.dot(np.sum(np.dot(tan_m2_m1, N_hat)), N_hat)

                # Transform back to telescope cordinates
                N_hat_t = np.zeros(3)
                tan_m2_m1_t = np.zeros(3)
                tan_og_t = np.zeros(3)

                N_x_temp = N_hat[0] * np.cos(np.pi) + N_hat[2] * np.sin(np.pi)
                N_y_temp = N_hat[1]
                N_z_temp = -N_hat[0] * np.sin(np.pi) + N_hat[2] * np.cos(np.pi)
                N_hat_t[0] = N_x_temp
                N_hat_t[1] = N_y_temp * np.cos(tele_geo.th_1) - N_z_temp * np.sin(
                    tele_geo.th_1
                )
                N_hat_t[2] = N_y_temp * np.sin(tele_geo.th_1) + N_z_temp * np.cos(
                    tele_geo.th_1
                )

                tan_m2_m1_x_temp = tan_m2_m1[0] * np.cos(np.pi) + tan_m2_m1[2] * np.sin(
                    np.pi
                )
                tan_m2_m1_y_temp = tan_m2_m1[1]
                tan_m2_m1_z_temp = -tan_m2_m1[0] * np.sin(np.pi) + tan_m2_m1[
                    2
                ] * np.cos(np.pi)
                tan_m2_m1_t[0] = tan_m2_m1_x_temp
                tan_m2_m1_t[1] = tan_m2_m1_y_temp * np.cos(
                    tele_geo.th_1
                ) - tan_m2_m1_z_temp * np.sin(tele_geo.th_1)
                tan_m2_m1_t[2] = tan_m2_m1_y_temp * np.sin(
                    tele_geo.th_1
                ) + tan_m2_m1_z_temp * np.cos(tele_geo.th_1)

                tan_og_x_temp = tan_og[0] * np.cos(np.pi) + tan_og[2] * np.sin(np.pi)
                tan_og_y_temp = tan_og[1]
                tan_og_z_temp = -tan_og[0] * np.sin(np.pi) + tan_og[2] * np.cos(np.pi)
                tan_og_t[0] = tan_og_x_temp
                tan_og_t[1] = tan_og_y_temp * np.cos(
                    tele_geo.th_1
                ) - tan_og_z_temp * np.sin(tele_geo.th_1)
                tan_og_t[2] = tan_og_y_temp * np.sin(
                    tele_geo.th_1
                ) + tan_og_z_temp * np.cos(tele_geo.th_1)

                ##################################################

                dist_m1_ap = abs((z_ap - P_m1[2]) / tan_og_t[2])

                pos_ap = P_m1 + dist_m1_ap * tan_og_t

                # Write out
                out[0, ii] = x_m2
                out[1, ii] = y_m2
                out[2, ii] = z_m2

                out[3, ii] = pos_ap[0]
                out[4, ii] = pos_ap[1]
                out[5, ii] = pos_ap[2]

                out[6, ii] = i_row_m2
                out[7, ii] = i_col_m2

                out[8, ii] = i_row_m1
                out[9, ii] = i_col_m1

    out_new = []
    for jj in range(len(out[:, 0])):
        out_new.append(out[jj, :][np.where(out[4, :] < 0)])

    return out_new


def ray_mirror_pts(P_rx, tele_geo, theta, phi):

    theta, phi = np.meshgrid(theta, phi)
    theta = np.ravel(theta)
    phi = np.ravel(phi)

    # Read in telescope geometry values
    th2 = tele_geo.th2
    z_ap = tele_geo.z_ap * 1e3
    horn_fwhp = tele_geo.th_fwhp
    N_linear = tele_geo.N_scan
    focal = tele_geo.F_2

    n_pts = len(theta)
    out = np.zeros((6, n_pts))

    for ii in range(n_pts):

        # Define the outgoing ray's direction
        th = theta[ii]
        ph = phi[ii]
        r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
        alpha = r_hat[0]
        beta = r_hat[1]
        gamma = r_hat[2]

        # Receiver feed position [mm] in telescope r.f.
        x_0 = P_rx[0]
        y_0 = P_rx[1]
        z_0 = P_rx[2]

        # Use a root finder to find where the ray intersects with M2
        def root_z2(t):

            # Endpoint of ray:
            x = x_0 + alpha * t
            y = y_0 + beta * t
            z = z_0 + gamma * t

            # Convert to M2 r.f.
            xm2, ym2, zm2 = tele_into_m2(x, y, z)

            # Z of mirror in M2 r.f.
            z_m2 = z2(xm2, ym2)
            return zm2 - z_m2

        t_m2 = optimize.brentq(root_z2, focal + 1e3, focal + 13e3)

        # Endpoint of ray:
        x_m2 = x_0 + alpha * t_m2
        y_m2 = y_0 + beta * t_m2
        z_m2 = z_0 + gamma * t_m2
        P_m2 = np.array([x_m2, y_m2, z_m2])

        ########## M2 r.f ###########################################################

        x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m2(P_m2[0], P_m2[1], P_m2[2])
        x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m2(x_0, y_0, z_0)

        # Normal vector of ray on M2
        norm = d_z2(x_m2_temp, y_m2_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        # Normalized vector from RX to M2
        vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2))
        tan_rx_m2 = vec_rx_m2 / dist_rx_m2

        # Vector of outgoing ray
        tan_og = tan_rx_m2 - 2 * np.dot(np.sum(np.dot(tan_rx_m2, N_hat)), N_hat)

        # Transform back to telescope cordinates
        N_hat_t = np.zeros(3)

        N_hat_x_temp = N_hat[0] * np.cos(np.pi) - N_hat[1] * np.sin(np.pi)
        N_hat_y_temp = N_hat[0] * np.sin(np.pi) + N_hat[1] * np.cos(np.pi)
        N_hat_z_temp = N_hat[2]

        N_hat_t[0] = N_hat_x_temp
        N_hat_t[1] = N_hat_y_temp * np.cos(th2) - N_hat_z_temp * np.sin(th2)
        N_hat_t[2] = N_hat_y_temp * np.sin(th2) + N_hat_z_temp * np.cos(th2)

        tan_rx_m2_t = np.zeros(3)

        tan_rx_m2_x_temp = tan_rx_m2[0] * np.cos(np.pi) - tan_rx_m2[1] * np.sin(np.pi)
        tan_rx_m2_y_temp = tan_rx_m2[0] * np.sin(np.pi) + tan_rx_m2[1] * np.cos(np.pi)
        tan_rx_m2_z_temp = tan_rx_m2[2]

        tan_rx_m2_t[0] = tan_rx_m2_x_temp
        tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(th2) - tan_rx_m2_z_temp * np.sin(th2)
        tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(th2) + tan_rx_m2_z_temp * np.cos(th2)

        tan_og_t = np.zeros(3)
        tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
        tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
        tan_og_z_temp = tan_og[2]

        tan_og_t[0] = tan_og_x_temp
        tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
        tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)

        ########## Tele. r.f ###########################################################

        # Vector of outgoing ray:
        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        # Use a root finder to find where the ray intersects with M1
        def root_z1(t):

            # Endpoint of ray:
            x = P_m2[0] + alpha * t
            y = P_m2[1] + beta * t
            z = P_m2[2] + gamma * t

            # Convert to M1 r.f.
            xm1, ym1, zm1 = tele_into_m1(x, y, z)

            # Z of mirror in M1 r.f.
            z_m1 = z1(xm1, ym1)
            return zm1 - z_m1

        t_m1 = optimize.brentq(root_z1, 50, 22000)

        # Endpoint of ray:
        x_m1 = P_m2[0] + alpha * t_m1
        y_m1 = P_m2[1] + beta * t_m1
        z_m1 = P_m2[2] + gamma * t_m1
        P_m1 = np.array([x_m1, y_m1, z_m1])

        # Write out
        out[0, ii] = x_m2
        out[1, ii] = y_m2
        out[2, ii] = z_m2

        out[3, ii] = x_m1
        out[4, ii] = y_m1
        out[5, ii] = z_m1

    return out


def aperature_fields_from_panel_model(
    panel_model1, panel_model2, P_rx, tele_geo, theta, phi, rxmirror
):

    theta, phi = np.meshgrid(theta, phi)
    theta = np.ravel(theta)
    phi = np.ravel(phi)

    th2 = tele_geo.th2
    z_ap = tele_geo.z_ap * 1e3
    horn_fwhp = tele_geo.th_fwhp
    focal = tele_geo.F_2
    # Step 1:  grid the plane of rays shooting out of receiver feed
    N_linear = tele_geo.N_scan
    col_m2 = adj_pos_m2[0]
    row_m2 = adj_pos_m2[1]
    x_adj_m2 = adj_pos_m2[4]
    y_adj_m2 = adj_pos_m2[3]
    col_m1 = adj_pos_m1[0]
    row_m1 = adj_pos_m1[1]
    x_adj_m1 = adj_pos_m1[2]
    y_adj_m1 = adj_pos_m1[3]

    x_panm_m2 = np.reshape(
        rxmirror[0, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )
    y_panm_m2 = np.reshape(
        rxmirror[2, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )
    x_panm_m1 = np.reshape(
        rxmirror[3, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )
    y_panm_m1 = np.reshape(
        rxmirror[4, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )

    pan_id_m2 = identify_panel(x_panm_m2, y_panm_m2, x_adj_m2, y_adj_m2, col_m2, row_m2)
    pan_id_m1 = identify_panel(
        x_panm_m1, y_panm_m1 - y_cent_m1, x_adj_m1, y_adj_m1, col_m1, row_m1
    )

    row_panm_m2 = np.ravel(pan_id_m2[0, :, :])
    col_panm_m2 = np.ravel(pan_id_m2[1, :, :])
    row_panm_m1 = np.ravel(pan_id_m1[0, :, :])
    col_panm_m1 = np.ravel(pan_id_m1[1, :, :])

    # Step 2: calculate the position + local surface normal for the dish
    n_pts = len(theta)
    out = np.zeros((17, n_pts))
    out[4, :] = y_cent_m1

    for ii in range(n_pts):
        i_row = row_panm_m2[ii]
        i_col = col_panm_m2[ii]
        i_panm = np.where((panel_model2[0, :] == i_row) & (panel_model2[1, :] == i_col))

        if len(i_panm[0]) != 0:

            th = theta[ii]
            ph = phi[ii]
            r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

            alpha = r_hat[0]
            beta = r_hat[1]
            gamma = r_hat[2]

            # Receiver feed position [mm] (in telescope reference frame):
            x_0 = P_rx[0]
            y_0 = P_rx[1]
            z_0 = P_rx[2]

            a = panel_model2[2, i_panm]
            b = panel_model2[3, i_panm]
            c = panel_model2[4, i_panm]
            d = panel_model2[5, i_panm]
            e = panel_model2[6, i_panm]
            f = panel_model2[7, i_panm]
            x0 = panel_model2[8, i_panm]
            y0 = panel_model2[9, i_panm]

            def root_z2(t):
                x = x_0 + alpha * t
                y = y_0 + beta * t
                z = z_0 + gamma * t
                xm2, ym2, zm2 = tele_into_m2(
                    x, y, z
                )  # Convert ray's endpoint into M2 coordinates

                if P_rx[2] != 0:
                    z /= np.cos(np.arctan(1 / 3))
                xm2_err, ym2_err, zm2_err = tele_into_m2(
                    x, y, z
                )  # Convert ray's endpoint into M2 coordinates

                x_temp = xm2_err * np.cos(np.pi) + zm2_err * np.sin(np.pi)
                y_temp = ym2_err
                z_temp = -xm2_err * np.sin(np.pi) + zm2_err * np.cos(np.pi)

                xpc = x_temp - x0
                ypc = y_temp - y0

                z_err = (
                    a
                    + b * xpc
                    + c * (ypc)
                    + d * (xpc ** 2 + ypc ** 2)
                    + e * (xpc * ypc)
                )
                z_err = z_err[0][0]

                z_m2 = z2(xm2, ym2)  # Z of mirror in M2 coordinates

                root = zm2 - (z_m2 + z_err)
                return root

            t_m2 = optimize.brentq(root_z2, focal + 1000, focal + 12000)

            # Location of where ray hits M2
            x_m2 = x_0 + alpha * t_m2
            y_m2 = y_0 + beta * t_m2
            z_m2 = z_0 + gamma * t_m2

            # Using x and y in M2 coordiantes, find the z err:

            P_m2 = np.array([x_m2, y_m2, z_m2])

            ###### in M2 coordinates ##########################
            x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m2(
                x_m2, y_m2, z_m2
            )  # P_m2 temp
            x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m2(x_0, y_0, z_0)  # P_rx temp
            norm = d_z2(x_m2_temp, y_m2_temp)
            norm_temp = np.array([-norm[0], -norm[1], 1])
            N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
            vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array(
                [x_rx_temp, y_rx_temp, z_rx_temp]
            )
            dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2))
            tan_rx_m2 = vec_rx_m2 / dist_rx_m2

            # Outgoing ray
            tan_og = tan_rx_m2 - 2 * np.dot(np.sum(np.dot(tan_rx_m2, N_hat)), N_hat)

            # Transform back to telescope cordinates
            N_hat_t = np.zeros(3)

            N_hat_x_temp = N_hat[0] * np.cos(np.pi) - N_hat[1] * np.sin(np.pi)
            N_hat_y_temp = N_hat[0] * np.sin(np.pi) + N_hat[1] * np.cos(np.pi)
            N_hat_z_temp = N_hat[2]

            N_hat_t[0] = N_hat_x_temp
            N_hat_t[1] = N_hat_y_temp * np.cos(th2) - N_hat_z_temp * np.sin(th2)
            N_hat_t[2] = N_hat_y_temp * np.sin(th2) + N_hat_z_temp * np.cos(th2)

            tan_rx_m2_t = np.zeros(3)

            tan_rx_m2_x_temp = tan_rx_m2[0] * np.cos(np.pi) - tan_rx_m2[1] * np.sin(
                np.pi
            )
            tan_rx_m2_y_temp = tan_rx_m2[0] * np.sin(np.pi) + tan_rx_m2[1] * np.cos(
                np.pi
            )
            tan_rx_m2_z_temp = tan_rx_m2[2]

            tan_rx_m2_t[0] = tan_rx_m2_x_temp
            tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(th2) - tan_rx_m2_z_temp * np.sin(
                th2
            )
            tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(th2) + tan_rx_m2_z_temp * np.cos(
                th2
            )

            tan_og_t = np.zeros(3)
            tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
            tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
            tan_og_z_temp = tan_og[2]

            tan_og_t[0] = tan_og_x_temp
            tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
            tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)
            ##################################################

            alpha = tan_og_t[0]
            beta = tan_og_t[1]
            gamma = tan_og_t[2]

            i_row = row_panm_m1[ii]
            i_col = col_panm_m1[ii]
            i_panm = np.where(
                (panel_model1[0, :] == i_row) & (panel_model1[1, :] == i_col)
            )
            if len(i_panm[0]) != 0:
                a = panel_model1[2, i_panm]
                b = panel_model1[3, i_panm]
                c = panel_model1[4, i_panm]
                d = panel_model1[5, i_panm]
                e = panel_model1[6, i_panm]
                f = panel_model1[7, i_panm]
                x0 = panel_model1[8, i_panm]
                y0 = panel_model1[9, i_panm]

                def root_z1(t):
                    x = P_m2[0] + alpha * t
                    y = P_m2[1] + beta * t
                    z = P_m2[2] + gamma * t
                    xm1, ym1, zm1 = tele_into_m1(
                        x, y, z
                    )  # take ray end coordinates and convert to M1 coordinates

                    xm1_err, ym1_err, zm1_err = tele_into_m1(x, y, z)

                    x_temp = xm1_err * np.cos(np.pi) + zm1_err * np.sin(np.pi)
                    y_temp = ym1_err
                    z_temp = -xm1_err * np.sin(np.pi) + zm1_err * np.cos(np.pi)

                    xpc = x_temp - x0
                    ypc = y_temp - y0

                    z_err = (
                        a
                        + b * xpc
                        + c * (ypc)
                        + d * (xpc ** 2 + ypc ** 2)
                        + e * (xpc * ypc)
                    )

                    z_err = z_err[0][0]
                    z_m1 = z1(xm1, ym1)  # Z of mirror 1 in M1 coordinates
                    root = zm1 - (z_m1 + z_err)
                    return root

                t_m1 = optimize.brentq(root_z1, 500, 22000)

                # Location of where ray hits M1
                x_m1 = P_m2[0] + alpha * t_m1
                y_m1 = P_m2[1] + beta * t_m1
                z_m1 = P_m2[2] + gamma * t_m1
                P_m1 = np.array([x_m1, y_m1, z_m1])

                ###### in M1 cordinates ##########################
                x_m1_temp, y_m1_temp, z_m1_temp = tele_into_m1(
                    x_m1, y_m1, z_m1
                )  # P_m2 temp
                x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m1(
                    P_m2[0], P_m2[1], P_m2[2]
                )  # P_rx temp
                norm = d_z1(x_m1_temp, y_m1_temp)
                norm_temp = np.array([-norm[0], -norm[1], 1])
                N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
                vec_m2_m1 = np.array([x_m1_temp, y_m1_temp, z_m1_temp]) - np.array(
                    [x_m2_temp, y_m2_temp, z_m2_temp]
                )
                dist_m2_m1 = np.sqrt(np.sum(vec_m2_m1 ** 2))
                tan_m2_m1 = vec_m2_m1 / dist_m2_m1

                # Outgoing ray
                tan_og = tan_m2_m1 - 2 * np.dot(np.sum(np.dot(tan_m2_m1, N_hat)), N_hat)

                # Transform back to telescope cordinates
                N_hat_t = np.zeros(3)
                tan_m2_m1_t = np.zeros(3)
                tan_og_t = np.zeros(3)

                N_x_temp = N_hat[0] * np.cos(np.pi) + N_hat[2] * np.sin(np.pi)
                N_y_temp = N_hat[1]
                N_z_temp = -N_hat[0] * np.sin(np.pi) + N_hat[2] * np.cos(np.pi)
                N_hat_t[0] = N_x_temp
                N_hat_t[1] = N_y_temp * np.cos(tele_geo.th_1) - N_z_temp * np.sin(
                    tele_geo.th_1
                )
                N_hat_t[2] = N_y_temp * np.sin(tele_geo.th_1) + N_z_temp * np.cos(
                    tele_geo.th_1
                )

                tan_m2_m1_x_temp = tan_m2_m1[0] * np.cos(np.pi) + tan_m2_m1[2] * np.sin(
                    np.pi
                )
                tan_m2_m1_y_temp = tan_m2_m1[1]
                tan_m2_m1_z_temp = -tan_m2_m1[0] * np.sin(np.pi) + tan_m2_m1[
                    2
                ] * np.cos(np.pi)
                tan_m2_m1_t[0] = tan_m2_m1_x_temp
                tan_m2_m1_t[1] = tan_m2_m1_y_temp * np.cos(
                    tele_geo.th_1
                ) - tan_m2_m1_z_temp * np.sin(tele_geo.th_1)
                tan_m2_m1_t[2] = tan_m2_m1_y_temp * np.sin(
                    tele_geo.th_1
                ) + tan_m2_m1_z_temp * np.cos(tele_geo.th_1)

                tan_og_x_temp = tan_og[0] * np.cos(np.pi) + tan_og[2] * np.sin(np.pi)
                tan_og_y_temp = tan_og[1]
                tan_og_z_temp = -tan_og[0] * np.sin(np.pi) + tan_og[2] * np.cos(np.pi)
                tan_og_t[0] = tan_og_x_temp
                tan_og_t[1] = tan_og_y_temp * np.cos(
                    tele_geo.th_1
                ) - tan_og_z_temp * np.sin(tele_geo.th_1)
                tan_og_t[2] = tan_og_y_temp * np.sin(
                    tele_geo.th_1
                ) + tan_og_z_temp * np.cos(tele_geo.th_1)

                ##################################################

                dist_m1_ap = abs((z_ap - P_m1[2]) / tan_og_t[2])
                total_path_length = t_m2 + t_m1 + dist_m1_ap
                # total_path_length = dist_rx_m2 + dist_m2_m1 + dist_m1_ap
                pos_ap = P_m1 + dist_m1_ap * tan_og_t

                # Estimate theta
                de_ve = np.arctan(tan_rx_m2_t[2] / (-tan_rx_m2_t[1]))
                de_ho = np.arctan(
                    tan_rx_m2_t[0] / np.sqrt(tan_rx_m2_t[1] ** 2 + tan_rx_m2_t[2] ** 2)
                )

                # Write out
                out[0, ii] = x_m2
                out[1, ii] = y_m2
                out[2, ii] = z_m2

                out[3, ii] = x_m1
                out[4, ii] = y_m1
                out[5, ii] = z_m1

                out[6, ii] = N_hat_t[0]
                out[7, ii] = N_hat_t[1]
                out[8, ii] = N_hat_t[2]

                out[9, ii] = pos_ap[0]
                out[10, ii] = pos_ap[1]
                out[11, ii] = pos_ap[2]

                out[12, ii] = tan_og_t[0]
                out[13, ii] = tan_og_t[1]
                out[14, ii] = tan_og_t[2]

                out[15, ii] = total_path_length
                out[16, ii] = np.exp(
                    (-0.5)
                    * ((th - np.mean(theta)) ** 2 + (ph - np.mean(phi)) ** 2)
                    / (horn_fwhp / (np.sqrt(8 * np.log(2)))) ** 2
                )
    return out


def model_of_adj_offs(p_init, shift, tele_geo_temp, name):
    if name == "total":
        adj_1 = p_init[0:385]
        adj_2 = p_init[385:]
    elif name == "m1":
        adj_1 = p_init[0:385]
        adj_2 = p_init[385:] * 0
    elif name == "m2":
        adj_1 = p_init[0:385] * 0
        adj_2 = p_init[385:]
    # Define panels on M1 and M2. Here you can define the
    # magnitude of the adjuster offsets on each mirror:
    pan_mod_m2 = pm.panel_model_from_adjuster_offsets(
        2, adj_2, 1, 0
    )  # Panel Model on M2
    pan_mod_m1 = pm.panel_model_from_adjuster_offsets(
        1, adj_1, 1, 0
    )  # Panel Model on M1

    # Define FOV of RX. In other words, define directions of
    # outgoing rays from the RX.
    th = np.linspace(-np.pi / 2 - 0.28, -np.pi / 2 + 0.28, tele_geo_temp.N_scan)
    ph = np.linspace(np.pi / 2 - 0.28, np.pi / 2 + 0.28, tele_geo_temp.N_scan)
    # Define the path of the rays from the RX to the aperture plane
    rx_temp = np.array([tele_geo_temp.rx_x, tele_geo_temp.rx_y, tele_geo_temp.rx_z])
    rxmirror_temp = ray_mirror_pts(rx_temp, tele_geo_temp, th, ph)
    out = aperature_fields_from_panel_model(
        pan_mod_m1, pan_mod_m2, rx_temp, tele_geo_temp, th, ph, rxmirror_temp
    )

    beam_temp = ff.far_field_sim(out, tele_geo_temp, rx_temp)

    np.savetxt(
        "/data/chesmore/sim_out/sim_temp.txt",
        np.c_[
            np.real(beam_temp[0, :]),
            np.real(beam_temp[1, :]),
            np.real(beam_temp[2, :]),
            np.imag(beam_temp[2, :]),
        ],
    )
    reference_motion_correction = 1
    dat_temp = np.loadtxt("/data/chesmore/sim_out/sim_temp.txt")
    x_temp, y_temp, phase_temp, ampl_temp, geo = afit.analyze_holography(
        dat_temp, tele_geo_temp, 0, reference_motion_correction, 0, shift
    )

    phase_temp = np.where(
        (abs(ampl_temp) / np.max(abs(ampl_temp))) >= 0.3,
        phase_temp - np.mean(phase_temp),
        0,
    )

    return phase_temp
