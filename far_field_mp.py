"""
Far field simulation with multiprocessing
based on the orignial code of Grace E. Chesmore (May 2021)
"""

import numpy as np
import multiprocessing as mp

class FarFieldSim():

    def __init__(self, ap_field, msmt_geo, rx):
        # Break out many quantities from msmt_geo
        self.N_scan = msmt_geo.N_scan
        self.de_ang = msmt_geo.de_ang
        lambda_ = msmt_geo.lambda_

        self.x_tow = msmt_geo.x_tow
        self.y_tow = msmt_geo.y_tow
        self.z_tow = msmt_geo.z_tow

        self.x_phref = msmt_geo.x_phref
        self.y_phref = msmt_geo.y_phref
        self.z_phref = msmt_geo.z_phref

        x_rotc = msmt_geo.x_rotc
        y_rotc = msmt_geo.y_rotc
        z_rotc = msmt_geo.z_rotc

        self.el0 = msmt_geo.el0
        self.az0 = msmt_geo.az0

        # Break out the geometric coordinates from ap_fields
        # Location of points on the aperture plane
        # in rotation centered coordinates
        self.x_ap = ap_field[9, :] / 1e3 - x_rotc
        self.y_ap = ap_field[10, :] / 1e3 - y_rotc
        self.z_ap = ap_field[11, :] / 1e3 - z_rotc

        # Propagation vector of the sample points (tan_og)
        # self.k_x = ap_field[12, :]
        # self.k_y = ap_field[13, :]
        self.k_z = ap_field[14, :]

        pathl = ap_field[15, :] / 1e3  # Path length convert to meters
        ampl = np.sqrt(ap_field[16, :])  # Amplitude

        self.k = 2.0 * np.pi / lambda_  # Wavenumber [1/m]

        # Complex fields
        self.ima = np.complex(0, 1)
        self.Fcomplex = ampl * np.exp(self.ima * pathl * self.k)

        self.Npts = len(self.x_ap)

    def find_beam(self, i_ang, j_ang):
        de_az = (i_ang) * self.de_ang
        de_el = (j_ang) * self.de_ang

        # Rotate to az,el
        el_cur = self.el0 + de_el
        az_cur = self.az0 + de_az

        # Elevation rotation (about x axis)
        x_temp = self.x_ap
        y_temp = np.cos(el_cur) * self.y_ap - np.sin(el_cur) * self.z_ap
        z_temp = np.sin(el_cur) * self.y_ap + np.cos(el_cur) * self.z_ap

        x_apr = np.cos(az_cur) * x_temp + np.sin(az_cur) * z_temp
        y_apr = y_temp
        z_apr = -np.sin(az_cur) * x_temp + np.cos(az_cur) * z_temp

        # Evaluate the distance to the phase reference if prompted to do so

        x_temp = self.x_phref
        y_temp = np.cos(el_cur) * self.y_phref - np.sin(el_cur) * self.z_phref
        z_temp = np.sin(el_cur) * self.y_phref + np.cos(el_cur) * self.z_phref

        x_phrefr = np.cos(az_cur) * x_temp + np.sin(az_cur) * z_temp
        y_phrefr = y_temp
        z_phrefr = -np.sin(az_cur) * x_temp + np.cos(az_cur) * z_temp

        r_phref = np.sqrt(
            (x_phrefr - self.x_tow) ** 2
            + (y_phrefr - self.y_tow) ** 2
            + (z_phrefr - self.z_tow) ** 2
        )

        # Evaluate r
        r = np.sqrt(
            (x_apr - self.x_tow) ** 2 + (y_apr - self.y_tow) ** 2 + (z_apr - self.z_tow) ** 2
        )
        z_dot_rhat = (z_apr - self.z_tow) * (-1) / r

        out0 = az_cur
        out1 = el_cur
        out2 = (
            np.exp(-self.ima * r_phref * self.k)
            * np.sum(
                (self.Fcomplex * np.exp(self.ima * self.k * r) / (4 * np.pi * r))
                * ((self.ima * self.k + 1 / r) * z_dot_rhat + self.ima * self.k * self.k_z)
            )
            / self.Npts
        )
        
        return [out0, out1, out2]

    def output(self):
        range_scan = [(i,j) for i in range(-self.N_scan, self.N_scan, 1) for j in range(-self.N_scan, self.N_scan, 1)]
        print()
        pool = mp.Pool(processes=mp.cpu_count())
        result = pool.starmap(self.find_beam, range_scan)
        pool.close()
        pool.join()

        return np.array(result).transpose()

# utility function
def far_field_sim(ap_field, msmt_geo, rx):
    FFS = FarFieldSim(ap_field, msmt_geo, rx)
    return FFS.output()

if __name__ == "__main__":
    '''
    To test the code 
    '''
    
    import time
    import tele_geo as tg
    import far_field as ff
    import ap_field as af
    import pan_mod as pm

    print("Calculating the aperture field.....")
    tele_geo_t = tg.initialize_telescope_geometry()

    # import pickle
    # from DEFAULTS import PARENT_PATH
    # with open(PARENT_PATH+"/tutorials/data/aperture_field_sim_out_tB_new.txt", "rb") as fp:   # Unpickling
    #     ap_field = pickle.load(fp)

    # Panel Models with surface errors
    adj_1_A = np.random.randn(1092) * 20
    adj_2_A = np.random.randn(1092) * 20

    pan_mod2_tA = pm.panel_model_from_adjuster_offsets(
        2, adj_2_A, 1, 0
    )  #  on M2
    pan_mod1_tA = pm.panel_model_from_adjuster_offsets(
        1, adj_1_A, 1, 0
    ) 

    # FOV of RX (directions of outgoing rays from the receiver feed)
    N_th = 50
    N_ph = 50
    th = np.linspace(-np.pi / 2 - 0.28, -np.pi / 2 + 0.28, N_th)
    ph = np.linspace(np.pi / 2 - 0.28, np.pi / 2 + 0.28, N_ph)

    rx_t = np.array([0, 0, 0])
    # Path of the rays from the RX to the aperture plane
    rxmirror_t = af.ray_mirror_pts(rx_t, tele_geo_t, th, ph)
    ap_field = af.aperature_fields_from_panel_model(
        pan_mod1_tA, pan_mod2_tA, rx_t, tele_geo_t, th, ph, rxmirror_t
    )
    print("Got the aperture field.")

    # Here we compare the processing time
    # # with multiprocessing
    print(f"Calculating the beam field WITH multiprocessing...") 
    time_start = time.time()
    FFS = FarFieldSim(ap_field, tele_geo_t, None)
    beam_mp = FFS.output()
    print(f"Processing time : {time.time()-time_start}") 

    # # without multiprocessing
    print(f"Calculating the beam field without multiprocessing...") 
    time_start = time.time()
    beam = ff.far_field_sim(ap_field, tele_geo_t, None)
    print(f"Processing time : {time.time()-time_start}") 

    assert beam.all() == beam_mp.all(), \
            "Inconsistency found. Please contact Tung at ctcheung@uchicago.edu for bug-fixing"