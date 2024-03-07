import numpy as np


PLANCK_C = 6.6261e-34
BOLTZMANN = 1.3806e-23
SPEED_LIGHT = 299792458
SIGMA = 5.6704e-08


def planck(f, t):
    """
    Planck function for blackbody radiation

    Parameters:
        f    Frequencies [Hz]
        t    A temperature [K]

    Returns:
        Planck function values [W/(m2*Hz*sr)]
    """

    a = 2 * PLANCK_C / SPEED_LIGHT**2
    b1 = PLANCK_C / BOLTZMANN

    b = a * f**3 / (np.exp(b1 * f / t) - 1)
    return b


def vmr2nd(vmr, p, t):
    """
    Derives molecular number densities from volume mixing ratios

    Parameters:
        vmr  Volume mixing ratios [-]
        p    A pressure [Pa]
        t    A temperature [K]

    Returns:
        Number densities [1/m3]
    """
    kb = 1.3806e-23

    n = vmr * p / (kb * t)
    return n


def absorption_coeff(vmr, p, t, xsec):
    """
    Calculates absorption coefficients for one altitude

    Parameters:
        vmr  Volume mixing ratios [-]
        p    A pressure [Pa]
        t    A temperature [K]
        xsec Absorption cross-sections [m2]. Dimensions (vmr,f)

    Returns:
        Absorption coefficients [1/m]. Dimension (f)

    """
    n = vmr2nd(vmr, p, t)
    # Add an axis to n to make multiplication work
    a = np.multiply(n[:, np.newaxis], xsec)
    a = sum(a, 0)
    return a


def spectral_radiance(f, z, p, t, vmr, xsec, za):
    """
    Calculates the spectral radiance at the top of the atmosphere

    Parameters:
        f     Frequencies [Hz]
        z     Altitudes [m]
        p     Pressures [Pa]
        t     Temperatures [K]
        vmr   Volume mixing ratios [-]. Dimensions (gas, altitudes)
        xsec  Absorption cross-sections [m2]. Dimensions (gas,vmr,f)
        za    Zenith angle [rad].

    Returns:
        Spectral radiance [W/(m2*Hz*sr)]. Dimension (f)
    """
    # Init rs to surface blackbody radiation
    rs = planck(f, t[0])
    tau_out = np.zeros(xsec.shape[1:])

    # Loop altitudes
    a_this = 0  # Dummy value
    #
    a_out = np.zeros(xsec.shape[1:])
    for i, z_i in enumerate(z):
        # Absorption at previous level
        a_old = a_this

        # Calculate absorption for this level
        a_this = absorption_coeff(vmr[:, i], p[i], t[i], xsec[:, :, i])
        a_out[:, i] = a_this

        # We only do radiance transfer from i=1
        if i > 0:
            # Optical thickness of layer
            tau = (a_old + a_this) / 2 * ((z_i - z[i - 1]) / np.cos(za))
            tau_out[:, i] = tau

            # Transmission of layer
            transmission = np.exp(-tau)

            # Effective Planck function of the layer
            b = planck(f, (t[i - 1] + t[i]) / 2)

            # Update I
            rs = rs * transmission + b * (1 - transmission)
    return rs
