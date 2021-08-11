import numpy as np
import matplotlib.patches as mpatches


def adams_bashforth(h, order, time_hist, t):

    if order == 1:
        return h

    # Now that we have disparate time history points, we need to
    # solve for the AB coefficients.
    thist = time_hist.copy() - t
    ab = np.zeros(order)
    vdmt = np.zeros((order, order))
    coeff_rhs = np.zeros(order)
    for i in range(0, order):
        coeff_rhs[i] = (1/(i+1))*h**(i+1)
        for j in range(0, order):
            vdmt[i][j] = thist[i]**j

    ab = np.linalg.solve(np.transpose(vdmt), coeff_rhs)

    return ab


def ab_svd(h, order, time_hist, t):

    # Extended history - use SVD, as is done in Leap.
    thist = time_hist.copy() - t
    ab = np.zeros(order+1)
    vdmt = np.zeros((order+1, order))
    coeff_rhs = np.zeros(order)
    for i in range(0, order):
        coeff_rhs[i] = (1/(i+1))*h**(i+1)
        for j in range(0, order+1):
            vdmt[j][i] = thist[j]**i

    u, sigma, v = np.linalg.svd(np.transpose(vdmt), full_matrices=False)
    ainv = np.dot(v.transpose(), np.dot(np.linalg.inv(np.diag(sigma)),
                  u.transpose()))
    ab = np.dot(ainv, coeff_rhs)

    return ab


def am_svd(h, order, time_hist, t):

    # Extended history - use SVD, as is done in Leap.
    thist = time_hist.copy() - t + h
    am = np.zeros(order+1)
    vdmt = np.zeros((order+1, order))
    coeff_rhs = np.zeros(order)
    for i in range(0, order):
        coeff_rhs[i] = (1/(i+1))*h**(i+1)
        for j in range(0, order+1):
            vdmt[j][i] = thist[j]**i

    u, sigma, v = np.linalg.svd(np.transpose(vdmt), full_matrices=False)
    ainv = np.dot(v.transpose(), np.dot(np.linalg.inv(np.diag(sigma)),
                  u.transpose()))
    am = np.dot(ainv, coeff_rhs)

    return am


def lagrange_coeffs(t, time_hist, order):

    # We use this routine to interpolate or extrapolate, given a history,
    # an accompanying time history, and a new time at which to obtain a
    # value.
    b = np.zeros(order)
    for i in range(0, order):
        b[i] = 1
        for j in range(0, order):
            if i != j:
                b[i] = b[i]*(t - time_hist[j])/(time_hist[i] - time_hist[j])

    return b


def main():

    # Determining max(z) contours for each SR in first
    # order MR.
    order = 3
    srs = [1, 2, 3, 4, 5]
    # Modify this to see what effect stiffness has on
    # the stability of the explicit part of the SBDF.
    mu = 0

    am_ext = True

    # root locus: loop through theta.
    imags = np.linspace(-4, 4, 501)
    reals = np.linspace(-8, 5, 500)
    zs = np.zeros((501, 500))

    sr_colors = ['red', 'blue', 'green', 'black', 'orange']
    patch_list = []

    lag_thist = np.zeros(order + 1)
    for i in range(0, order + 1):
        lag_thist[i] = 1 - i
    bl = np.zeros((order + 1, order + 1))

    # Now, calculate the stability bound for a
    # single part of the IMEX scheme.
    import matplotlib.pyplot as plt
    plt.clf()
    for sr_i, sr in enumerate(srs):
        print("SR = ", sr)
        for i, re in enumerate(reals):
            for j, im in enumerate(imags):

                lbda = re + 1j*im
                if order == 1:
                    # FIXME: set up extended history
                    s = 0
                    for lind in range(0, sr):
                        s += (lbda/sr)*(lind/sr)*((1 + lbda)/(1 - mu) - 1)
                    crit = (abs((1/(1 - mu))*(1 + lbda + s)))
                    zs[j, i] = crit
                elif order == 2:
                    # FIXME: set up extended history
                    # Set up requisite time histories.
                    time_hist = np.zeros(2)
                    time_hist[0] = 0
                    time_hist[1] = -1
                    # First explicit microstep
                    ab = adams_bashforth((1/sr), order, time_hist, 0)
                    ynp_n = 1 + ab[0]*lbda
                    ynp_nm1 = ab[1]*lbda
                    # First IMEX macrostep to get trial state
                    # this will be used in state interp for substep args
                    y_trial_n_c = (1/(1 - (1/2)*mu))*(1 + (3/2)*lbda
                                                      + (1/2)*mu)
                    y_trial_nm1_c = (1/(1 - (1/2)*mu))*(-(1/2)*lbda)
                    # Rotate time history for AB coefficient handling.
                    for lind in range(order-1, 0, -1):
                        time_hist[lind] = time_hist[lind-1]
                    time_hist[0] = (1/sr)
                    # Substepping.
                    for k in range(2, sr+1):
                        # get AB coeffs.
                        ab = adams_bashforth((1/sr), order, time_hist,
                                             (k-1)/sr)
                        # get lagrange interpolating coeffs for each utilized
                        # state value.
                        bl[0, :] = lagrange_coeffs((k-1)/sr, lag_thist,
                                                   order + 1)
                        bl[1, :] = lagrange_coeffs((k-2)/sr, lag_thist,
                                                   order + 1)
                        # combine all coeffs to append to state multipliers
                        for lind in range(0, order):
                            ynp_n += lbda*ab[lind]*(bl[lind, 1] +
                                                    bl[lind, 0]*y_trial_n_c)
                            ynp_nm1 += lbda*ab[lind]*(bl[lind, 2] +
                                                      bl[lind, 0] *
                                                      y_trial_nm1_c)
                        # Rotate time history for AB coefficient handling.
                        for lind in range(order-1, 0, -1):
                            time_hist[lind] = time_hist[lind-1]
                        time_hist[0] = (k/sr)
                    # Final macrostep solve: apply multiplicative factor
                    ynp_n += (1/2)*mu
                    ynp_n *= (1/(1 - (1/2)*mu))
                    ynp_nm1 *= (1/(1 - (1/2)*mu))
                    a = -1
                    b = ynp_n
                    c = ynp_nm1
                    crit1 = abs((-b + np.sqrt(b**2 - 4*a*c)) / (2*a))
                    crit2 = abs((-b - np.sqrt(b**2 - 4*a*c)) / (2*a))
                    zs[j, i] = max(crit1, crit2)
                elif order == 3:
                    # Set up requisite time histories.
                    time_hist = np.zeros(4)
                    time_hist[0] = 0
                    time_hist[1] = -1
                    time_hist[2] = -2
                    time_hist[3] = -3
                    # First explicit microstep
                    ab = ab_svd((1/sr), order, time_hist, 0)
                    ynp_n = 1 + ab[0]*lbda
                    ynp_nm1 = ab[1]*lbda
                    ynp_nm2 = ab[2]*lbda
                    ynp_nm3 = ab[3]*lbda
                    # First IMEX macrostep to get trial state
                    # this will be used in state interp for substep args
                    ab_mac = ab_svd(1, order, time_hist, 0)
                    if am_ext:
                        am_mac = am_svd(1, order, time_hist, 0)
                        y_trial_n_c = (1/(1 - am_mac[0]*mu))*(1 +
                                                              ab_mac[0]*lbda +
                                                              am_mac[1]*mu)
                        y_trial_nm1_c = (1/(1 - am_mac[0]*mu))*(ab_mac[1] *
                                                                lbda +
                                                                am_mac[2]*mu)
                        y_trial_nm2_c = (1/(1 - am_mac[0]*mu))*(ab_mac[2] *
                                                                lbda +
                                                                am_mac[3]*mu)
                        y_trial_nm3_c = (1/(1 - am_mac[0]*mu))*(ab_mac[3]*lbda)
                    else:
                        y_trial_n_c = (1/(1 - (5/12)*mu))*(1 + ab_mac[0]*lbda +
                                                           (2/3)*mu)
                        y_trial_nm1_c = (1/(1 - (5/12)*mu))*(ab_mac[1]*lbda -
                                                             (1/12)*mu)
                        y_trial_nm2_c = (1/(1 - (5/12)*mu))*(ab_mac[2]*lbda)
                        y_trial_nm3_c = (1/(1 - (5/12)*mu))*(ab_mac[3]*lbda)
                    # Rotate time history for AB coefficient handling.
                    for lind in range(order, 0, -1):
                        time_hist[lind] = time_hist[lind-1]
                    time_hist[0] = (1/sr)
                    # Substepping.
                    for k in range(2, sr+1):
                        # get AB coeffs.
                        ab = ab_svd((1/sr), order, time_hist, (k-1)/sr)
                        # get lagrange interpolating coeffs.
                        bl[0, :] = lagrange_coeffs((k-1)/sr, lag_thist,
                                                   order + 1)
                        bl[1, :] = lagrange_coeffs((k-2)/sr, lag_thist,
                                                   order + 1)
                        if k == 2:
                            bl[2, :] = lagrange_coeffs(-1, lag_thist,
                                                       order + 1)
                            bl[3, :] = lagrange_coeffs(-2, lag_thist,
                                                       order + 1)
                        elif k == 3:
                            bl[2, :] = lagrange_coeffs((k-3)/sr, lag_thist,
                                                       order + 1)
                            bl[3, :] = lagrange_coeffs(-1, lag_thist,
                                                       order + 1)
                        else:
                            bl[2, :] = lagrange_coeffs((k-3)/sr, lag_thist,
                                                       order + 1)
                            bl[3, :] = lagrange_coeffs((k-4)/sr, lag_thist,
                                                       order + 1)
                        # combine all coeffs to append to state multipliers
                        for lind in range(0, order+1):
                            ynp_n += lbda*ab[lind]*(bl[lind, 1] +
                                                    bl[lind, 0] * y_trial_n_c)
                            ynp_nm1 += lbda*ab[lind]*(bl[lind, 2] +
                                                      bl[lind, 0] *
                                                      y_trial_nm1_c)
                            ynp_nm2 += lbda*ab[lind]*(bl[lind, 3] +
                                                      bl[lind, 0] *
                                                      y_trial_nm2_c)
                            ynp_nm3 += lbda*ab[lind]*(bl[lind, 0] *
                                                      y_trial_nm3_c)
                        # Rotate time history for AB coefficient handling.
                        for lind in range(order, 0, -1):
                            time_hist[lind] = time_hist[lind-1]
                        time_hist[0] = (k/sr)
                    # Final macrostep solve: apply multiplicative factor
                    if am_ext:
                        ynp_n += am_mac[1]*mu
                        ynp_n *= (1/(1 - am_mac[0]*mu))
                        ynp_nm1 += am_mac[2]*mu
                        ynp_nm1 *= (1/(1 - am_mac[0]*mu))
                        ynp_nm2 += am_mac[3]*mu
                        ynp_nm2 *= (1/(1 - am_mac[0]*mu))
                        ynp_nm3 *= (1/(1 - am_mac[0]*mu))
                    else:
                        ynp_n += (2/3)*mu
                        ynp_n *= (1/(1 - (5/12)*mu))
                        ynp_nm1 += -(1/12)*mu
                        ynp_nm1 *= (1/(1 - (5/12)*mu))
                        ynp_nm2 *= (1/(1 - (5/12)*mu))
                        ynp_nm3 *= (1/(1 - (5/12)*mu))
                    a = -1
                    b = ynp_n
                    c = ynp_nm1
                    d = ynp_nm2
                    e = ynp_nm3
                    crit = max(abs(np.roots([a, b, c, d, e])))
                    zs[j, i] = crit
                else:
                    # Set up requisite time histories.
                    time_hist = np.zeros(4)
                    time_hist[0] = 0
                    time_hist[1] = -1
                    time_hist[2] = -2
                    time_hist[3] = -3
                    # First explicit microstep
                    ab = adams_bashforth((1/sr), order, time_hist, 0)
                    ynp_n = 1 + ab[0]*lbda
                    ynp_nm1 = ab[1]*lbda
                    ynp_nm2 = ab[2]*lbda
                    ynp_nm3 = ab[3]*lbda
                    # First IMEX macrostep to get trial state
                    # this will be used in state interp for substep args
                    y_trial_n_c = (1/(1 - (9/24)*mu))*(1 + (55/24)*lbda +
                                                       (19/24)*mu)
                    y_trial_nm1_c = (1/(1 - (9/24)*mu))*(-(59/24)*lbda -
                                                         (5/24)*mu)
                    y_trial_nm2_c = (1/(1 - (9/24)*mu))*((37/24)*lbda +
                                                         (1/24)*mu)
                    y_trial_nm3_c = (1/(1 - (9/24)*mu))*(-(9/24)*lbda)
                    # Rotate time history for AB coefficient handling.
                    for lind in range(order-1, 0, -1):
                        time_hist[lind] = time_hist[lind-1]
                    time_hist[0] = (1/sr)
                    # Substepping.
                    for k in range(2, sr+1):
                        # get AB coeffs.
                        ab = adams_bashforth((1/sr), order, time_hist,
                                             (k-1)/sr)
                        # get lagrange interpolating coeffs.
                        bl[0, :] = lagrange_coeffs((k-1)/sr, lag_thist,
                                                   order + 1)
                        bl[1, :] = lagrange_coeffs((k-2)/sr, lag_thist,
                                                   order + 1)
                        if k == 2:
                            bl[2, :] = lagrange_coeffs(-1, lag_thist,
                                                       order + 1)
                        else:
                            bl[2, :] = lagrange_coeffs((k-3)/sr, lag_thist,
                                                       order + 1)
                        if k == 2:
                            bl[3, :] = lagrange_coeffs(-2, lag_thist,
                                                       order + 1)
                        elif k == 3:
                            bl[3, :] = lagrange_coeffs(-1, lag_thist,
                                                       order + 1)
                        else:
                            bl[3, :] = lagrange_coeffs((k-4)/sr, lag_thist,
                                                       order + 1)
                        # combine all coeffs to append to state multipliers
                        for lind in range(0, order):
                            ynp_n += lbda*ab[lind]*(bl[lind, 1] +
                                                    bl[lind, 0]*y_trial_n_c)
                            ynp_nm1 += lbda*ab[lind]*(bl[lind, 2] +
                                                      bl[lind, 0] *
                                                      y_trial_nm1_c)
                            ynp_nm2 += lbda*ab[lind]*(bl[lind, 3] +
                                                      bl[lind, 0] *
                                                      y_trial_nm2_c)
                            ynp_nm3 += lbda*ab[lind]*(bl[lind, 4] +
                                                      bl[lind, 0] *
                                                      y_trial_nm3_c)
                        # Rotate time history for AB coefficient handling.
                        for lind in range(order-1, 0, -1):
                            time_hist[lind] = time_hist[lind-1]
                        time_hist[0] = (k/sr)
                    # Final macrostep solve: apply multiplicative factor
                    ynp_n += (19/24)*mu
                    ynp_n *= (1/(1 - (9/24)*mu))
                    ynp_nm1 += -(5/24)*mu
                    ynp_nm1 *= (1/(1 - (9/24)*mu))
                    ynp_nm2 += (1/24)*mu
                    ynp_nm2 *= (1/(1 - (9/24)*mu))
                    ynp_nm3 *= (1/(1 - (9/24)*mu))
                    a = -1
                    b = ynp_n
                    c = ynp_nm1
                    d = ynp_nm2
                    e = ynp_nm3
                    crit = max(abs(np.roots([a, b, c, d, e])))
                    zs[j, i] = crit

        # Contour plot of the maximum amplification factor
        # plt.contour(reals, imags, zs, levels=[1], colors=sr_colors[sr_i])
        # Normalize
        plt.contour(reals+mu, imags, zs, levels=[1], colors=sr_colors[sr_i])
        patch_list.append(mpatches.Patch(color=sr_colors[sr_i],
                                         label="SR = {}".format(sr)))

    plt.title("Extended IMEX Adams MR Order {order} Stability Regions, "
              "mu = {mu}".format(order=order, mu=mu))
    plt.xlabel("Re(lambda)")
    plt.ylabel("Im(lambda)")
    plt.legend(handles=patch_list, loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
