import numpy as np


def nonstiff(y, lbda):

    dy = np.empty(1, dtype=complex)
    dy = lbda*y
    return dy


def stiff(y, mu):

    dy = np.empty(1, dtype=complex)
    dy = mu*y
    return dy


def exact(t, mu, lbda):
    y = np.empty(1, dtype=complex)
    y = np.exp((lbda+mu)*t)
    return y


def lagrange(t, hist, time_hist, order):

    # We use this routine to interpolate or extrapolate, given a history,
    # an accompanying time history, and a new time at which to obtain a
    # value.
    a_new = 0
    for i in range(0, order):
        term = hist[i]
        for j in range(0, order):
            if i != j:
                term = term*(t - time_hist[j])/(time_hist[i] - time_hist[j])
        a_new += term

    return a_new


def adams_inc(y, h, order, time_hist, rhs_hist, t):

    if order == 1:
        y_new = h*(rhs_hist[0])
        return y_new

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

    # Obtain new substep-level state estimate with these coeffs.
    if order == 2:
        y_new = ab[0]*rhs_hist[0] + ab[1]*rhs_hist[1]
    elif order == 3:
        y_new = (ab[0]*rhs_hist[0] +
                 ab[1]*rhs_hist[1] +
                 ab[2]*rhs_hist[2])
    elif order == 4:
        y_new = (ab[0]*rhs_hist[0] +
                 ab[1]*rhs_hist[1] +
                 ab[2]*rhs_hist[2] +
                 ab[3]*rhs_hist[3])

    return y_new


def imex_adams(y, h, t, lbda, mu, state_hist, rhs_hist, rhsns_hist,
               time_hist, order, sr):

    # "Slowest first:" - first, use RHS extrapolation to form implicit
    # prediction, then solve.
    new_nonstiff = adams_inc(y, h, order,
                             time_hist, rhsns_hist,
                             time_hist[0])

    # Solve.
    if order == 1:
        y_new = (state_hist[0] + new_nonstiff)/(1 - h*mu)
    elif order == 2:
        # Adams way.
        y_new = (1/(1 - (1.0/2.0)*h*mu))*(state_hist[0] +
                                          (1.0/2.0) * h * rhs_hist[0] +
                                          new_nonstiff)
    elif order == 3:
        # Adams way.
        y_new = (1/(1 - (5.0/12.0)*h*mu)) * \
            (state_hist[0] +
             (2.0/3.0)*h*rhs_hist[0] -
             (1.0/12.0)*h*rhs_hist[1] +
             new_nonstiff)
    else:
        # Adams way.
        y_new = (1/(1 - (9.0/24.0)*h*mu)) * \
            (state_hist[0] +
             (19.0/24.0)*h*rhs_hist[0] -
             (5.0/24.0)*h*rhs_hist[1] +
             (1.0/24.0)*h*rhs_hist[2] +
             new_nonstiff)

    new_nonstiff = nonstiff(y_new, lbda)

    # Now do AB at the substep level, supplementing with interpolation of the
    # newfound stiff RHS.
    new_stiff = stiff(y_new, mu)

    if sr > 1:
        # Different approach: accrue state increments at each substep, using
        # the explicit RHS and *interpolated* state.
        # Now build a new state history so we can interpolate with it.
        new_state_hist = np.empty(order+1, dtype=complex)
        for i in range(0, order):
            new_state_hist[i+1] = state_hist[i]
        new_state_hist[0] = y_new

        # For the first substep, use existing nonstiff RHS history...
        substep_rhs_hist = rhsns_hist.copy()
        substep_time_hist = np.zeros(order)
        for i in range(0, order):
            substep_time_hist[i] = t - i*h
        new_nonstiff = adams_inc(y, (1/sr)*h, order,
                                 substep_time_hist, substep_rhs_hist,
                                 substep_time_hist[0])

        lag_thist = np.zeros(order + 1)
        for j in range(0, order + 1):
            lag_thist[j] = t - (j-1)*h
        for j in range(1, sr):
            # Again, we have the available history to go one order higher.
            substep_state = lagrange(t + (j/sr)*h, new_state_hist, lag_thist,
                                     order + 1)
            nonstiff_rhs = nonstiff(substep_state, lbda)
            # Rotate history.
            for i in range(order-1, 0, -1):
                substep_rhs_hist[i] = substep_rhs_hist[i-1]
                substep_time_hist[i] = substep_time_hist[i-1]
            substep_rhs_hist[0] = nonstiff_rhs
            substep_time_hist[0] = t + (j/sr)*h
            # Add to explicit increment for the macrostep.
            new_nonstiff += adams_inc(y, (1/sr)*h, order,
                                      substep_time_hist, substep_rhs_hist,
                                      substep_time_hist[0])

        # Re-solve.
        if order == 1:
            y_new = (state_hist[0] + new_nonstiff)/(1 - h*mu)
        elif order == 2:
            # Adams way.
            y_new = (1/(1 - (1.0/2.0)*h*mu))*(state_hist[0] +
                                              (1.0/2.0) * h * rhs_hist[0] +
                                              new_nonstiff)
        elif order == 3:
            # Adams way.
            y_new = (1/(1 - (5.0/12.0)*h*mu)) * \
                (state_hist[0] +
                 (2.0/3.0)*h*rhs_hist[0] -
                 (1.0/12.0)*h*rhs_hist[1] +
                 new_nonstiff)
        else:
            # Adams way.
            y_new = (1/(1 - (9.0/24.0)*h*mu)) * \
                (state_hist[0] +
                 (19.0/24.0)*h*rhs_hist[0] -
                 (5.0/24.0)*h*rhs_hist[1] +
                 (1.0/24.0)*h*rhs_hist[2] +
                 new_nonstiff)

    new_stiff = stiff(y_new, mu)
    new_nonstiff = nonstiff(y_new, lbda)
    return (y_new, new_nonstiff, new_stiff)


def main():

    # Stability test
    t_start = 0
    t_end = 100
    dt = 1
    order = 4
    srs = [1, 2, 3, 4, 5]

    # Set ratio criteria - order 4 is persnickety
    if order < 4:
        ratio_threshold = 9
    else:
        ratio_threshold = 3

    n_thetas = 500
    # just check the real axis.
    thetas = [-np.pi, 0]
    theta_lbda = -np.pi
    rs_lbda = [0.3]

    # Now, calculate the stability bound for a
    # single part of the IMEX scheme.
    for sr in srs:
        print("========================================")
        print("SR = ", sr)
        rs_max = []
        for _i in range(0, n_thetas):
            rs_max.append(1000)
        for r_lbda in rs_lbda:
            lbda = r_lbda*np.exp(theta_lbda*1j)
            # We will be finding the minimum (limiting)
            # lambdas for all mus in the left half plane.
            for ith, theta in enumerate(thetas):

                print("Theta = ", theta)
                r = 0
                dr = 10
                while abs(dr) > 0.00001:

                    mu = r*np.exp(theta*1j)
                    y_old = 1

                    times = []
                    states = []
                    states.append(y_old)
                    exact_states = []
                    exact_states.append(y_old)

                    t = t_start
                    times.append(t)
                    step = 0
                    rhs_hist = np.empty(order, dtype=complex)
                    rhsns_hist = np.empty(order, dtype=complex)
                    state_hist = np.empty(order, dtype=complex)
                    time_hist = np.empty(order, dtype=complex)
                    rhs_hist[0] = stiff(y_old, mu)
                    rhsns_hist[0] = nonstiff(y_old, lbda)
                    state_hist[0] = y_old
                    time_hist[0] = t
                    tiny = 1e-15
                    fail = False
                    ratio_counter = 0
                    # If r = 0, our ratio will always be 1.
                    if r == 0:
                        crit = 1.0 + 1e-10
                    else:
                        crit = 1.0
                    while t < t_end - tiny:
                        if step < order - 1:
                            # "Bootstrap" using known exact solution.
                            y = exact(t + dt, mu, lbda)
                            dy_ns = nonstiff(y, lbda)
                            dy_stiff = stiff(y, mu)
                        else:
                            y, dy_ns, dy_stiff = imex_adams(y_old, dt, t, lbda,
                                                            mu, state_hist,
                                                            rhs_hist,
                                                            rhsns_hist,
                                                            time_hist, order,
                                                            sr)
                        # Rotate histories.
                        for i in range(order-1, 0, -1):
                            rhs_hist[i] = rhs_hist[i-1]
                            rhsns_hist[i] = rhsns_hist[i-1]
                            state_hist[i] = state_hist[i-1]
                            time_hist[i] = time_hist[i-1]
                        rhs_hist[0] = dy_stiff
                        rhsns_hist[0] = dy_ns
                        state_hist[0] = y
                        time_hist[0] = t + dt
                        # Append to states and prepare for next step.
                        states.append(y)
                        t += dt
                        times.append(t)
                        y_old = y
                        step += 1
                        ratio = abs(states[-1]/states[-2])
                        if ratio >= crit:
                            ratio_counter += 1
                        else:
                            ratio_counter = 0
                        if ratio_counter > ratio_threshold:
                            fail = True
                            break
                    if fail:
                        dr = 0
                    else:
                        # Success: increase r.
                        r += 0.01
                        if r >= 15:
                            break

                if r < rs_max[ith]:
                    rs_max[ith] = r
                    print("r = ", r)


if __name__ == "__main__":
    main()
