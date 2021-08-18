import numpy as np


def adams_bashforth_const(h, order, time_hist, t):

    # Now that we have disparate time history points, we need to
    # solve for the AB coefficients.
    thist = time_hist.copy() - t
    # ab = np.zeros(order)
    vdmt = np.zeros((len(thist), order))
    coeff_rhs = np.zeros(order)
    for i in range(0, order):
        coeff_rhs[i] = (1/(i+1))*h**(i+1)
        for j in range(0, len(thist)):
            vdmt[j][i] = thist[j]**i

    return np.transpose(vdmt), coeff_rhs


def adams_moulton_const(h, order, time_hist, t):

    # Now that we have disparate time history points, we need to
    # solve for the AB coefficients.
    thist = time_hist.copy() - t + h
    # am = np.zeros(order)
    vdmt = np.zeros((len(thist), order))
    coeff_rhs = np.zeros(order)
    for i in range(0, order):
        coeff_rhs[i] = (1/(i+1))*h**(i+1)
        for j in range(0, len(thist)):
            vdmt[j][i] = thist[j]**i

    return np.transpose(vdmt), coeff_rhs


def main():

    # Attempt to optimize AB3 stability region size by adding
    # additional history values.
    order = 3
    hl = order + 1
    time_hist = np.zeros(hl)
    for lind in range(0, hl):
        time_hist[lind] = -lind
    # First explicit microstep
    Avec, bvec = adams_bashforth_const(1, order, time_hist, 0)
    x0 = np.zeros(hl)
    # An initial guess full of zeros causes linear algebra issues...
    x0[0] = 1

    # POTENTIAL OBJECTIVE FUNCTIONS
    def coeff_sum(x):
        coeff_sum = 0
        for i in range(0, len(x)):
            if (i + 1) % 2 == 0:
                coeff_sum += x[i]
            else:
                coeff_sum += -x[i]
        # Minimize denominator of characteristic
        # polynomial solve
        return abs(coeff_sum)

    def coeff_norm(x):
        return np.linalg.norm(x)

    def quadrant_hunt(x):
        theta = np.linspace(0, 2*np.pi, 251)
        # implement angle check on lambda
        angles = [-np.pi, -np.pi/2, np.pi/2]
        rad_sum = 0
        max_lbdas = np.zeros(len(angles))
        for i in range(0, len(theta)):
            z = np.exp(1j*theta[i])
            denom = 0 + 0j
            for j in range(0, hl):
                denom += x[j]*z**(3-j)
            lbda = (z**4 - z**3)/denom
            # Need to check that 1 is the maximum-amplitude root
            # for this lambda
            coeffs = np.empty(hl+1, dtype=complex)
            coeffs[0] = -1
            coeffs[1] = (1 + lbda*x[0])
            for j in range(1, hl):
                coeffs[j+1] = lbda*x[j]
            maxroot = max(abs(np.roots(coeffs)))
            if maxroot < 1 + 1e-2:
                for j in range(0, len(angles)):
                    if abs(angles[j] - np.arctan2(np.imag(lbda),
                                                  np.real(lbda))) <= 1e-2:
                        # If we are at or near a target angle at which we
                        # want to maximize stability region extension,
                        # we include this radius in the objective to be
                        # minimized.
                        if abs(lbda) > max_lbdas[j]:
                            max_lbdas[j] = abs(lbda)

        rad_sum = -sum(max_lbdas)
        return rad_sum

    # Equality constraint for the order conditions.
    from scipy.optimize import LinearConstraint
    order_constraints = LinearConstraint(Avec, bvec, bvec)

    # Optimization via Scipy.
    import scipy
    res = scipy.optimize.minimize(quadrant_hunt, x0, method='trust-constr',
                                  constraints=[order_constraints],
                                  options={'verbose': 1})
    alpha = res.x

    res_orig = scipy.optimize.minimize(coeff_norm, x0, method='trust-constr',
                                       constraints=[order_constraints],
                                       options={'verbose': 1})
    alpha_orig = res_orig.x

    # Now that we have a solution, loop through theta to plot the
    # stability region.
    theta = np.linspace(0, 2*np.pi, 251)
    lbda = []
    lbda_orig = []
    lbda_std = []
    for i in range(0, len(theta)):
        z = np.exp(1j*theta[i])
        denom = 0 + 0j
        denom_orig = 0 + 0j
        for j in range(0, hl):
            denom += alpha[j]*z**(3-j)
            denom_orig += alpha_orig[j]*z**(3-j)
        lbda.append((z**4 - z**3)/denom)
        lbda_orig.append((z**4 - z**3)/denom_orig)
        # Need to check that 1 is the maximum-amplitude root for this lambda
        a = -1
        b = (1 + lbda_orig[-1]*alpha_orig[0])
        c = lbda_orig[-1]*alpha_orig[1]
        d = lbda_orig[-1]*alpha_orig[2]
        e = lbda_orig[-1]*alpha_orig[3]
        maxroot = max(abs(np.roots([a, b, c, d, e])))
        if maxroot > 1 + 1e-2:
            # There is a larger, unstable maximal root.
            lbda_orig[-1] = lbda_orig[-2]
        # Need to check that 1 is the maximum-amplitude root for this lambda
        a = -1
        b = (1 + lbda[-1]*alpha_orig[0])
        c = lbda[-1]*alpha[1]
        d = lbda[-1]*alpha[2]
        e = lbda[-1]*alpha[3]
        maxroot = max(abs(np.roots([a, b, c, d, e])))
        if maxroot > 1 + 1e-2:
            # There is a larger, unstable maximal root.
            lbda[-1] = lbda[-2]
        lbda_std.append((z**3 - z**2)/((23/12)*z**2 - (16/12)*z + (5/12)))

    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(np.real(lbda_std), np.imag(lbda_std), 'r-', label="Standard")
    plt.plot(np.real(lbda), np.imag(lbda), 'b-', label="Optimized")
    plt.plot(np.real(lbda_orig), np.imag(lbda_orig), 'g-',
             label="Original Optimized")
    plt.title("Standard vs. Optimized Stability Regions, 3rd Order")
    plt.legend()
    plt.xlabel("Re(lambda)")
    plt.ylabel("Im(lambda)")
    plt.show()


if __name__ == "__main__":
    main()
