import numpy as np


def kaps_nonstiff(y, eps):

    # Kaps' first problem - nonstiff part.
    dy = np.zeros(2)
    dy[0] = -2*y[0]
    dy[1] = y[0] - y[1] - y[1]*y[1]
    return dy


def kaps_stiff(y, eps):

    # Kaps' first problem - stiff part.
    einv = 1/eps
    dy = np.zeros(2)
    dy[0] = -einv*y[0] + einv*y[1]*y[1]
    dy[1] = 0
    return dy


def kaps_exact(t):
    y_exact = np.zeros(2)
    y_exact[1] = np.exp(-t)
    y_exact[0] = y_exact[1]*y_exact[1]
    return y_exact


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


def imex_adams(y, h, eps, state_hist, rhs_hist, rhsns_hist, time_hist,
               order):

    new_nonstiff = adams_inc(y, h, order,
                             time_hist, rhsns_hist,
                             time_hist[0])

    y_new = np.zeros(2)
    einv = 1/eps

    # Stiff contribution to y2 is zero, so we can do an implicit solve
    # directly without any Newton nonsense...
    if order == 1:
        y_new[1] = state_hist[0, 1] + new_nonstiff[1]
        y_new[0] = (1/(1 + h*einv))*(state_hist[0, 0] +
                                     h*einv*y_new[1]*y_new[1] +
                                     new_nonstiff[0])
    elif order == 2:
        # Adams way.
        y_new[1] = state_hist[0, 1] + \
                (1.0/2.0)*h*rhs_hist[0, 1] + \
                new_nonstiff[1]
        y_new[0] = (1/(1 + (1.0/2.0)*h*einv))*(state_hist[0, 0] +
                                               (1.0/2.0) * h * rhs_hist[0, 0] +
                                               (1.0/2.0) * h * einv *
                                               y_new[1]*y_new[1] +
                                               new_nonstiff[0])
    elif order == 3:
        # Adams way.
        y_new[1] = state_hist[0, 1] + \
                (2.0/3.0)*h*rhs_hist[0, 1] - \
                (1.0/12.0)*h*rhs_hist[1, 1] + \
                new_nonstiff[1]
        y_new[0] = (1/(1 + (5.0/12.0)*h*einv)) * \
            (state_hist[0, 0] +
             (2.0/3.0)*h*rhs_hist[0, 0] -
             (1.0/12.0)*h*rhs_hist[1, 0] +
             (5.0/12.0) * h * einv *
             y_new[1]*y_new[1] +
             new_nonstiff[0])
    else:
        # Adams way.
        y_new[1] = state_hist[0, 1] + \
                (19.0/24.0)*h*rhs_hist[0, 1] - \
                (5.0/24.0)*h*rhs_hist[1, 1] + \
                (1.0/24.0)*h*rhs_hist[2, 1] + \
                new_nonstiff[1]
        y_new[0] = (1/(1 + (9.0/24.0)*h*einv)) * \
            (state_hist[0, 0] +
             (19.0/24.0)*h*rhs_hist[0, 0] -
             (5.0/24.0)*h*rhs_hist[1, 0] +
             (1.0/24.0)*h*rhs_hist[2, 0] +
             (9.0/24.0) * h * einv *
             y_new[1]*y_new[1] +
             new_nonstiff[0])

    return (y_new, kaps_nonstiff(y_new, eps),
            kaps_nonstiff(y_new, eps) + kaps_stiff(y_new, eps))


def adams_bashforth(y, h, eps, order, state_hist, rhs_hist):

    # These exist mostly to see where the stiffness bound is.
    if order == 1:
        y_new = y + h*(rhs_hist[0])
    elif order == 2:
        y_new = y + h*((3.0/2.0)*rhs_hist[0] - (1.0/2.0)*rhs_hist[1])
    elif order == 3:
        y_new = y + h*((23.0/12.0)*rhs_hist[0] -
                       (16.0/12.0)*rhs_hist[1] +
                       (5.0/12.0)*rhs_hist[2])
    elif order == 4:
        y_new = y + h*((55.0/24.0)*rhs_hist[0] -
                       (59.0/24.0)*rhs_hist[1] +
                       (37.0/24.0)*rhs_hist[2] -
                       (9.0/24.0)*rhs_hist[3])

    return (y_new, kaps_nonstiff(y_new, eps),
            kaps_nonstiff(y_new, eps) + kaps_stiff(y_new, eps))


def rk4(y, h, eps, order, state_hist, rhs_hist):

    # These exist mostly to see where the stiffness bound is.
    k1 = kaps_nonstiff(y, eps) + kaps_stiff(y, eps)
    k2 = kaps_nonstiff(y + h*k1/2, eps) + kaps_stiff(y + h*k1/2, eps)
    k3 = kaps_nonstiff(y + h*k2/2, eps) + kaps_stiff(y + h*k2/2, eps)
    k4 = kaps_nonstiff(y + h*k3, eps) + kaps_stiff(y + h*k3, eps)
    y_new = y + h*((1.0/6.0)*k1 + (1.0/3.0)*k2 + (1.0/3.0)*k3 + (1.0/6.0)*k4)

    return (y_new, kaps_nonstiff(y_new, eps),
            kaps_nonstiff(y_new, eps) + kaps_stiff(y_new, eps))


def main():

    # Time integration of Kaps' problem to test
    # some new IMEX methods.
    t_start = 0
    t_end = 1
    # dts = [0.0025, 0.001, 0.0005, 0.0001]
    dts = [0.005, 0.0025, 0.001, 0.0005]
    orders = [1, 2, 3, 4]
    # orders = [4]
    errors = np.zeros((4, 4))
    stepper = 'imex'
    eps = 0.001

    from pytools.convergence import EOCRecorder
    for j, order in enumerate(orders):
        eocrec = EOCRecorder()
        for k, dt in enumerate(dts):

            y_old = np.zeros(2)
            y_old[0] = 1
            y_old[1] = 1

            times = []
            states0 = []
            states0.append(y_old[0])
            states1 = []
            states1.append(y_old[1])
            exact_states0 = []
            exact_states1 = []
            exact_states0.append(y_old[0])
            exact_states1.append(y_old[1])

            t = t_start
            times.append(t)
            step = 0
            rhs_hist = np.empty((order, 2), dtype=y_old.dtype)
            rhs_stiff_hist = np.empty((order, 2), dtype=y_old.dtype)
            rhsns_hist = np.empty((order, 2), dtype=y_old.dtype)
            state_hist = np.empty((order, 2), dtype=y_old.dtype)
            time_hist = np.empty(order)
            rhs_hist[0] = (kaps_stiff(y_old, eps) + kaps_nonstiff(y_old, eps))
            rhsns_hist[0] = kaps_nonstiff(y_old, eps)
            rhs_stiff_hist[0] = kaps_stiff(y_old, eps)
            state_hist[0] = y_old
            time_hist[0] = t
            tiny = 1e-15
            while t < t_end - tiny:
                if step < order-1:
                    # "Bootstrap" using known exact solution.
                    y = kaps_exact(t + dt)
                    dy_ns = kaps_nonstiff(y, eps)
                    dy_full = dy_ns + kaps_stiff(y, eps)
                else:
                    # Step normally - we have all the history we need.
                    if stepper == 'imex':
                        y, dy_ns, dy_full = imex_adams(y_old, dt, eps,
                                                       state_hist,
                                                       rhs_stiff_hist,
                                                       rhsns_hist, time_hist,
                                                       order)
                    elif stepper == 'rk4':
                        y, dy_ns, dy_full = rk4(y_old, dt, eps,
                                                order, state_hist,
                                                rhs_hist)
                    else:
                        y, dy_ns, dy_full = adams_bashforth(y_old, dt, eps,
                                                            order, state_hist,
                                                            rhs_hist)
                # Rotate histories.
                for i in range(order-1, 0, -1):
                    rhs_hist[i] = rhs_hist[i-1]
                    rhsns_hist[i] = rhsns_hist[i-1]
                    time_hist[i] = time_hist[i-1]
                    state_hist[i] = state_hist[i-1]
                    rhs_stiff_hist[i] = rhs_stiff_hist[i-1]
                rhs_hist[0] = dy_full
                rhs_stiff_hist[0] = dy_full - dy_ns
                rhsns_hist[0] = dy_ns
                state_hist[0] = y
                time_hist[0] = t + dt
                # Append to states and prepare for next step.
                states0.append(y[0])
                states1.append(y[1])
                t += dt
                times.append(t)
                ey = kaps_exact(t)
                exact_states0.append(ey[0])
                exact_states1.append(ey[1])
                y_old = y
                step += 1

            errors[j, k] = np.linalg.norm(y - kaps_exact(t))
            eocrec.add_data_point(dt, errors[j, k])

        print("------------------------------------------------------")
        print("expected order: ", order)
        print("------------------------------------------------------")
        print(eocrec.pretty_print())

        orderest = eocrec.estimate_order_of_convergence()[0, 1]
        print("Estimated order of accuracy: ", orderest)

    z1 = np.polyfit(np.log10(np.array(dts)),
                    np.log10(np.abs(np.array(errors[0, :]))), 1)
    z2 = np.polyfit(np.log10(np.array(dts)),
                    np.log10(np.abs(np.array(errors[1, :]))), 1)
    z3 = np.polyfit(np.log10(np.array(dts)),
                    np.log10(np.abs(np.array(errors[2, :]))), 1)
    z4 = np.polyfit(np.log10(np.array(dts)),
                    np.log10(np.abs(np.array(errors[3, :]))), 1)
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)
    p3 = np.poly1d(z3)
    p4 = np.poly1d(z4)
    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[0, :]))), c='g')
    plt.plot(np.log10(np.array(dts)), p1(np.log10(np.array(dts))), 'g--')
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[1, :]))), c='b')
    plt.plot(np.log10(np.array(dts)), p2(np.log10(np.array(dts))), 'b--')
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[2, :]))), c='r')
    plt.plot(np.log10(np.array(dts)), p3(np.log10(np.array(dts))), 'r--')
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[3, :]))), c='k')
    plt.plot(np.log10(np.array(dts)), p4(np.log10(np.array(dts))), 'k--')
    # plt.legend(["EOC=%.6f" % (z1[0]), "EOC=%.6f" % (z2[0]),
    #             "EOC=%.6f" % (z3[0]), "EOC=%.6f" % (z4[0]),
    #             'Order 1 Data, State', 'Order 2 Data, State',
    #             'Order 3 Data, State', 'Order 4 Data, State'])
    plt.xlabel('log10(dt)')
    plt.ylabel('log10(err)')
    plt.ylim((-16, -1))
    plt.title('Log-Log Error Timestep Plots: IMEX Adams')
    plt.show()


if __name__ == "__main__":
    main()
