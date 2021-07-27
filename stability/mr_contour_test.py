import numpy as np


def main():

    # Determining max(z) contours for each SR in first
    # order MR.
    order = 1
    sr = 1
    # Modify this to see what effect stiffness has on
    # the stability of the explicit part of the SBDF.
    mu = -5

    # root locus: loop through theta.
    imags = np.linspace(-30, 30, 500)
    reals = np.linspace(-30, 30, 500)
    zs = np.zeros((500, 500))

    # Now, calculate the stability bound for a
    # single part of the IMEX scheme.
    for i, re in enumerate(reals):
        print("i = ", i)
        for j, im in enumerate(imags):

            lbda = re + 1j*im
            if order == 1:
                s = 0
                for lind in range(0, sr):
                    s += (lbda/sr)*(lind/sr)*((1 + lbda)/(1 - mu) - 1)
                crit = (abs((1/(1 - mu))*(1 + lbda + s)))
                zs[j, i] = crit
            else:
                raise ValueError("Higher orders not yet implemented")

    import matplotlib.pyplot as plt
    # Contour plot of the maximum amplification factor
    plt.clf()
    plt.contourf(reals, imags, zs, levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                           0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3,
                                           1.4, 1.5])
    plt.title("IMEX Adams MR Order {order} Maximum Amplification "
              "Factor, Explicit, SR = {sr}, mu = {mu}".format(order=order,
                                                              sr=sr, mu=mu))
    plt.xlabel("Re(lambda)")
    plt.ylabel("Im(lambda)")
    plt.colorbar(label="z")
    plt.contour(reals, imags, zs, levels=[0, 1], colors=['red'])
    plt.show()


if __name__ == "__main__":
    main()
