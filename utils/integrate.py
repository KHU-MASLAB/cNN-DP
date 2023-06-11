import numpy as np
import scipy
from scipy.interpolate import Akima1DInterpolator as akispl


def YDot(func_XDDot, t, Y):  # dY
    statedim = int(len(Y) / 2)
    X = Y[:statedim]
    XDot = Y[statedim:]
    dY = np.concatenate([XDot, func_XDDot(t, X, XDot)], axis=0)
    return dY


def RK4(func_XDDot, h, t, Y):
    k1 = YDot(func_XDDot, t, Y)
    k2 = YDot(func_XDDot, t + h / 2, Y + h * k1 / 2)
    k3 = YDot(func_XDDot, t + h / 2, Y + h * k2 / 2)
    k4 = YDot(func_XDDot, t + h, Y + h * k3)
    dY = k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    return dY


def RK5(func_XDDot, h, t, Y):
    k1 = YDot(func_XDDot, t, Y)
    k2 = YDot(func_XDDot, t + h / 4, Y + h * k1 / 4)
    k3 = YDot(func_XDDot, t + h / 4, Y + h * k1 / 8 + h * k2 / 8)
    k4 = YDot(func_XDDot, t + h / 2, Y - h * k2 / 2 + h * k3)
    k5 = YDot(func_XDDot, t + 3 * h / 4, Y + 3 * h * k1 / 16 + 9 * h * k4 / 16)
    k6 = YDot(
        func_XDDot,
        t + h,
        Y
        - 3 * h * k1 / 7
        + 2 * h * k2 / 7
        + 12 * h * k3 / 7
        - 12 * h * k4 / 7
        + 8 * h * k5 / 7,
    )
    dY = (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6) / 90
    return dY


def newmarkbetaint(ddY, dt, y0, dy0, gamma=1 / 2, beta=1 / 4):
    """
    https://en.wikipedia.org/wiki/Newmark-beta_method
    Average constant acceleration (Middle point rule, gamma=0.5, beta=0.25) is unconditionally stable.
    """
    Y = np.zeros_like(ddY)
    dY = np.zeros_like(ddY)
    Y[0] = y0
    dY[0] = dy0
    for i in range(1, ddY.shape[0]):
        dY[i] = dY[i - 1] + (1 - gamma) * dt * ddY[i - 1] + gamma * dt * ddY[i]
        Y[i] = (
            Y[i - 1]
            + dt * dY[i - 1]
            + dt**2 / 2 * ((1 - 2 * beta) * ddY[i - 1] + 2 * beta * ddY[i])
        )
    return Y, dY


def galphaint(ddY, dt, y0, dy0, rho_inf=1):
    """
    http://www.dymoresolutions.com/AnalysisControls/GeneralizedAlpha.pdf
    rho_inf=1 is equivalent to Newmark-beta method
    """
    alpha_m = (2 * rho_inf - 1) / (rho_inf + 1)
    alpha_f = (rho_inf) / (rho_inf + 1)
    gamma = 1 / 2 - alpha_m + alpha_f
    beta = 1 / 4 * (1 - alpha_m + alpha_f) ** 2

    Y = np.zeros_like(ddY)
    dY = np.zeros_like(ddY)
    Y[0] = y0
    dY[0] = dy0
    for i in range(1, ddY.shape[0]):
        dY[i] = dY[i - 1] + (1 - gamma) * dt * ddY[i - 1] + gamma * dt * ddY[i]
        Y[i] = (
            Y[i - 1]
            + dt * dY[i - 1]
            + dt**2 / 2 * ((1 - 2 * beta) * ddY[i - 1] + 2 * beta * ddY[i])
        )
    return Y, dY


def rkint(
    func_XDDot, X_0: np.array, XDot_0: np.array, t_array: np.array, method: str = "rk4"
):
    """
    Applies Runge-Kutta numerical second-order integration on multi-dimensional state-space represented vector.
    @param func_XDDot: func_XDDot(t, X,XDot): returns XDDot (1D array)
    @type func_XDDot: function
    @param X_0: Initial values
    @type X_0: 1D array or list
    @param XDot_0:
    @type XDot_0:
    @param t_array: Discrete time steps
    @type t_array: 1D array or list
    @param method: Supports 'rk4' or 'rk5'
    @type method: str
    @return: X, XDot, XDDot
    @rtype: 2D array
    """
    t_array = t_array.flatten()
    X_0 = X_0.flatten()
    XDot_0 = XDot_0.flatten()

    t = t_array[0]
    h = t_array[1] - t_array[0]
    steps = len(t_array) - 1
    Y = np.concatenate([X_0, XDot_0], axis=0)  # State vector [diff0, diff1]
    dY = np.concatenate([XDot_0, func_XDDot(t, X_0, XDot_0)], axis=0)

    rk_Y = np.zeros((steps + 1, len(X_0) * 2))  # pos and vel
    rk_dY = np.zeros_like(rk_Y)  # update gradients

    for idx in range(steps + 1):
        # save
        t_array[idx] = t
        rk_Y[idx] = Y
        rk_dY[idx] = dY

        # calculate
        if method.lower() == "rk4":
            dY = RK4(func_XDDot, h, t, Y)
        elif method.lower() == "rk5":
            dY = RK5(func_XDDot, h, t, Y)

        # update
        Y = Y + h * dY
        t = t + h

        # print
        if (idx + 1) % 10000 == 0:
            print(f"Solving t={t:.5f} ({idx + 1}/{steps + 1})")

    X = rk_Y[:, : len(X_0)]
    XDot = rk_Y[:, len(X_0) :]
    XDDot = rk_dY[:, len(X_0) :]
    return X, XDot, XDDot


def PolyFitCustomized(x, y, deg_start, deg_end):
    A = np.zeros((deg_end - deg_start + 1, deg_end - deg_start + 1))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = np.sum(x ** (deg_start * 2 + i + j))
    B = np.zeros(deg_end - deg_start + 1)
    for i in range(B.shape[0]):
        B[i] = np.sum(np.power(x, deg_start + i) * y)
    coefficients = np.linalg.solve(A, B)  # low->high order
    # Add low order coefficients
    if deg_start > 0:
        coefficients = np.concatenate((np.zeros(deg_start), coefficients), axis=0)
    return coefficients


def rkint_detrended(
    func_XDDot,
    X_0: np.array,
    XDot_0: np.array,
    t_array: np.array,
    method: str = "rk4",
    order: int = 4,
):
    """
    Applies Runge-Kutta numerical second-order integration on multi-dimensional state-space represented vector with the detrending algorithm from:
    Pan, C., Zhang, R., Luo, H., & Shen, H. (2016).
    Baseline correction of vibration acceleration signals with inconsistent initial velocity and displacement.
    Advances in Mechanical Engineering, 8(10).

    @param func_XDDot: func_XDDot(t, X,XDot): returns XDDot (1D array)
    @type func_XDDot: function
    @param X_0: Initial values
    @type X_0: 1D array or list
    @param XDot_0:
    @type XDot_0:
    @param t_array: Discrete time steps
    @type t_array: 1D array or list
    @param method: Supports 'rk4' or 'rk5'
    @type method: str
    :param order: Detrending correction order
    :type order: int
    :return: X, XDot, XDDot
    :rtype: each 2D array
    """
    t_array = t_array.flatten()
    X_0 = X_0.flatten()
    XDot_0 = XDot_0.flatten()

    statedim = int(X_0.shape[0])
    t_0 = t_array[0]
    t_end = t_array[-1]
    h = t_array[1] - t_array[0]
    # Save uncorrected integration
    X_uncorrected, XDot_uncorrected, XDDot_uncorrected = rkint(
        func_XDDot, X_0, XDot_0, t_array, method=method
    )
    # STEP 1
    # Correct acceleration with velocity
    funclist_XDDot_corrected1 = []
    for i in range(statedim):
        xdot_uncorrected = XDot_uncorrected[:, i]
        xddot_corrective1 = np.zeros_like(t_array)  # basis
        coefficients = PolyFitCustomized(
            t_array, xdot_uncorrected, 1, order + 1
        )  # init value=0
        for order, coeff in enumerate(coefficients):
            if coeff == 0:  # 상수차수의 미분항 무시 (=0)
                continue
            else:
                xddot_corrective1 += order * coeff * t_array ** (order - 1)
        xddot_corrected1 = XDDot_uncorrected[:, i] - xddot_corrective1
        funclist_XDDot_corrected1.append(akispl(t_array, xddot_corrected1))

    def func_XDDot_corrected1(t, X, XDot):
        XDDot = np.zeros_like(X)
        for j in range(len(X)):
            XDDot[j] = funclist_XDDot_corrected1[j](t)
        return XDDot

    # Integrate velocity corrected acceleration
    X_corrected1, XDot_corrected1, XDDot_corrected1 = rkint(
        func_XDDot_corrected1, X_0, XDot_0, t_array, method=method
    )

    # STEP 2
    # Correct acceleration with displacement from velocity-corrected acceleration
    funclist_XDDot_corrected2 = []
    for i in range(statedim):
        xddot_corrective2 = np.zeros_like(t_array)
        coefficients = PolyFitCustomized(t_array, X_corrected1[:, i], 2, order + 2)
        for order, coeff in enumerate(coefficients):
            if coeff == 0:
                continue
            else:
                xddot_corrective2 += (
                    order * (order - 1) * coeff * t_array ** (order - 2)
                )
        xddot_corrected2 = XDDot_corrected1[:, i] - xddot_corrective2
        funclist_XDDot_corrected2.append(
            scipy.interpolate.Akima1DInterpolator(t_array, xddot_corrected2)
        )

    def func_XDDot_corrected2(t, X, XDot):
        XDDot = np.zeros_like(X)
        for j in range(len(X)):
            XDDot[j] = funclist_XDDot_corrected2[j](t)
        return XDDot

    # Integrate displacement corrected acceleration
    X_corrected2, XDot_corrected2, XDDot_corrected2 = rkint(
        func_XDDot_corrected2, X_0, XDot_0, t_array, method=method
    )
    return X_corrected2, XDot_corrected2, XDDot_corrected2
