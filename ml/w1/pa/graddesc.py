# @author lmiguelmh
# @since 20160619T


def graddesc(x, y, iter=100, rate=0.25, theta0=0, theta1=0, resume=False):
    """"
    Gradient descent implementation - week1
    update simultaneously:
    h(x) = a0 + a1*x
    a0 = a0 - learning_rate*1/m*sum_i_1_m(h(x)-y)
    a1 = a1 - learning_rate*1/m*sum_i_1_m((h(x)-y)*x)
    """
    if len(x) != len(y):
        raise ValueError("invalid argument, len(x) != len(y)")

    __a0 = theta0
    __a1 = theta1
    for i in range(0, iter):
        # print(a0)
        # print(a1)
        hx = theta0 + theta1 * x
        # print()
        # print(hx)
        _a0 = theta0 - rate * 1 / len(x) * sum(hx - y)
        _a1 = theta1 - rate * 1 / len(x) * sum((hx - y) * x)
        theta0 = _a0
        theta1 = _a1
    if resume :
        print(" a0 = " + str(theta0))
        print(" a1 = " + str(theta1))
        print("  iterations    = " + str(iter))
        print("  learning rate = " + str(rate))
        print("  starting a0   = " + str(__a0))
        print("  starting a1   = " + str(__a1))
    return round(theta0, 6), round(theta1, 6)
