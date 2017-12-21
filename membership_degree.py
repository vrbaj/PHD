"""
.. versionadded:: 0.1

Module for estimation of membership degree to various types of fuzzy sets.

Contents of this page:

.. contents::
    :local:
    :depth: 1


Membership degree estimation
=============================
"""


import numpy as np


def get_membership_degree(x, fuzzy_set_type, fuzzy_set_params):
    membership_degree = -1
    if fuzzy_set_type == "TRIANGULAR":
        membership_degree = get_triangular_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "GAUSSIAN":
        membership_degree = get_gaussian_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "GAUSSIAN2":
        membership_degree = get_gaussian2_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "TRAPEZOIDAL":
        membership_degree = get_trapezoidal_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "R":
        membership_degree = get_r_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "L":
        membership_degree = get_l_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "BELL":
        membership_degree = get_bell_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "SIGMOID":
        membership_degree = get_sigmoid_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "DIFSIGMOID":
        membership_degree = get_difsigmoid_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "PRODSIGMOID":
        membership_degree = get_prodsigmoid_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "ZSPLINE":
        membership_degree = get_zspline_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "SSPLINE":
        membership_degree = get_sspline_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "PSPLINE":
        membership_degree = get_pspline_mf_degree(x, fuzzy_set_params)
    elif fuzzy_set_type == "SINGLETON":
        membership_degree == get_singleton_mf_degree(x, fuzzy_set_params)
    return membership_degree


def get_triangular_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by triangular  member ship function.
    Triangular membership function is specified by parameters in params

    *Args:*
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of triangular membership function [lower_limit, center, upper_limit] or
        in symmetrical case [distance_from_center_to_bounds, center]

    Returns:
        float -- value of membership degree to triangular function.

    """
    membership_degree = 0
    if params.size <= 1 or params.size > 3:
        print('get_triangular_mf_value: invalid number of parameters')
    else:
        center = params[1]
        lower_limit = 0
        upper_limit = 0
        if params.size == 2:
            lower_limit = params[1] - params[0]
            upper_limit = params[1] + params[0]
        elif params.size == 3:
            lower_limit = params[0]
            upper_limit = params[2]
        if lower_limit < x <= center:
            membership_degree = (x - lower_limit) / (center - lower_limit)
        elif center < x < upper_limit:
            membership_degree = (upper_limit - x) / (upper_limit - center)
        else:
            membership_degree = 0

    return membership_degree


def get_gaussian_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by gaussian  member ship function.
     Gaussian membership function is specified by parameters in params. Gaussian function is implemented as
     :math:`e^{\\frac{(x-center)^2}{2\sigma^2}}`


    *Args:*
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of gaussian membership function [center, sigma]

    Returns:
        float -- value of membership degree to gaussian function

    """
    membership_degree = 0
    if params.size <= 1 or params.size > 2:
        print('get_gaussian_mf_value: invalid number of parameters')
    else:
        center = params[0]
        sigma = params[1]
        membership_degree = np.exp((-0.5 * (x - center) ** 2) / (sigma ** 2))
    return membership_degree


def get_gaussian2_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by 2 gaussian  member ship function.
     Gaussian membership functions are specified by parameters in params. The value of membership degree between two
     centers of gaussian functions is always 1. Gaussian function is implemented as
     np.exp((-0.5*(x - center)**2)/(sigma**2))

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of gaussian2 membership function [center1, sigma1, center2, sigma2]

    Returns:
        float -- value of membership degree to gaussian2 function

    """
    membership_degree = 0
    if params.size <= 3 or params.size > 4:
        print('get_gaussian2_mf_value: invalid number of parameters')
    else:
        center1 = params[0]
        sigma1 = params[1]
        center2 = params[2]
        sigma2 = params[3]
        if x < center1:
            membership_degree = np.exp((-0.5 * (x - center1) ** 2) / (sigma1 ** 2))
        elif center1 <= x <= center2:
            membership_degree = 1
        elif x > center2:
            membership_degree = np.exp((-0.5 * (x - center2) ** 2) / (sigma2 ** 2))
    return membership_degree


def get_trapezoidal_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by trapezoidal  member ship function.
     Trapezoidal membership function is specified by parameters in params.

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of trapezoidal membership function [a, b, c, d]

    Returns:
        float -- value of membership degree to trapezoidal function

    """
    membership_degree = 0
    if params.size <= 3 or params.size > 4:
        print("get_trapezoidal_mf_degree: invalid number of parameters")
    else:
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        if x < a or x > d:
            membership_degree = 0
        elif a <= x <= b:
            membership_degree = (x - a) / (b - a)
        elif b <= x <= c:
            membership_degree = 1
        elif c <= x <= d:
            membership_degree = (d - x) / (d - c)
    return membership_degree


def get_r_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by R  member ship function.
     R membership function is specified by parameters in params.

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of R membership function [c, d]

    Returns:
        float -- value of membership degree to R function

    """
    membership_degree = 0
    if params.size <= 1 or params.size > 2:
        print("get_r_mf_degree: invalid number of parameters")
    else:
        c = params[0]
        d = params[1]
        if x > d:
            membership_degree = 0
        elif c <= x <= d:
            membership_degree = (d - x) / (d - c)
        elif x < c:
            membership_degree = 1
    return membership_degree


def get_l_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by L  member ship function.
     L membership function is specified by parameters in params.

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of L membership function [a, b]

    Returns:
        float -- value of membership degree to L function

    """
    membership_degree = 0
    if params.size <= 1 or params.size > 2:
        print("get_l_mf_degree: invalid number of parameters")
    else:

        a = params[0]
        b = params[1]
        if x > b:
            membership_degree = 1
        elif a <= x <= b:
            membership_degree = (x - a) / (b - a)
        elif x < a:
            membership_degree = 0
    return membership_degree


def get_bell_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by bell-shaped  member ship function.
     Bell-shaped membership function is specified by parameters in params. Bell-shaped function is implemented as
     1/(1+abs((x-center)/a)**2b

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of bell-shaped membership function [a, b, center]

    Returns:
        float -- value of membership degree to L function

    """
    membership_degree = 0
    if params.size <= 2 or params.size > 3:
        print("get_bell_mf_degree: invalid number of parameters")
    else:

        a = params[0]
        b = params[1]
        c = params[2]
        membership_degree = 1/(1 + (abs((x - c) / a) ** (2 * b)))
    return membership_degree


def get_sigmoid_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by sigmoidal  member ship function.
     Sigmoidal membership function is specified by parameters in params. This function is implemented as
     1/(1+exp(-a*(x-center)).

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of sigmoid membership function [center, a]

    Returns:
        float -- value of membership degree to L function

    """
    membership_degree = 0
    if params.size <= 1 or params.size > 2:
        print("get_sigmoid_mf_degree: invalid number of parameters")
    else:

        center = params[0]
        a = params[1]
        membership_degree = 1/(1 + np.exp(-a * (x - center)))
    return membership_degree


def get_difsigmoid_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by difference of two sigmoid
    member ship function. Sigmoid membership function is specified by parameters in params.
    This function is implemented as 1/(1+exp(-a*(x-center)).

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of sigmoid membership functions [center1, a1, center2, a2]

    Returns:
        float -- value of membership degree to difference of sigmoid functions

    """
    membership_degree = 0
    if params.size <= 3 or params.size > 4:
        print("get_sigmoid_mf_degree: invalid number of parameters")
    else:

        center1 = params[0]
        a1 = params[1]
        center2 = params[2]
        a2 = params[3]
        membership_degree = 1 / (1 + np.exp(-a1 * (x - center1))) - 1 / (1 + np.exp(-a2 * (x - center2)))
    return membership_degree


def get_prodsigmoid_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by product of two sigmoid
    member ship function. Sigmoid membership function is specified by parameters in params.
    This function is implemented as 1/(1+exp(-a*(x-center)).

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of sigmoid membership functions [center1, a1, center2, a2]

    Returns:
        float -- value of membership degree to product of sigmoid functions

    """
    membership_degree = 0
    if params.size <= 3 or params.size > 4:
        print("get_sigmoid_mf_degree: invalid number of parameters")
    else:
        center1 = params[0]
        a1 = params[1]
        center2 = params[2]
        a2 = params[3]
        membership_degree = 1 / (1 + np.exp(-a1 * (x - center1))) * 1 / (1 + np.exp(-a2 * (x - center2)))
    return membership_degree


def get_zspline_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by Z spline based
    member ship function. Z spline membership function is specified by parameters in params.
    This function is implemented as 1-2*((x - a) \ (b - a)) ** 2 for a <= x <= (a + b) \ 2 and
    2*((x - b) \ (b - a)) ** 2 for (a + b) / 2 <= x <= b. For x <= a is output 1,
    for x >= b is output 0.

    Args:
        x (float):  x in which we want to estimate membership degree
        params (numpy array):  specification of sigmoid membership functions [a, b]

    Returns:
        float -- value of membership degree to product of Z-spline based function

    """
    membership_degree = 0
    if params.size <= 1 or params.size > 2:
        print("get_get_zspline_mf_degree: invalid number of parameters")
    else:
        a = float(params[0])
        b = float(params[1])
        if x <= a:
            membership_degree = 1
        elif a <= x <= (a + b) / 2:
            membership_degree = 1 - 2 * ((x - a) / (b - a)) ** 2
        elif (a + b) / 2 <= x <= b:
            membership_degree = 2 * ((x - b) / (b - a)) ** 2
        elif b <= x:
            membership_degree = 0
    return membership_degree


def get_sspline_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by S spline based
    member ship function. S spline membership function is specified by parameters in params.
    This function is implemented as 1-2*((x - a) \ (b - a)) ** 2 for (a + b) / 2 <= x <= b and
    2*((x - b) \ (b - a)) ** 2 for a <= x <= (a + b) \ 2. For x <= a is output 0,
    for x >= b is output 1.

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of sigmoid membership functions [a, b]

    Returns:
        float -- value of membership degree to product of S-spline based function

    """
    membership_degree = 0
    if params.size <= 1 or params.size > 2:
        print("get_get_zspline_mf_degree: invalid number of parameters")
    else:
        a = params[0]
        b = params[1]
        if x >= b:
            membership_degree = 1
        elif a <= x <= (a + b) / 2:
            membership_degree = 2 * ((x - a) / (b - a)) ** 2
        elif (a + b) / 2 <= x <= b:
            membership_degree = 1 - 2 * ((x - b) / (b - a)) ** 2
        elif x >= a:
            membership_degree = 0
    return membership_degree


def get_pspline_mf_degree(x, params):
    """
    This function returns the membership degree of x to fuzzy set specified by S spline based
    member ship function. S spline membership function is specified by parameters in params.
    This function is implemented as combination of sspline a zspline functions.

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  specification of pspline membership functions [a, b, c, d]

    Returns:
        float -- value of membership degree to product of Z-spline based function

    """
    membership_degree = 0
    if params.size <= 3 or params.size > 4:
        print("get_get_zspline_mf_degree: invalid number of parameters")
    else:
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        if x <= a:
            membership_degree = 0
        elif a <= x <= (a + b) / 2:
            membership_degree = 2 * ((x - a) / (b - a)) ** 2
        elif b <= x <= c:
            membership_degree = 1
        elif c <= x <= (c + d) / 2:
            membership_degree = 1 - 2 * ((x - c) / (d - c)) ** 2
        elif (c + d) / 2 <= x <= d:
            membership_degree = 2 * ((x - c) / (d - c)) ** 2
        elif x >= d:
            membership_degree = 0
    return membership_degree


def get_singleton_mf_degree(x, params):
    """
    This function returns the membership degree of x to singleton fuzzy set specified by its center in params.

    Args:
        x (number):  x in which we want to estimate membership degree
        params (numpy array):  center of singleton

    Returns:
        float -- value of membership degree to fuzzy singleton

    """
    if x == params:
        membership_degree = 1
    elif x != params:
        membership_degree = 0
    return membership_degree


test_x = 11
test_fuzzy_set_params = np.array([1, 4, 5, 10])
# test_fuzzy_set_type = "TRIANGULAR"
# test_fuzzy_set_type = "GAUSSIAN"
# test_fuzzy_set_type = "TRAPEZOIDAL"
# test_fuzzy_set_type = "R"
# test_fuzzy_set_type = "L"
# test_fuzzy_set_type = "GAUSSIAN2"
# test_fuzzy_set_type = "SIGMOID"
# test_fuzzy_set_type = "DIFSIGMOID"
# test_fuzzy_set_type = "PRODSIGMOID"
# test_fuzzy_set_type = "ZSPLINE"
# test_fuzzy_set_type = "SSPLINE"
test_fuzzy_set_type = "PSPLINE"
mf = get_membership_degree(test_x, test_fuzzy_set_type, test_fuzzy_set_params)
print(mf)
