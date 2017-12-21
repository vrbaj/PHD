import numpy as np
import matplotlib.pylab as plt
import signalz
import padasip as pa
import membership_degree as fuzz

def SNR(x, v):
    # get standard deviation of signal
    if hasattr(x, '__len__') and (not isinstance(x, str)):
        s1 = np.std(d)**2
    else:
        s1 = x**2
    # get standard deviation of noise
    if hasattr(x, '__len__') and (not isinstance(x, str)):
        s2 = np.std(v)**2
    else:
        s2 = v**2
    return 10*np.log10(s1/s2)

def roc_curve(predicted_values, actual_conditions, steps=100, interpolation_steps=100):
    # convert to boolean array
    actual_conditions = actual_conditions != 0
    # get maximum and minimum
    predicted_max = predicted_values.max()
    predicted_min = predicted_values.min()
    # range of criteria
    crits = np.linspace(predicted_min, predicted_max, steps)
    # empty variables
    tp = np.zeros(steps)
    fp = np.zeros(steps)
    tn = np.zeros(steps)
    fn = np.zeros(steps)
    for idx, crit in enumerate(crits):
        # count stuff
        tp[idx] = ((predicted_values > crit) * actual_conditions).sum()
        fn[idx] = ((predicted_values <= crit) * actual_conditions).sum()
        fp[idx] = ((predicted_values > crit) * np.invert(actual_conditions)).sum()
        tn[idx] = ((predicted_values <= crit) * np.invert(actual_conditions)).sum()
    # calculations
    total = tp + fp + tn + fn
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = (tp + tn) / total
    # AUROC integration
    points_x = np.linspace(0, 1, interpolation_steps)
    points_y = np.interp(points_x, (1-spe)[::-1], (sen)[::-1])
    auroc = np.sum(points_y*(1/interpolation_steps))
    return sen, spe, acc, auroc

def sample_entropy(x, m=2, r=0, distance="chebyshev"):
    # select r if it is not provided
    if r == 0:
        r = 0.3*np.std(x)
    # create embeded matrix
    xa = pa.preprocess.input_from_history(x, m+1)
    xb = pa.preprocess.input_from_history(x, m)[:-1]
    N = len(xa)
    A = np.zeros(N, dtype="float")
    B = np.zeros(N, dtype="float")
    # iterate over all samples
    for i in range(N):
        xia = xa[i]
        xib = xb[i]
        if distance == "chebyshev":
            da = np.max(xia-xa, axis=1)
            db = np.max(xib-xb, axis=1)
            crit = r
        elif distance == "euclidean":
            da = np.sum((xia-xa)**2,axis=1)
            db = np.sum((xib-xb)**2,axis=1)
            crit = r**2
        A[i] = np.sum(da < crit)
        B[i] = np.sum(db < crit)
    # estimate the output and insert zero padding
    out = np.zeros(len(x))
    out[m:] = -np.log10(A/B)
    return out


# Sinus Parameters
data_length = 10000
frequency = 0.1
change_point = 5000
changed_frequency = 0.11
y = np.zeros(data_length)
d = np.zeros(data_length)
for k in range(data_length):
    if k == change_point:
        frequency = changed_frequency
    y[k] = np.sin(k * frequency)
    d[k] = np.sin(k * frequency)

plt.plot(y)
plt.show()

# fuzzy system parameters
n_input_sets = 30
n_inputs = 2
rules_number = n_input_sets ** n_inputs

input_sets_centers = np.linspace(np.amin(y), np.amax(y), n_input_sets)
input_sets_width = 2 * abs(input_sets_centers[1] - input_sets_centers[0])
# print(input_sets_centers)
# print(input_sets_width)
step = abs((np.amin(y) - np.amax(y)) / (rules_number - 1))
# theta = np.linspace(np.amin(y), np.amax(y), rules_number)
theta = np.zeros((1, rules_number))
# print(theta.shape)
for i in range(rules_number):
    theta[0, i] = np.amin(y) + i * step

b = np.zeros((rules_number, 1))

# print(theta)

l = 0

mf1 = np.zeros((n_input_sets, 1))

mf2 = np.zeros((n_input_sets, 1))

input_data = np.zeros((n_inputs, 1))

P = np.zeros((rules_number, rules_number))
np.fill_diagonal(P, 10000)

output_data = 0
fuzzy_output = np.zeros((data_length, 1))
K = 0
error = np.zeros((data_length, 1))
w = np.zeros((rules_number, data_length))
print(b.shape)

for p in range(10, data_length - 1):
    input_data[0] = y[p - 1]
    input_data[1] = y[p - 10]
    output_data = y[p]
    print p
    for j in range(n_input_sets):
        if j == 0:
            # trapezoid
            params = input_sets_centers[j]
            mf1[j] = fuzz.get_r_mf_degree(input_data[0], np.array([input_sets_centers[0], input_sets_centers[1]]))
            mf2[j] = fuzz.get_r_mf_degree(input_data[1], np.array([input_sets_centers[0], input_sets_centers[1]]))
        elif j == n_input_sets - 1:
            # trapezoid
            mf1[j] = fuzz.get_l_mf_degree(input_data[0], np.array([input_sets_centers[-2],input_sets_centers[-1]]))
            mf2[j] = fuzz.get_l_mf_degree(input_data[1], np.array([input_sets_centers[-2], input_sets_centers[-1]]))
        else:
            # trojuhelnik
            mf1[j] = fuzz.get_triangular_mf_degree(input_data[0], np.array([input_sets_centers[j - 1], input_sets_centers[j], input_sets_centers[j + 1]]))
            mf2[j] = fuzz.get_triangular_mf_degree(input_data[1], np.array([input_sets_centers[j - 1], input_sets_centers[j], input_sets_centers[j + 1]]))
    l = 0
    for l1 in range(n_input_sets):
        for l2 in range(n_input_sets):
            b[l] = mf1[l1]*mf2[l2]
            l = l + 1

    b = b / sum(b)
    # print(b)
    # print("theta: ", (theta.shape))
    # print("b: ", (b.shape))
    # fuzzy_output[p] = np.dot(np.transpose(b), theta)
    for k in range(rules_number):
        fuzzy_output[p] = b[k] * theta[0, k] + fuzzy_output[p]
    print("fuzzy_out:", fuzzy_output[p])
    print("data out:", output_data)
    # print(pomoc)
    error[p] = output_data - fuzzy_output[p]
    print(error[p])
    theta = np.add(theta,K * (output_data - np.transpose(b) * theta))
    K = P * b / (np.transpose(b) * P * b + 1)
    P = P - (P * b / (np.transpose(b) * P * b + 1)) * np.transpose(b) * P
    print("theta shape:", theta.shape)
    w[:, p] = theta
e = abs(error)
plt.plot(error)
plt.show()

# get ELBND
elbnd = pa.detection.ELBND(w, e, function="sum")

# get LE
le = pa.detection.learning_entropy(w, m=1000, order=1, alpha=[6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.])

plt.plot(le)
plt.show()
plt.plot(elbnd)
plt.show()

