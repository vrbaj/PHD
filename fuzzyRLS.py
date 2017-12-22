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
data_length = 1000
frequency = 0.1
change_point = 614 #3140
changed_frequency = 0.11
y = np.zeros(data_length)
d = np.zeros(data_length)
for k in range(data_length):
    if k >= change_point:

        y[k] = 1 * np.sin(k * changed_frequency)
        d[k] = 1*np.sin(k * changed_frequency)
    else:
        y[k] = np.sin(k * frequency)
        d[k] = np.sin(k * frequency)

plt.plot(y)
plt.show()

# fuzzy system parameters
n_input_sets = 5
n_inputs = 2
rules_number = n_input_sets ** n_inputs

input_sets_centers = np.linspace(np.amin(y), np.amax(y), n_input_sets)
input_sets_width = abs(input_sets_centers[1] - input_sets_centers[0])
# print(input_sets_centers)
# print(input_sets_width)
step = abs((np.amin(y) - np.amax(y)) / (rules_number - 1))
# theta = np.linspace(np.amin(y), np.amax(y), rules_number)
theta = np.zeros((rules_number, 1))
# print(theta.shape)
for i in range(rules_number):
    theta[i, 0] = np.amin(y) + i * step

b = np.zeros((rules_number, 1))

# print(theta)

l = 0

mf1 = np.zeros((n_input_sets, 1))

mf2 = np.zeros((n_input_sets, 1))

input_data = np.zeros((n_inputs, 1))

P = np.zeros((rules_number, rules_number))
np.fill_diagonal(P, 10 ** 9)

output_data = 0
fuzzy_output = np.zeros((data_length, 1))
K = np.zeros((rules_number, 1))
error = np.zeros((data_length, 1))
w = np.zeros((data_length, rules_number))
print(b.shape)

for p in range(10, data_length - 1):
    input_data[0] = y[p - 1]
    input_data[1] = y[p - 10]
    output_data = y[p]
    w[p, :] = np.transpose(theta[:])
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
        fuzzy_output[p] = b[k] * theta[k, 0] + fuzzy_output[p]
    # print("fuzzy_out:", fuzzy_output[p])
    # print("data out:", output_data)
    # print(pomoc)
    error[p] = output_data - fuzzy_output[p]

    # print("b shape:", b.shape)
    # pomoc1 = np.dot(np.transpose(b), theta)
    # print("pomoc1 shape: ", pomoc1.shape)
    K = np.matmul(P, b)
    jmenovatel = np.matmul(np.transpose(b), P)
    jmenovatel = np.matmul(jmenovatel, b)
    jmenovatel = np.add(jmenovatel, 1)
    K = np.divide(K, jmenovatel)
    # / (np.transpose(b) * P * b + 1)
    # print("HIOVNO:",(output_data - fuzzy_output))
    hovno = np.matmul(K, error[p])

    theta = np.add(theta, K * error[p])

    pomocna = np.matmul(np.transpose(b), P)
    pomocna = np.matmul(pomocna, b)
    pomocna = pomocna + 1
    pomocna = np.matmul(pomocna, np.transpose(b))
    pomocna = np.matmul(pomocna,P)
    pomocna = np.multiply(pomocna, -1)
    # P = P - (P * b / (np.transpose(b) * P * b + 1)) * np.transpose(b) * P
    np.add(P, pomocna)
    # print("P:", P)
    print("p =", p)
    # w[p, :] = np.transpose(theta[:])
    print(error[p])
    print(np.amax(theta))
e = np.abs(error)
e = np.transpose(e)
plt.plot(abs(error))
plt.title('err')
plt.show()

# get ELBND
elbnd = pa.detection.ELBND(w, e, function="sum")
a = [1000., 20000., 10000., 1000000., 5000., 500., 105., 10.]
# get LE m = 10 nebo 100
le = pa.detection.learning_entropy((w), m=100, order=1, alpha=a)
dw = np.zeros(w.shape)
dw[0:-1] = np.abs(np.diff(w, axis=0))

plt.plot(le)
plt.title('LE')
plt.show()
plt.plot(elbnd)
plt.title('ELBND')
plt.show()

plt.plot(dw)
plt.show()

