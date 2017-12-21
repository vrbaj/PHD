import numpy as np
import matplotlib.pylab as plt
import signalz
import padasip as pa
import membership_degree as fuzz

# Sinus Parameters
data_length = 10000
frequency = 0.1
change_point = 5000
changed_frequency = 0.11
y = np.zeros(data_length)
for k in range(data_length):
    if k == change_point:
        frequency = changed_frequency
    y[k] = np.sin(k * frequency)

plt.plot(y)
plt.show()

# fuzzy system parameters
n_input_sets = 30
n_inputs = 2
rules_number = n_input_sets ** n_inputs

input_sets_centers = np.linspace(np.amin(y), np.amax(y), n_input_sets)
input_sets_width = 2 * abs(input_sets_centers[1] - input_sets_centers[0])
print(input_sets_centers)
print(input_sets_width)
step = abs((np.amin(y) - np.amax(y)) / (rules_number - 1))
# theta = np.linspace(np.amin(y), np.amax(y), rules_number)
theta = np.zeros((1, rules_number))
print(theta.shape)
for i in range(rules_number):
    theta[0, i] = np.amin(y) + i * step

b = np.zeros((rules_number, 1))

print(theta)

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
    theta = theta + K * (output_data - np.transpose(b) * theta)
    K = P * b / (np.transpose(b) * P * b + 1)
    P = P - (P * b / (np.transpose(b) * P * b + 1)) * np.transpose(b) * P


plt.plot(error)
plt.show()
