import numpy as np


def forward(t_model, s_model, fw):
    # As in equation 15.12
    rv = s_model * np.transpose(t_model) * fw
    # Normalize
    return rv / np.sum(rv)


def backward(t_model, s_model, bw):
    # As in equation 15.13
    return t_model * s_model * bw


def forward_backward(ev, prior):
    # Setup of forward and backward messages, as well as smoothed estimates
    t = len(ev)
    fv = [np.matrix([0.0, 0.0]) for _ in range(t + 1)]
    b = np.matrix([[1.0], [1.0]])
    sv = [np.matrix([0.0, 0.0]) for _ in range(t)]
    fv[0] = prior

    # For-loops as described in the algorithm in figure 15.4 (p.576)
    for i in range(0, t):
        fv[i + 1] = forward(T, (O_false, O_true)[ev[i]], fv[i])

    for i in range(t, 0, -1):
        smoothed = np.multiply(fv[i], b)
        # Normalize
        sv[i - 1] = smoothed / smoothed.sum()
        b = backward(T, (O_false, O_true)[ev[i - 1]], b)
        print(b)

    return sv


# Transition model and sensor models given true or false as evidence
T = np.matrix([[0.7, 0.3],
               [0.3, 0.7]])

O_true = np.matrix([[0.9, 0.0],
                    [0.0, 0.2]])

O_false = np.matrix([[0.1, 0.0],
                     [0.0, 0.8]])


# Test of assignments
def partb_1():
    ev = [True, True]
    fw = [np.matrix([0.0, 0.0]) for i in range((len(ev) + 1))]
    fw[0] = np.matrix([[0.5], [0.5]])
    for i in range(1, len(ev) + 1):
        fw[i] = forward(T, O_true, fw[i - 1])
        print(fw[i])


def partb_2():
    ev = [True, True, False, True, True]
    fw = [np.matrix([0.0, 0.0]) for _ in range((len(ev) + 1))]
    fw[0] = np.matrix([[0.5], [0.5]])
    for i in range(1, len(ev) + 1):
        fw[i] = forward(T, (O_false, O_true)[ev[i - 1]], fw[i - 1])
        print(fw[i])


def partc_1():
    ev = [True, True]
    prior = np.matrix([[0.5], [0.5]])
    x = forward_backward(ev, prior)
    print("Backward messages:")
    x = forward_backward(ev, prior)
    print("Result:")
    for i in x:
        print(i)


def partc_2():
    ev = [True, True, False, True, True]
    prior = np.matrix([[0.5], [0.5]])
    print("Backward messages:")
    x = forward_backward(ev, prior)
    print("Result:")
    for i in x:
        print(i)


# Uncomment for testing of each task
# partb_1()
# partb_2()
# partc_1()
# partc_2()
