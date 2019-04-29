import math

##############################################################################
#   Constants
##############################################################################
w14 = 0.2
w24 = 0.2
w34 = 0.2

w15 = 0.2
w25 = 0.2
w35 = 0.2

w46 = 0.5
w56 = 0.5
w36 = 0.5

LEARNING_RATE = 0.5

h4_str = "g(w14 * x1 + w24 * x2 + w34 * x3)"
h5_str = "g(w15 * x1 + w25 * x2 + w35 * x3)"
o6_str = "g(w46 * h4 + w56 * h5 + w36 * x3)"


##############################################################################
#   Lamda Functions for actication / back propogation
##############################################################################
activation              = lambda x: 1 / (1 + math.e**(-x))
activation_derivative   = lambda x: ((math.e ** (-x)) / (1 + math.e**(-x))) ** 2

g                       = activation
gprime                  = activation_derivative

H4 = lambda w14, x1, w24, x2, w34, x3: g(w14 * x1 + w24 * x2 + w34 * x3)
H5 = lambda w15, x1, w25, x2, w35, x3: g(w15 * x1 + w25 * x2 + w35 * x3)
O6 = lambda w46, h4, w56, h5, w36, x3: g(w46 * h4 + w56 * h5 + w36 * x3)

new_weight_outer = lambda old, lr, ia, e: old + lr*ia*e
new_weight_inner = lambda diriv, z, weight, error: diriv(z) * weight * error

##############################################################################
#   Formatting / Output functions
##############################################################################
def print_ho(h4, h5, o6, e):
    print("H4:\n  {} = {}".format(h4_str, h4))
    print("H5:\n  {} = {}".format(h5_str, h5))
    print("O6:\n  {} = {}".format(o6_str, o6))
    print("Error: {}\n".format(e))

def print_new_outer_weight(new_val, name, old_val, lr, ia, e):
    print("{}\n  {} = {} + {} * {} * {}".format(
        name, new_val, old_val, lr, ia, e))

def print_new_inner_weight(new_val, name, old_val, z, e):
    print("{}\n  {} = g'({}) * {} * {}".format(
        name, new_val, z, old_val, e))

##############################################################################
#   Main stuff
##############################################################################
rounds = (
    ((1, 1, 1), 1),
    ((0, 0, 1), 0)
)

def do_round(r, round_num=1):
    print("Round number {}".format(round_num))

    x1, x2, x3 = r[0]
    res = r[1]

    h4 = H4(w14, x1, w24, x2, w34, x3)
    h5 = H5(w15, x1, w25, x2, w35, x3)
    o6 = O6(w46, h4, w56, h5, w36, x3)

    e = res - o6

    print_ho(h4, h5, o6, e)

    new_w46 = new_weight_outer(w46, LEARNING_RATE, o6, e)
    new_w56 = new_weight_outer(w56, LEARNING_RATE, o6, e)
    new_w36 = new_weight_outer(w36, LEARNING_RATE, o6, e)

    print_new_outer_weight(new_w46, "w46", w46, LEARNING_RATE, o6, e)
    print_new_outer_weight(new_w56, "w56", w56, LEARNING_RATE, o6, e)
    print_new_outer_weight(new_w36, "w36", w36, LEARNING_RATE, o6, e)

    print()

    new_w14 = new_weight_inner(gprime, h4, w14, e)
    print_new_inner_weight(new_w14, 'w14', w14, h4, e)
