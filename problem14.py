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
activation_derivative   = lambda x: ((math.e ** (-x)) / ((1 + math.e**(-x)) ** 2))

g                       = activation
gprime                  = activation_derivative

H4 = lambda w14, x1, w24, x2, w34, x3: g(w14 * x1 + w24 * x2 + w34 * x3)
H5 = lambda w15, x1, w25, x2, w35, x3: g(w15 * x1 + w25 * x2 + w35 * x3)
O6 = lambda w46, h4, w56, h5, w36, x3: g(w46 * h4 + w56 * h5 + w36 * x3)

#new_weight_outer = lambda old, lr, ia, e: old + lr*ia*e
#new_weight_inner = lambda diriv, z, weight, error: diriv(z) * weight * error

new_weight = lambda old_weight, learn_rate, outer_node, inner_node, error:  \
    old_weight + learn_rate * error * (outer_node * (1 - outer_node)) * inner_node

##############################################################################
#   Formatting / Output functions
##############################################################################
def print_ho(h4, h5, o6, e):
    print("H4:\n  {} = {}".format(h4_str, h4))
    print("H5:\n  {} = {}".format(h5_str, h5))
    print("O6:\n  {} = {}".format(o6_str, o6))
    print("Error: {}\n".format(e))

def print_new_weight(new_val, name, old_val, lr, node_val, downstream_node, e,
                     outer_node_name, inner_node_name):
    print("{0:<3} = OldWeight - LEARN_RATE * error * ({1} * (1 - {1}) * {2}".format(
        name, outer_node_name, inner_node_name))
    print("    = {0} - {1} * {2} * ({3} * (1 - {3})) * {4}".format(
        old_val, lr, e, node_val, downstream_node))
    print("    = {}".format(new_val))



##############################################################################
#   Main stuff
##############################################################################
rounds = (
    ((1, 1, 1), 1),
    ((0, 0, 1), 0),
)

def do_round(r, weights, verbose=True):
    w14, w24, w34, w15, w25, w35, w46, w56, w36 = weights

    x1, x2, x3 = r[0]
    res = r[1]

    h4 = H4(w14, x1, w24, x2, w34, x3)
    h5 = H5(w15, x1, w25, x2, w35, x3)
    o6 = O6(w46, h4, w56, h5, w36, x3)

    e = res - o6


    print("O6: {}".format(o6))
    print("Error: {}".format(e))


    new_w46 = new_weight(w46, LEARNING_RATE, o6, h4, e)
    new_w56 = new_weight(w56, LEARNING_RATE, o6, h5, e)
    new_w36 = new_weight(w36, LEARNING_RATE, o6, x3, e)


    w14_error = e * w14
    w24_error = e * w24
    w34_error = e * w34

    new_w14 = new_weight(w14, LEARNING_RATE, h4, x1, w14_error)
    new_w24 = new_weight(w24, LEARNING_RATE, h4, x2, w24_error)
    new_w34 = new_weight(w34, LEARNING_RATE, h4, x3, w34_error)

    w15_error = e * w15
    w25_error = e * w25
    w35_error = e * w35

    new_w15 = new_weight(w15, LEARNING_RATE, h5, x1, w15_error)
    new_w25 = new_weight(w25, LEARNING_RATE, h5, x2, w25_error)
    new_w35 = new_weight(w35, LEARNING_RATE, h5, x3, w35_error)

    if verbose:
        print_ho(h4, h5, o6, e)

        print_new_weight(new_w46, "w46", w46, LEARNING_RATE, o6, h4, e, 'O6', 'H4')
        print_new_weight(new_w56, "w56", w56, LEARNING_RATE, o6, h5, e, 'O6', 'H5')
        print_new_weight(new_w36, "w36", w36, LEARNING_RATE, o6, x3, e, 'O6', 'X3')
        print()

        print_new_weight(new_w14, 'w14', w14, LEARNING_RATE, h4, x1, w14_error, 'H4', 'X1')
        print_new_weight(new_w24, 'w24', w24, LEARNING_RATE, h4, x2, w24_error, 'H4', 'X2')
        print_new_weight(new_w34, 'w34', w34, LEARNING_RATE, h4, x3, w34_error, 'H4', 'X3')
        print()

        print_new_weight(new_w15, 'w15', w15, LEARNING_RATE, h5, x1, w15_error, 'H5', 'X1')
        print_new_weight(new_w25, 'w25', w25, LEARNING_RATE, h5, x2, w25_error, 'H5', 'X2')
        print_new_weight(new_w35, 'w35', w35, LEARNING_RATE, h5, x3, w35_error, 'H5', 'X3')
        print()

    return (new_w14, new_w15, new_w24, new_w24, new_w34, new_w35, new_w36, new_w46, new_w56)

org_weights = (w14, w24, w34, w15, w25, w35, w46, w56, w36)

def main():
    weights = org_weights
    for ii, r in enumerate(rounds):
        print("ROUND {}".format(ii+ 1))
        print("    {}".format(r))
        weights = do_round(r, weights, verbose=False)

        print("weights:\n  w14 = {}, w24 = {}, w34 = {}, w15 = {}, w25 = {}, w35 = {},"
              " w46 = {}, w56 = {}, w36 = {}\n".format(*weights))

if __name__ == '__main__':
    main()
