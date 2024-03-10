# Testes das Unidades
from tensorflow.keras.activations import sigmoid as tf_keras_sigmoid
from tensorflow.keras.layers import Dense

import numpy as np


def test_c1(target):
    assert (
        len(target.layers) == 3
    ), f"Número errado de camadas. Esperava 3, mas recebi {len(target.layers)}"
    assert target.input.shape.as_list() == [
        None,
        400,
    ], f"Forma de entrada incorreta. Esperava [None, 400], mas obtive {target.input.shape.as_list()}"
    i = 0
    expected = [
        [Dense, [None, 25], tf_keras_sigmoid],
        [Dense, [None, 15], tf_keras_sigmoid],
        [Dense, [None, 1], tf_keras_sigmoid],
    ]

    for layer in target.layers:
        assert (
            type(layer) == expected[i][0]
        ), f"Tipo errado na camada {i}. Esperado {expected[i][0]} mas obtive {type(layer)}"
        assert (
            layer.output.shape.as_list() == expected[i][1]
        ), f"Número incorreto de unidades na camada {i}. Esperado {expected[i][1]} mas obtive {layer.output.shape.as_list()}"
        assert (
            layer.activation == expected[i][2]
        ), f"Ativação incorreta na camada {i}. Esperado {expected[i][2]} mas obtive {layer.activation}"
        i = i + 1

    print("\033[92mTodos os testes foram aprovados!")


def test_c2(target):

    def linear(a):
        return a

    def linear_times3(a):
        return a * 3

    x_tst = np.array([1.0, 2.0, 3.0, 4.0])  # (1 examples, 3 features)
    W_tst = np.array(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    )  # (3 input features, 2 output features)
    b_tst = np.array([0.0, 0.0])  # (2 features)

    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape[0] == len(b_tst)
    assert np.allclose(
        A_tst, [10.0, 20.0]
    ), "Saída incorreta. Verifique o produto escalar"

    b_tst = np.array([3.0, 5.0])  # (2 features)

    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(
        A_tst, [13.0, 25.0]
    ), "Saída incorreta. Verifique o termo de polarização na fórmula"

    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(
        A_tst, [39.0, 75.0]
    ), "Saída incorreta. Você aplicou a função de ativação no final?"

    print("\033[92mTodos os testes foram aprovados!")


def test_c3(target):

    def linear(a):
        return a

    def linear_times3(a):
        return a * 3

    x_tst = np.array([1.0, 2.0, 3.0, 4.0])  # (1 examples, 3 features)
    W_tst = np.array(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    )  # (3 input features, 2 output features)
    b_tst = np.array([0.0, 0.0])  # (2 features)

    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape[0] == len(b_tst)
    assert np.allclose(
        A_tst, [10.0, 20.0]
    ), "Saída incorreta. Verifique o produto escalar"

    b_tst = np.array([3.0, 5.0])  # (2 features)

    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(
        A_tst, [13.0, 25.0]
    ), "Saída incorreta. Verifique o termo de polarização na fórmula"

    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(
        A_tst, [39.0, 75.0]
    ), "Saída incorreta. Você aplicou a função de ativação no final?"

    x_tst = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )  # (2 examples, 4 features)
    W_tst = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12]]
    )  # (3 input features, 2 output features)
    b_tst = np.array([0.0, 0.0, 0.0])  # (2 features)

    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape == (2, 3)
    assert np.allclose(
        A_tst, [[70.0, 80.0, 90.0], [158.0, 184.0, 210.0]]
    ), "Saída incorreta. Verifique o produto escalar"

    b_tst = np.array([3.0, 5.0, 6])  # (3 features)

    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(
        A_tst, [[73.0, 85.0, 96.0], [161.0, 189.0, 216.0]]
    ), "Saída incorreta. Verifique o termo de polarização na fórmula"

    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(
        A_tst, [[219.0, 255.0, 288.0], [483.0, 567.0, 648.0]]
    ), "Saída incorreta. Você aplicou a função de ativação no final?"

    print("\033[92mTodos os testes foram aprovados!")
