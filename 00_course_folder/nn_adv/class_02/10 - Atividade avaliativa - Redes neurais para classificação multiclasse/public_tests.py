import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, sigmoid, relu


def test_my_softmax(target):
    z = np.array([1.0, 2.0, 3.0, 4.0])
    a = target(z)
    atf = tf.nn.softmax(z)

    assert np.allclose(
        a, atf, atol=1e-10
    ), f"Valores incorretos. Esperado {atf}, obtive {a}"

    z = np.array([np.log(0.1)] * 10)
    a = target(z)
    atf = tf.nn.softmax(z)

    assert np.allclose(
        a, atf, atol=1e-10
    ), f"Valores errados. Esperado {atf}, obtive {a}"

    print("\033[92mPassou em todos os testes.")


def test_model(target, classes, input_size):
    target.build(input_shape=(None, input_size))

    assert (
        len(target.layers) == 3
    ), f"Número errado de camadas. Esperava 3, mas recebi {len(target.layers)}"
    assert target.input.shape.as_list() == [
        None,
        input_size,
    ], f"Forma de entrada incorreta. Esperado [None,  {input_size}] mas obtive {target.input.shape.as_list()}"
    i = 0
    expected = [
        [Dense, [None, 25], relu],
        [Dense, [None, 15], relu],
        [Dense, [None, classes], linear],
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

    print("\033[92mPassou em todos os testes!")
