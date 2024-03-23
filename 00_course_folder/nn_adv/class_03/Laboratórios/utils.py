import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.style.use("./deeplearning.mplstyle")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def plot_dataset(x, y, title):
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["lines.markersize"] = 12
    plt.scatter(x, y, marker="x", c="r")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title):
    plt.scatter(x_train, y_train, marker="x", c="r", label="treino")
    plt.scatter(x_cv, y_cv, marker="o", c="b", label="validação cruzada")
    plt.scatter(x_test, y_test, marker="^", c="g", label="teste")
    plt.title("entrada vs. alvo")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def plot_train_cv_mses(degrees, train_mses, cv_mses, title):
    degrees = range(1, 11)
    plt.plot(degrees, train_mses, marker="o", c="r", label="MSEs de treino")
    plt.plot(degrees, cv_mses, marker="o", c="b", label="MSEs de VC")
    plt.title(title)
    plt.xlabel("grau")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def plot_bc_dataset(x, y, title):
    for i in range(len(y)):
        marker = "x" if y[i] == 1 else "o"
        c = "r" if y[i] == 1 else "b"
        plt.scatter(x[i, 0], x[i, 1], marker=marker, c=c)
    plt.title("x1 vs x2")
    plt.xlabel("x1")
    plt.ylabel("x2")
    y_0 = mlines.Line2D(
        [], [], color="r", marker="x", markersize=12, linestyle="None", label="y=1"
    )
    y_1 = mlines.Line2D(
        [], [], color="b", marker="o", markersize=12, linestyle="None", label="y=0"
    )
    plt.title(title)
    plt.legend(handles=[y_0, y_1])
    plt.show()


def build_models():

    tf.random.set_seed(20)

    model_1 = Sequential(
        [
            Dense(25, activation="relu"),
            Dense(15, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name="model_1",
    )

    model_2 = Sequential(
        [
            Dense(20, activation="relu"),
            Dense(12, activation="relu"),
            Dense(12, activation="relu"),
            Dense(20, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name="model_2",
    )

    model_3 = Sequential(
        [
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(4, activation="relu"),
            Dense(12, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name="model_3",
    )

    model_list = [model_1, model_2, model_3]

    return model_list


# Não usado no laboratório. Você pode chamar isso para o problema de classificação binária
# binário se você definir `from_logits=False`
# ao declarar a perda. Com isso, você também não precisará
# chamar a função `tf.math.sigmoid()` no loop
# porque a saída do modelo já é uma probabilidade
def build_bc_models():

    tf.random.set_seed(20)

    model_1_bc = Sequential(
        [
            Dense(25, activation="relu"),
            Dense(15, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="model_1_bc",
    )

    model_2_bc = Sequential(
        [
            Dense(20, activation="relu"),
            Dense(12, activation="relu"),
            Dense(12, activation="relu"),
            Dense(20, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="model_2_bc",
    )

    model_3_bc = Sequential(
        [
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(4, activation="relu"),
            Dense(12, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="model_3_bc",
    )

    models_bc = [model_1_bc, model_2_bc, model_3_bc]

    return models_bc


def prepare_dataset(filename):

    data = np.loadtxt(filename, delimiter=",")

    x = data[:, :-1]
    y = data[:, -1]

    # Obtenha 60% do conjunto de dados como o conjunto de treinamento.
    # Coloque os 40% restantes em variáveis temporárias.
    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=80)

    # Dividir o subconjunto de 40% acima em dois: uma metade para validação cruzada e a outra para o conjunto de teste
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_, y_, test_size=0.50, random_state=80
    )

    return x_train, y_train, x_cv, y_cv, x_test, y_test


def train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=None):

    train_mses = []
    cv_mses = []
    models = []
    scalers = []
    degrees = range(1, max_degree + 1)

    # Faça um loop de 10 vezes. Cada vez adicionando mais um grau de polinômio maior que o anterior.
    for degree in degrees:

        # Adicionar recursos polinomiais ao conjunto de treinamento
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Dimensionar o conjunto de treinamento
        model.fit(X_train_mapped_scaled, y_train)
        models.append(model)

        # Calcular o MSE de treinamento
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Adicionar recursos polinomiais e dimensionar o conjunto de validação cruzada
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Calcular o MSE de validação cruzada
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

    # Plotar s resultados
    plt.plot(degrees, train_mses, marker="o", c="r", label="training MSEs")
    plt.plot(degrees, cv_mses, marker="o", c="b", label="CV MSEs")
    plt.plot(
        degrees, np.repeat(baseline, len(degrees)), linestyle="--", label="baseline"
    )
    plt.title("grau do polinômio vs. MSEs de treino e VC")
    plt.xticks(degrees)
    plt.xlabel("grau")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def train_plot_reg_params(
    reg_params, x_train, y_train, x_cv, y_cv, degree=1, baseline=None
):

    train_mses = []
    cv_mses = []
    models = []
    scalers = []

    # Faça um loop de 10 vezes. Cada vez adicionando mais um grau de polinômio maior que o anterior.
    for reg_param in reg_params:

        # Adicionar recursos polinomiais ao conjunto de treinamento
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Escalar o conjunto de treinamento
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Criar e treinar o modelo
        model = Ridge(alpha=reg_param)
        model.fit(X_train_mapped_scaled, y_train)
        models.append(model)

        # Calcular o MSE de treinamento
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Adicionar recursos polinomiais e dimensionar o conjunto de validação cruzada
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Calcular o MSE de validação cruzada
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

    # Plotar os resultados
    reg_params = [str(x) for x in reg_params]
    plt.plot(reg_params, train_mses, marker="o", c="r", label="MSEs de treino")
    plt.plot(reg_params, cv_mses, marker="o", c="b", label="MSEsde VC")
    plt.plot(
        reg_params,
        np.repeat(baseline, len(reg_params)),
        linestyle="--",
        label="baseline",
    )
    plt.title("lambda vs. MSEs de treino e VC")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def train_plot_diff_datasets(model, files, max_degree=10, baseline=None):

    for file in files:

        x_train, y_train, x_cv, y_cv, x_test, y_test = prepare_dataset(file["filename"])

        train_mses = []
        cv_mses = []
        models = []
        scalers = []
        degrees = range(1, max_degree + 1)

        # Faça um loop de 10 vezes. Cada vez adicionando mais um grau de polinômio maior que o anterior.
        for degree in degrees:

            # Adicionar recursos polinomiais ao conjunto de treinamento
            poly = PolynomialFeatures(degree, include_bias=False)
            X_train_mapped = poly.fit_transform(x_train)

            # Dimensionar o conjunto de treinamento
            scaler_poly = StandardScaler()
            X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
            scalers.append(scaler_poly)

            # Criar e treinar o modelo
            model.fit(X_train_mapped_scaled, y_train)
            models.append(model)

            # Calcular o MSE de treinamento
            yhat = model.predict(X_train_mapped_scaled)
            train_mse = mean_squared_error(y_train, yhat) / 2
            train_mses.append(train_mse)

            # Adicionar recursos polinomiais e dimensionar o conjunto de validação cruzada
            poly = PolynomialFeatures(degree, include_bias=False)
            X_cv_mapped = poly.fit_transform(x_cv)
            X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

            # Calcular o MSE de validação cruzada
            yhat = model.predict(X_cv_mapped_scaled)
            cv_mse = mean_squared_error(y_cv, yhat) / 2
            cv_mses.append(cv_mse)

        # Plotar os resultados
        plt.plot(
            degrees,
            train_mses,
            marker="o",
            c="r",
            linestyle=file["linestyle"],
            label=f"{file['label']} - MSEs de treino",
        )
        plt.plot(
            degrees,
            cv_mses,
            marker="o",
            c="b",
            linestyle=file["linestyle"],
            label=f"{file['label']} - MSEs de VC",
        )

    plt.plot(
        degrees, np.repeat(baseline, len(degrees)), linestyle="--", label="baseline"
    )
    plt.title("grau do polinômio vs. MSEs de treino e VC")
    plt.xticks(degrees)
    plt.xlabel("graus")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def train_plot_learning_curve(
    model, x_train, y_train, x_cv, y_cv, degree=1, baseline=None
):

    train_mses = []
    cv_mses = []
    models = []
    scalers = []
    num_samples_train_and_cv = []
    percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Faça um loop de 10 vezes. Cada vez adicionando mais um grau de polinômio maior que o anterior.
    for percent in percents:

        num_samples_train = round(len(x_train) * (percent / 100.0))
        num_samples_cv = round(len(x_cv) * (percent / 100.0))
        num_samples_train_and_cv.append(num_samples_train + num_samples_cv)

        x_train_sub = x_train[:num_samples_train]
        y_train_sub = y_train[:num_samples_train]
        x_cv_sub = x_cv[:num_samples_cv]
        y_cv_sub = y_cv[:num_samples_cv]

        # Adicionar recursos polinomiais ao conjunto de treinamento
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train_sub)

        # Dimensionar o conjunto de treinamento
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Criar e treinar o modelo
        model.fit(X_train_mapped_scaled, y_train_sub)
        models.append(model)

        # Calcular o MSE de treinamento
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train_sub, yhat) / 2
        train_mses.append(train_mse)

        # Adicionar recursos polinomiais e dimensionar o conjunto de validação cruzada
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv_sub)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Calcular o MSE de validação cruzada
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv_sub, yhat) / 2
        cv_mses.append(cv_mse)

    # Plotar os resultados
    plt.plot(
        num_samples_train_and_cv, train_mses, marker="o", c="r", label="training MSEs"
    )
    plt.plot(num_samples_train_and_cv, cv_mses, marker="o", c="b", label="CV MSEs")
    plt.plot(
        num_samples_train_and_cv,
        np.repeat(baseline, len(percents)),
        linestyle="--",
        label="baseline",
    )
    plt.title("quantidade de exemplos vs. MSEs de treino e VC")
    plt.xlabel("número total de exemplos de treinamento e cv")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
