import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_validate

df_training = pd.read_csv('D:\Arquivos do Usuário\Desktop\VictimSim2-main\datasets\data_4000v\env_vital_signals.txt', header=None, names=['id', 'x', 'y', 'qPA', 'pulso', 'fResp', 'grav', 'label'])
df_testing = pd.read_csv('D:\Arquivos do Usuário\Desktop\VictimSim2-main\datasets\data_800v\env_vital_signals.txt', header=None, names=['id', 'x', 'y', 'qPA', 'pulso', 'fResp', 'grav', 'label'])

def train_test_regressor(type):
    x_train = df_training[['qPA', 'pulso', 'fResp']]
    y_train = df_training['grav'].values
    
    x_test = df_testing[['qPA', 'pulso', 'fResp']]
    y_test = df_testing['grav'].values

    print(f"Tamanho do dataset de validacao: {len(df_training):.0f}")
    print(f"Tamanho do dataset de teste: {len(df_testing):.0f}")

    train_scores = []
    vld_scores = []
    best_index = []
    best_model = []
    model = []

    max_depth=[5, 10, 20]
    min_samples_leaf=[0.25, 0.1, 0.05]
    max_iters = [1000, 2000, 3000]
    hidden_layers = [5, 10, 15]

    for i in range(3):
            if type == "CART":
                alg = DecisionTreeRegressor(max_depth=max_depth[i], min_samples_leaf=min_samples_leaf[i], random_state=420)
            elif type == 'MLP':
                alg = MLPRegressor(hidden_layer_sizes=(hidden_layers[i]), max_iter=max_iters[i], random_state=42, early_stopping=True)

            cv_results = cross_validate(
            alg,
            x_train,
            y_train,
            cv=5,
            scoring='neg_mean_squared_error', 
            return_train_score=True, 
            return_estimator=True    
            )

            train_scores.append(cv_results['train_score'])
            vld_scores.append(cv_results['test_score'])

            bias = np.abs(cv_results['train_score'] - cv_results['test_score'])
            best_index.append(np.argmin(bias))
            best_model.append(cv_results['estimator'][best_index[i]])
            model.append(cv_results['estimator'])

            train_rmse = np.mean(np.sqrt(-cv_results['train_score']))
            vld_rmse = np.mean(np.sqrt(-cv_results['test_score']))
            
            print("---------------------------------------------------------")

            if type == "CART":
                print(f"Configuração {i+1}: max_depth={max_depth[i]}, min_samples_leaf={min_samples_leaf[i]}\n")
            elif type == "MLP":
                print(f"Configuração {i+1}: max_iter={max_iters[i]}, hidden_layer_sizes={hidden_layers[i]}\n")
            print(f"RMSE médio (treino):     {train_rmse:.4f}\n")
            print(f"RMSE médio (validação): {vld_rmse:.4f}\n")

            y_predict_train = best_model[i].predict(x_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_predict_train))
            print(f"RMSE treino: {rmse_train:.4f}")

            y_predict_test = best_model[i].predict(x_test)
            rmse_test = np.sqrt(mean_squared_error(y_test,y_predict_test))
            print(f"RMSE teste: {rmse_test:.4f}")

    val_rmse = [np.mean(np.sqrt(-score)) for score in vld_scores]
    best_idx = np.argmin(val_rmse)
    return best_model[best_idx]

def train_test_classifier(type):
    x_train = df_training[['qPA', 'pulso', 'fResp']]
    y_train = df_training['label'].values
    
    x_test = df_testing[['qPA', 'pulso', 'fResp']]
    y_test = df_testing['label'].values

    print(f"Tamanho do dataset de validacao: {len(df_training):.0f}")
    print(f"Tamanho do dataset de teste: {len(df_testing):.0f}")

    train_scores = []
    vld_scores = []
    best_index = []
    best_model = []
    model = []

    max_depth=[5, 10, 20]
    min_samples_leaf=[0.25, 0.1, 0.05]
    max_iters = [1000, 2000, 3000]
    hidden_layers = [5, 10, 15]

    for i in range(3):
            if type == "CART":
                alg = DecisionTreeClassifier(max_depth=max_depth[i], min_samples_leaf=min_samples_leaf[i], random_state=420)
            elif type == 'MLP':
                alg = MLPClassifier(hidden_layer_sizes=(hidden_layers[i]), max_iter=max_iters[i], random_state=42, early_stopping=True)
            cv_results = cross_validate(
            alg,
            x_train,
            y_train,
            cv=5,
            scoring='accuracy', 
            return_train_score=True, 
            return_estimator=True    
            )

            train_scores.append(cv_results['train_score'])
            vld_scores.append(cv_results['test_score'])

            bias = np.abs(cv_results['train_score'] - cv_results['test_score'])
            best_index.append(np.argmin(bias))
            best_model.append(cv_results['estimator'][best_index[i]])
            model.append(cv_results['estimator'])

            train_acc = np.mean(cv_results['train_score'])
            val_acc = np.mean(cv_results['test_score'])

            print("---------------------------------------------------------")
            if type == "CART":
                print(f"Configuração {i+1}: max_depth={max_depth[i]}, min_samples_leaf={min_samples_leaf[i]}")
            elif type == "MLP":
                print(f"Configuração {i+1}: max_iter={max_iters[i]}, hidden_layer_sizes={hidden_layers[i]}")

            print(f"Acurácia média (treino):    {train_acc:.4f}")
            print(f"Acurácia média (validação): {val_acc:.4f}")

            y_predict_train = best_model[i].predict(x_train)

            print(classification_report(y_train, y_predict_train, digits=4, zero_division=0))

            acc_train = accuracy_score(y_train, y_predict_train)
            print(f"Acurácia treino:            {acc_train:.4f}")

            y_predict_test = best_model[i].predict(x_test)

            acc_test = accuracy_score(y_test, y_predict_test)
            print(f"Acurácia teste:             {acc_test:.4f}")

    val_accs = [np.mean(score) for score in vld_scores]
    best_idx = np.argmax(val_accs)
    return best_model[best_idx]
    
if __name__ == '__main__':
    cart_regressor = train_test_regressor("CART")
    mlp_regressor = train_test_regressor("MLP")

    cart_classifier = train_test_classifier("CART")
    mlp_classifier = train_test_classifier("MLP")

    