import numpy as np

class LogRegressionL2:
    def __init__(self, lr=0.01, lam=0.1, n_iter=1000):
        self.lr = lr
        self.lambda_ = lam
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def set_model(self, X, y, w = None):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        if w is None:
            w = np.ones(n_samples) # hace que sino se ponen pesos, w sea lo mismo que multiplicar por 1
        
        w = w / w.sum() * n_samples
        
        for i in range(self.n_iter):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            dw = (1 / n_samples) * (X.T @ (w *(y_pred - y))) + self.lambda_ * self.weights
            db = (1 / n_samples) * np.sum(w * (y_pred - y))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict_prob(self, X): # para las metricas que necesitan scores continuos (las curvas)
            return self.sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)


class LDA:
    def __init__(self):
        self.means = None
        self.priors = None
        self.cov_inv = None
        self.classes = None
    
    def set_model(self, X, y):
            self.classes = np.unique(y) #Guarda las clases 
            n_samples, n_features = X.shape
            n_classes = len(self.classes) #Cantidad de clases diferentes
            
            #inicializo los parametros
            self.means = {}
            self.priors = {}
            cov = np.zeros((n_features, n_features)) #La matriz de covarianza la comparten todas las clases

            for k in self.classes:
                X_k = X[y == k] #filtra los datos por clase para calcular la media y la proporcion de esa clase sobre los datos
                self.means[k] = X_k.mean(axis=0) 
                self.priors[k] = len(X_k) / n_samples
                
                X_cent = X_k - self.means[k] #Se centran los datos
                cov += X_cent.T @ X_cent #se le suma a la matriz de cov el prod int


            cov /= (n_samples - n_classes) #normaliza para que quede la covarianza compartida por todas las clases 
            #en el libro es tied covariances, dice que si las covarianzas son las mismas, entones las fronteras van a ser lineales
            cov += 1e-6 * np.eye(n_features) #estabilizacion y regularizacion si la matriz es regular
            self.cov_inv = np.linalg.inv(cov)

    def predict_prob(self, X):
        #Para cada clase calcula el score lineal por la formula de bayes y log

        scores = []
        for k in self.classes:
            mu_k = self.means[k]
            log_prior = np.log(self.priors[k])

            #Score lineal discriminante
            lin = X @ self.cov_inv @ mu_k
            quadr = 0.5 * mu_k @ self.cov_inv @ mu_k
            scores.append(lin - quadr + log_prior)

        scores = np.array(scores).T  #El clasificador de Bayes con la covarianza compartida resulta en una funcion lineal
        
        #Aplico softmax
        scores -= scores.max(axis=1, keepdims=True)  # estabilidad numérica
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    
    def predict(self, X):
        probs = self.predict_prob(X)
        i = np.argmax(probs, axis=1)
        return self.classes[i]


class LogRegressionMulticlass:
    def __init__(self, lr=0.01, lam=0.5, n_iter=1000):
        self.lr = lr
        self.lambda_ = lam
        self.n_iter = n_iter
        self.weights = None  
        self.bias = None     
        self.classes = None

    def softmax(self, z):

        z -= z.max(axis=1, keepdims=True)  # estabilidad numerica
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def set_model(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        #inicializo
        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

        # Hago one hot encoding de los rendimientos porque necesito que la clase este en el mismo formato que la salida del modelo
        Y = np.zeros((n_samples, n_classes))
        for i, k in enumerate(self.classes):
            Y[:, i] = (y == k).astype(int)

        for _ in range(self.n_iter):
            z = X @ self.weights.T + self.bias  
            y_pred = self.softmax(z)                  

            diff = y_pred - Y                         

            dw = (1 / n_samples) * (diff.T @ X) + self.lambda_ * self.weights
            db = (1 /n_samples) * diff.sum(axis=0)

            self.weights -= self.lr * dw 
            self.bias -= self.lr * db

    def predict_prob(self, X):
        z = X @ self.weights.T + self.bias
        return self.softmax(z)  

    def predict(self, X):
        probs = self.predict_prob(X)
        i = np.argmax(probs, axis=1)
        return self.classes[i]


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf #evita que hayan splits con muy pocos datos
        self.max_features = max_features
        self.tree = None
        self.classes = None

    def entropy(self, y): #mide cuan mezcladas estan las clases, el arbol busca minimizarla
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def best_split(self, X, y): #elige el mejor corte probando cada threshold para todas las features
        n_samples, n_features = X.shape
        best_gain = -1
        best_feat = None
        best_thresh = None

        # seleccion aleatoria de features
        n_feats  = self.max_features or n_features
        feat_idx = np.random.choice(n_features, size=n_feats, replace=False)

        parent_entropy = self.entropy(y)

        for feat in feat_idx:
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feat], np.linspace(10, 90, 10))
            for thresh in thresholds:
                left  = y[X[:, feat] <= thresh]
                right = y[X[:, feat] >  thresh]

                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                gain = parent_entropy \
                     - (len(left)  / n_samples) * self.entropy(left) \
                     - (len(right) / n_samples) * self.entropy(right)

                if gain > best_gain: #elige el par feature, threshold que maximiza la ganancia de info
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain #devuelve la mejor feature, el mejor threshold 

    def build_tree(self, X, y, depth=0): #construccion recursiva de arboles

        #condiciones de parada
        if (self.max_depth is not None and depth >= self.max_depth) \
        or len(np.unique(y)) == 1 \
        or len(y) < 2 * self.min_samples_leaf:
            classes, counts = np.unique(y, return_counts=True) #si para se crea una hoja
            probs = np.zeros(len(self.classes))
            for k, c in zip(classes, counts):
                idx = np.where(self.classes == k)[0][0]
                probs[idx] = c / len(y)
            return {"leaf": True, "probs": probs}

        #sino crea un nodo interno
        feat, thresh, gain = self.best_split(X, y)

        if feat is None:
            classes, counts = np.unique(y, return_counts=True)
            probs = np.zeros(len(self.classes))
            for k, c in zip(classes, counts):
                idx = np.where(self.classes == k)[0][0]
                probs[idx] = c / len(y)
            return {"leaf": True, "probs": probs}

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        return {
            "leaf"     : False,
            "feat"     : feat,
            "thresh"   : thresh,
            "gain"     : gain,
            "n_samples": len(y), 
            "left"     : self.build_tree(X[left_mask],  y[left_mask],  depth+1),
            "right"    : self.build_tree(X[right_mask], y[right_mask], depth+1)
        }

    def set_model(self, X, y):
        self.classes = np.unique(y)
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, node):
        if node["leaf"]:
            return node["probs"]
        if x[node["feat"]] <= node["thresh"]:
            return self.predict_one(x, node["left"])
        return self.predict_one(x, node["right"])

    def predict_prob(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

    def predict(self, X):
        probs = self.predict_prob(X)
        indices = np.argmax(probs, axis=1)
        return self.classes[indices]


class RandomForest:
    def __init__(self, n_estimators=30, max_depth=12, min_samples_leaf=10, max_features=6, random_state = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state 
        self.trees = []
        self.classes = None

    def set_model(self, X, y):
        self.classes = np.unique(y)
        n_feats = self.max_features or int(np.sqrt(X.shape[1]))
        self.trees = []

        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            #Bootstrap: cada arbol ve aporx 63% de los datos
            idx = rng.choice(len(y), size=len(y), replace=True) 
            X_b = X[idx]
            y_b = y[idx]

            tree = DecisionTree(
                max_depth = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                max_features = n_feats
            )
            tree.classes = self.classes  
            tree.tree = tree.build_tree(X_b, y_b)
            self.trees.append(tree)

    def predict_prob(self, X):

        all_probs = np.array([t.predict_prob(X) for t in self.trees])
        return all_probs.mean(axis=0) 

    def predict(self, X):
        probs = self.predict_prob(X)
        indices = np.argmax(probs, axis=1)
        return self.classes[indices]

    def feature_importance(self): #como en cada nodo se elige la feature que minimiza la entropia, si una feature fue elegida muchas veces, es una feature muy importante
        n_features  = len(set(self._get_features(self.trees[0].tree)))
        importances = np.zeros(n_features)

        for tree in self.trees:
            n_samples = tree.tree["n_samples"]  # total de muestras del árbol
            self._compute_importance(tree.tree, importances, n_samples) # la importancia de una feature es la suma de las importancia de todos los nodos en los que se eligio esa feature

        # los nodos son importantes segun el nivel en el que se encuentren, ya que cuanto mas arriba, tienen en consideracion mas datos
        importances /= len(self.trees) # se promedian la importancia de las features sobre todos los arboles
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances
    
    def _get_features(self, node):
        if node["leaf"]:
            return [0]
        return [node["feat"]] + self._get_features(node["left"]) \
                              + self._get_features(node["right"])

    def _compute_importance(self, node, importances, n_samples):
        if node["leaf"]:
            return
        importances[node["feat"]] += node["gain"] * (node["n_samples"] / n_samples)
        self._compute_importance(node["left"],  importances, n_samples)
        self._compute_importance(node["right"], importances, n_samples)

