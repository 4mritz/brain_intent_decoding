from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train_csp_lda(X, y, test_size=0.3, random_state=42):
    """Train CSP+LDA and return model + metrics + splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    csp = CSP(n_components=6, reg=None, log=True)
    lda = LinearDiscriminantAnalysis()

    clf = Pipeline([("csp", csp), ("lda", lda)])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, acc, (X_train, X_test, y_train, y_test, y_pred)
