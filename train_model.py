import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

import joblib


# ============================
# Utils
# ============================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # sécurité espaces
    return df


def detect_target(df: pd.DataFrame) -> str:
    target_candidates = ["isFraud", "is_fraud", "fraud", "Class", "target", "label"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"Aucune colonne cible trouvée. Colonnes dispo: {list(df.columns)}")
    return target_col


def eval_binary_classifier(name: str, model, X_test, y_test):
    """Affiche confusion matrix + classification report + ROC-AUC si proba dispo."""
    y_pred = model.predict(X_test)

    print(f"\n================ {name} ================")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC test: {auc:.4f}")


def main():
    # ============================
    # 1) Chargement
    # ============================
    df = load_data("train_resampled.csv")
    print("Shape:", df.shape)
    print("Colonnes:", list(df.columns))

    # ============================
    # 2) Cible
    # ============================
    target_col = detect_target(df)
    print("Colonne cible:", target_col)
    print(df[target_col].value_counts())

    # ============================
    # 3) X / y
    # ============================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ============================
    # 4) Split
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # ============================
    # 5) Modèle baseline (LogReg)
    # ============================
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )
    lr.fit(X_train, y_train)
    eval_binary_classifier("LOGISTIC REGRESSION (baseline)", lr, X_test, y_test)

    # ============================
    # 6) GridSearch sur HistGradientBoosting
    # ============================
    hgb = HistGradientBoostingClassifier(random_state=42)

    # Grille volontairement raisonnable (12 configs)
    param_grid = {
        "max_depth": [3, 5, None],
        "learning_rate": [0.05, 0.1],
        "max_iter": [100, 200],
    }

    grid = GridSearchCV(
        estimator=hgb,
        param_grid=param_grid,
        scoring="roc_auc",   # meilleur critère que accuracy pour fraude
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\n=========== GRIDSEARCH RESULTS (HGB) ===========")
    print("Meilleurs paramètres:", grid.best_params_)
    print(f"Meilleur ROC-AUC (CV): {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    # ============================
    # 7) Évaluation du meilleur modèle sur test
    # ============================
    eval_binary_classifier("BEST HIST GRADIENT BOOSTING (GridSearch)", best_model, X_test, y_test)

    # ============================
    # 8) Exemple de prédiction (1 ligne du test)
    # ============================
    sample = X_test.iloc[[0]]
    pred = best_model.predict(sample)[0]
    proba = best_model.predict_proba(sample)[0, 1]
    print("\nExemple prédiction (1 transaction du test):")
    print("Pred:", pred, "| Proba fraude:", round(float(proba), 4))

    # ============================
    # 9) Sauvegarde du meilleur modèle
    # ============================
    joblib.dump(best_model, "best_model_hgb.joblib")
    print("\n✅ Modèle sauvegardé: best_model_hgb.joblib")


if __name__ == "__main__":
    main()
