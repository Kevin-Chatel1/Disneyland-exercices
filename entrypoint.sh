#!/bin/sh

# Étape 1: Migrer automatiquement la base de données
echo "🛠️ Migration de la base de données MLflow..."
mlflow db upgrade sqlite:///mlflow.db || {
    echo "❌ Échec de la migration de la base de données."
    exit 1
}

# Étape 2: Démarrer le serveur MLflow en arrière-plan
echo "🚀 Démarrage du serveur MLflow..."
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 \
              --port 5000 &

# Capturer le PID du serveur MLflow
MLFLOW_PID=$!

# Étape 3: Attendre que MLflow soit prêt
echo "⏳ En attente du serveur MLflow..."
sleep 5
until curl -s http://localhost:5000/api/2.0/mlflow/experiments/list; do
    echo "⏳ En attente de MLflow..."
    sleep 2
done

echo "✅ MLflow est démarré avec succès!"

# Étape 4: Lancer le script Python d'entraînement
echo "📊 Démarrage du script d'entraînement..."
python train.py --epochs "${EPOCHS:-5}" --initial_lr "${INITIAL_LR:-0.001}"

# Étape 5: Garder le conteneur actif même après l'entraînement
echo "🛡️ Le serveur MLflow reste actif. Vous pouvez accéder à l'interface à http://localhost:5000"
wait $MLFLOW_PID


