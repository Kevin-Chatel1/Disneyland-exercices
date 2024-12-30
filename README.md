
# 📄 **README.md** – *Disneyland Review Classifier avec MLflow & Docker*

## 📝 **Description du Projet**

Ce projet utilise **TensorFlow** et **MLflow** pour entraîner un modèle de classification des avis Disneyland. Grâce à **Docker**, tout est automatisé : l'environnement est standardisé et prêt à l'emploi.

## 🚀 **Prérequis**

- **Docker** installé : [Télécharger Docker](https://www.docker.com/get-started)  
- **Git** installé : [Télécharger Git](https://git-scm.com/downloads)  

## 📂 **Structure du Projet**

```
.
├── Dockerfile          # Configuration Docker
├── entrypoint.sh       # Script d'entrée automatisé
├── requirements.txt    # Dépendances Python
├── train.py            # Script d'entraînement du modèle
├── MLproject           # Configuration MLflow
└── README.md           # Guide rapide
```

## 🔑 **Configuration Initiale**

### 1️⃣ **Créer un fichier `.env`**

Ajoutez un fichier `.env` à la racine du projet avec les variables suivantes :

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=Disneyland_review_detector
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

> ⚠️ **Ce fichier ne doit pas être ajouté au dépôt Git.**

## 🛠️ **Démarrage Rapide**

### 1. **Cloner le projet :**

```bash
git clone https://github.com/your-username/disneyland_review_analyser.git
cd disneyland_review_analyser
```

### 2. **Construire l'image Docker :**

```bash
docker build -t disneyland-mlflow .
```

### 3. **Lancer le conteneur Docker :**

```bash
docker run -p 5000:5000 --env-file .env -e EPOCHS=5 -e INITIAL_LR=0.001 disneyland-mlflow
```

## 🌐 **Accès à l'Interface MLflow**

Ouvrez votre navigateur et accédez à l'interface MLflow :

👉 **[http://localhost:5000](http://localhost:5000)**

✨ **Voilà, tout est automatisé. Lancez, entraînez, suivez et analysez vos modèles directement depuis Docker et MLflow. 🚀**
