import argparse
import mlflow
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from mlflow.models.signature import infer_signature

if __name__ == "__main__":
    ### MLFLOW Experiment setup
    experiment_name = "Disneyland_review_detector"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Parse arguments given in shell script
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs")
    parser.add_argument("--initial_lr")
    args = parser.parse_args()
    
    try:
        epochs = int(args.epochs)
        lr = float(args.initial_lr)
    except (ValueError, TypeError):
        raise ValueError("Les arguments --epochs et --initial_lr doivent être des valeurs numériques valides.")

    ### Assurer qu'aucune session active MLflow n'existe
    while mlflow.active_run():
        print("🛑 Une session MLflow est déjà active. Fermeture forcée...")
        mlflow.end_run()

    ### Create autolog 
    mlflow.tensorflow.autolog(log_models=False)
    mlflow.set_tag("version", "1.0")
    mlflow.set_tag("author", "Kevin")

    ### Import dataset of french reviews of Disneyland
    try:
        french_reviews = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/images/M08-DeepLearning/NLP/french_review_clean.csv")
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du dataset : {e}")

    # Shuffle your dataset 
    shuffle_df = french_reviews.sample(frac=1)

    # Define a size for your train set 
    train_size = int(0.7 * len(french_reviews))

    # Split your dataset 
    train_set = shuffle_df.iloc[:train_size].copy()
    test_set = shuffle_df.iloc[train_size:].copy()

    # Extract only reviews and target
    X_train = tf.convert_to_tensor(train_set["review_format"])
    y_train = tf.convert_to_tensor(train_set["stars"]-1)

    X_test = tf.convert_to_tensor(test_set["review_format"])
    y_test = tf.convert_to_tensor(test_set["stars"]-1)

    pre_trained_model = "https://tfhub.dev/google/nnlm-en-dim50/2"
    model = tf.keras.Sequential([
        hub.KerasLayer(pre_trained_model, input_shape=[], dtype=tf.string, trainable=True),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation="softmax")
    ])

    ### Configure learning rate
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=30,
        decay_rate=0.96,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    weights = 1/(french_reviews["stars"]-1).value_counts()
    weights = weights * len(french_reviews)/5
    weights = {index: values for index, values in zip(weights.index, weights.values)}

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Log experiment to MLFlow
    with mlflow.start_run() as run:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("initial_lr", lr)
        mlflow.log_param("dataset_url", "https://full-stack-assets.s3.eu-west-3.amazonaws.com/images/M08-DeepLearning/NLP/french_review_clean.csv")

        model.fit(X_train,
                y_train,
                epochs=epochs, 
                batch_size=64,
                validation_data=(X_test, y_test),
                class_weight=weights,
                callbacks=[early_stopping])

        predictions = model.predict(X_train)

        mlflow.keras.log_model(
            keras_model=model,
            artifact_path="Sentiment_detector",
            registered_model_name="Sentiment_detector_RNN",
            signature=infer_signature(french_reviews, predictions)
        )
