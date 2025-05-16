import pandas as pd
import pickle
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from google.cloud import storage
import os
from io import BytesIO

# Configuracion de Bucket y rutas
BUCKET_NAME = "comentarios-app"
CSV_BLOB = "dataset/comentarios.csv"
MODELO_BLOB = "modelo/modelo_sentimiento.pkl"
VECTORIZADOR_BLOB = "modelo/vectorizador_tfidf.pkl"

# Inicializacion del cliente Google Cloud Storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credenciales.json"
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def descargar_blob(blob_name):
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

def subir_blob(blob_name, data_bytes):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data_bytes)

def entrenar_modelo():
    csv_bytes = descargar_blob(CSV_BLOB)
    df = pd.read_csv(BytesIO(csv_bytes))
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["Comentario"])
    y = df["Sentimiento"]
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X, y)

    modelo_bytes = pickle.dumps(modelo)
    vectorizer_bytes = pickle.dumps(vectorizer)

    subir_blob(MODELO_BLOB, modelo_bytes)
    subir_blob(VECTORIZADOR_BLOB, vectorizer_bytes)

    return modelo, vectorizer

modelo, vectorizer = entrenar_modelo()

def procesar_entrada(profesor, comentario):
    global modelo, vectorizer

    comentario_vector = vectorizer.transform([comentario])
    sentimiento = modelo.predict(comentario_vector)[0]

    csv_bytes = descargar_blob(CSV_BLOB)
    df = pd.read_csv(BytesIO(csv_bytes))
    nuevo = pd.DataFrame([{"Comentario": comentario, "Sentimiento": sentimiento, "Profesor": profesor}])
    df = pd.concat([df, nuevo], ignore_index=True)

    output_buffer = BytesIO()
    df.to_csv(output_buffer, index=False)
    subir_blob(CSV_BLOB, output_buffer.getvalue())

    modelo, vectorizer = entrenar_modelo()

    df_profesor = df[df["Profesor"] == profesor]
    resumen = df_profesor["Sentimiento"].value_counts().to_dict()
    total = sum(resumen.values())
    resumen_final = {
        "Profesor": profesor,
        "Total Comentarios": total,
        "Positivos": resumen.get("positivo", 0),
        "Neutrales": resumen.get("neutral", 0),
        "Negativos": resumen.get("negativo", 0),
        "% Positivos": round(resumen.get("positivo", 0) / total * 100, 2) if total else 0,
        "% Neutrales": round(resumen.get("neutral", 0) / total * 100, 2) if total else 0,
        "% Negativos": round(resumen.get("negativo", 0) / total * 100, 2) if total else 0
    }

    return (
        f"Sentimiento detectado: {sentimiento}",
        pd.DataFrame([resumen_final]),
        df_profesor[["Comentario", "Sentimiento"]].reset_index(drop=True)
    )

app = gr.Interface(
    fn=procesar_entrada,
    inputs=[
        gr.Dropdown(choices=["David", "Ulises", "Aldo", "Omar", "Miguel", "Julián"], label="Profesor"),
        gr.Textbox(lines=4, label="Comentario del Alumno")
    ],
    outputs=[
        gr.Textbox(label="Resultado del Análisis"),
        gr.Dataframe(label="Resumen del Profesor"),
        gr.Dataframe(label="Comentarios del Profesor")
    ],
    title="Sistema Inteligente de Retroalimentación Docente",
    description="Clasifica comentarios por sentimiento, guarda el registro y muestra métricas + historial de comentarios por profesor."
)

app.launch()