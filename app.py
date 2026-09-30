import streamlit as st
import subprocess
import sys
import tempfile
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="STOP Sign AI Evaluation",
    page_icon="🚦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
INFERENCE_FILE = BASE_DIR / "src" / "inference.py"


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #07111f;
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background-color: #07111f;
}

/* Titres */

h1, h2, h3 {
    color: #f8fafc !important;
}

h1 {
    font-size: 42px !important;
}

h2 {
    margin-top: 35px !important;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e3a5f;
}

h3 {
    color: #60a5fa !important;
}

/* Texte */

p {
    color: #cbd5e1;
    line-height: 1.7;
}

/* Conteneurs Streamlit */

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #0d1b2e;
    border-color: #1e3a5f !important;
    border-radius: 15px;
}

/* Upload */

[data-testid="stFileUploader"] {
    background-color: #0d1b2e;
    border-radius: 15px;
    padding: 10px;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #1e3a5f;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.title("🚦 STOP Sign Detection")

st.caption(
    "Computer Vision · Deep Learning · Image Classification"
)


# ============================================================
# 1. PRESENTATION DU PROJET
# ============================================================

st.header("01. Présentation du projet")

col1, col2 = st.columns(2)

with col1:

    with st.container(border=True):

        st.subheader("🎯 Objectif")

        st.write(
            "Ce projet consiste à développer un système de "
            "Computer Vision capable d'analyser une image et "
            "de déterminer automatiquement si elle contient "
            "un panneau STOP ou non."
        )

        st.write(
            "Le problème est traité comme une classification "
            "d'images binaire avec deux classes : "
            "**STOP** et **NOT STOP**."
        )


with col2:

    with st.container(border=True):

        st.subheader("🔎 Problématique")

        st.write(
            "Comment utiliser les techniques de Deep Learning "
            "et de Computer Vision pour permettre à un système "
            "informatique de reconnaître automatiquement un "
            "panneau STOP à partir d'une image ?"
        )

        st.write(
            "Le projet met en œuvre un modèle de classification "
            "capable de produire une prédiction accompagnée "
            "de son niveau de confiance."
        )


# ============================================================
# 2. TRAVAIL REALISE
# ============================================================

st.header("02. Travail réalisé")

with st.container(border=True):

    st.subheader("🛠️ Pipeline de développement")

    st.markdown(
        """
        **1. Préparation des données**

        Organisation des images dans les différentes catégories
        nécessaires à l'entraînement, à la validation et au test.

        **2. Prétraitement des images**

        Préparation des images avant leur passage dans le réseau
        de neurones afin de respecter le format attendu par le modèle.

        **3. Transfer Learning**

        Utilisation d'une architecture ResNet-18 pré-entraînée
        afin de bénéficier de représentations visuelles déjà apprises.

        **4. Adaptation du modèle**

        La couche finale du réseau est adaptée au problème de
        classification à deux classes.

        **5. Inférence**

        Une image est envoyée au modèle afin d'obtenir une classe
        prédite, une confiance et le temps nécessaire à l'inférence.

        **6. Déploiement**

        Une interface Streamlit permet de présenter le modèle
        et de tester directement le système avec de nouvelles images.
        """
    )


# ============================================================
# 3. ARCHITECTURE TECHNIQUE
# ============================================================

st.header("03. Architecture technique")

with st.container(border=True):

    st.subheader("🔄 Pipeline de traitement")

    st.info("📷 Image d'entrée")

    st.markdown("↓")

    st.info("🔧 Prétraitement")

    st.markdown("↓")

    st.info("📐 Resize 224 × 224")

    st.markdown("↓")

    st.info("🧠 ResNet-18")

    st.markdown("↓")

    st.info("🔬 Couche de classification")

    st.markdown("↓")

    st.info("🎯 Deux classes : STOP / NOT STOP")

    st.markdown("↓")

    st.success("📊 Classe prédite + confiance + temps d'inférence")


# ============================================================
# 4. TECHNOLOGIES UTILISEES
# ============================================================

st.header("04. Technologies utilisées")

with st.container(border=True):

    st.subheader("💻 Stack technique")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Langage", "Python")

    with col2:
        st.metric("Deep Learning", "PyTorch")

    with col3:
        st.metric("Computer Vision", "Torchvision")

    with col4:
        st.metric("Interface", "Streamlit")

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Images", "PIL")

    with col2:
        st.metric("Architecture", "ResNet-18")

    with col3:
        st.metric("Approche", "Transfer Learning")

    with col4:
        st.metric("Classes", "2")


# ============================================================
# 6. MODELE UTILISE
# ============================================================

st.header("06. Modèle utilisé")

col1, col2 = st.columns(2)

with col1:

    with st.container(border=True):

        st.subheader("🧠 Architecture")

        st.write("**Modèle :** ResNet-18")

        st.write("**Approche :** Transfer Learning")

        st.write("**Nombre de classes :** 2")


with col2:

    with st.container(border=True):

        st.subheader("📐 Paramètres")

        st.write("**Classes :** STOP / NOT STOP")

        st.write("**Taille d'entrée :** 224 × 224 pixels")

        st.write("**Sortie :** classe + confiance")


# ============================================================
# TEST EN TEMPS REEL
# ============================================================

st.header("🚦 Test en temps réel")

st.write(
    "Importez une image afin de tester directement le modèle "
    "de Computer Vision."
)

uploaded_file = st.file_uploader(
    "Choisissez une image à analyser",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# FONCTION D'INFERENCE
# ============================================================

def run_inference(image_path):

    command = [
        sys.executable,
        str(INFERENCE_FILE),
        str(image_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    return result.stdout, result.stderr, result.returncode


# ============================================================
# TEST
# ============================================================

if uploaded_file:

    col_image, col_result = st.columns(2)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col_image:

        st.subheader("📷 Image analysée")

        st.image(
            uploaded_file,
            use_container_width=True
        )

    # --------------------------------------------------------
    # RESULTAT
    # --------------------------------------------------------

    with col_result:

        st.subheader("🤖 Résultat du modèle")

        if not INFERENCE_FILE.exists():

            st.error(
                "Le fichier src/inference.py est introuvable."
            )

        else:

            suffix = Path(uploaded_file.name).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:

                tmp.write(uploaded_file.getbuffer())
                temp_path = Path(tmp.name)

            with st.spinner("Évaluation du modèle..."):

                stdout, stderr, returncode = run_inference(
                    temp_path
                )

            # ------------------------------------------------
            # ERREUR
            # ------------------------------------------------

            if returncode != 0:

                st.error("Erreur pendant l'inférence.")

                st.code(stderr)

            else:

                # ------------------------------------------------
                # EXTRACTION
                # ------------------------------------------------

                prediction = None
                confidence = None
                inference_time = None

                prediction_match = re.search(
                    r"(?:Classe prédite|Predicted class|Prediction|Classe)\s*[:=]\s*([A-Za-z_]+)",
                    stdout,
                    re.IGNORECASE
                )

                if prediction_match:
                    prediction = prediction_match.group(1)

                confidence_match = re.search(
                    r"(?:Confiance|Confidence)\s*[:=]\s*([0-9.,]+)\s*%",
                    stdout,
                    re.IGNORECASE
                )

                if confidence_match:
                    confidence = confidence_match.group(1)

                time_match = re.search(
                    r"(?:Temps d'inférence|Inference time|Temps)\s*[:=]\s*([0-9.,]+)\s*ms",
                    stdout,
                    re.IGNORECASE
                )

                if time_match:
                    inference_time = time_match.group(1)

                # ------------------------------------------------
                # RESULTAT PRINCIPAL
                # ------------------------------------------------

                if prediction:

                    st.success(
                        f"Classe prédite : {prediction.upper()}"
                    )

                else:

                    st.warning(
                        "La classe prédite n'a pas pu être extraite."
                    )

                # ------------------------------------------------
                # METRIQUES
                # ------------------------------------------------

                metric1, metric2 = st.columns(2)

                with metric1:

                    if confidence:

                        st.metric(
                            "Confiance",
                            f"{confidence}%"
                        )

                    else:

                        st.metric(
                            "Confiance",
                            "N/A"
                        )

                with metric2:

                    if inference_time:

                        st.metric(
                            "Temps d'inférence",
                            f"{inference_time} ms"
                        )

                    else:

                        st.metric(
                            "Temps d'inférence",
                            "N/A"
                        )

                # ------------------------------------------------
                # SORTIE TECHNIQUE
                # ------------------------------------------------

                with st.expander(
                    "Voir la sortie technique du modèle"
                ):

                    st.code(stdout)

            # ------------------------------------------------
            # NETTOYAGE
            # ------------------------------------------------

            try:
                temp_path.unlink()
            except Exception:
                pass


# ============================================================
# FOOTER
# ============================================================

st.markdown("")

st.markdown(
    """
    <div class="footer">
        STOP Sign Detection · Computer Vision ·
        Deep Learning · AI / Machine Learning Portfolio
    </div>
    """,
    unsafe_allow_html=True
)