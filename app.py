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
    background: #07111f;
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: #07111f;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 5px;
}

.main-title span {
    color: #3b82f6;
}

.subtitle {
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 35px;
}

.section-title {
    font-size: 30px;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 45px;
    margin-bottom: 15px;
}

.section-number {
    color: #3b82f6;
}

.card {
    background: #0d1b2e;
    border: 1px solid #1e3a5f;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 12px;
}

.card-text {
    color: #cbd5e1;
    font-size: 16px;
    line-height: 1.7;
}

.tech {
    display: inline-block;
    background: #102542;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 10px 16px;
    margin: 5px;
    color: #dbeafe;
    font-weight: 600;
}

.architecture {
    background: #081525;
    border: 1px solid #1e3a5f;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 15px;
    margin-bottom: 20px;
}

.arch-step {
    background: #0d1b2e;
    border: 1px solid #2563eb;
    border-radius: 12px;
    padding: 14px;
    margin: 8px auto;
    max-width: 450px;
    color: #f8fafc;
    font-weight: 600;
}

.arrow {
    color: #60a5fa;
    font-size: 22px;
    font-weight: bold;
}

.model-box {
    background: #0d1b2e;
    border: 1px solid #2563eb;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.model-label {
    color: #94a3b8;
    font-size: 13px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.model-value {
    color: #f8fafc;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

.test-box {
    background: #0b1728;
    border: 1px solid #2563eb;
    border-radius: 20px;
    padding: 30px;
    margin-top: 20px;
}

.result {
    text-align: center;
    background: #0d1b2e;
    border: 1px solid #2563eb;
    border-radius: 20px;
    padding: 35px;
}

.prediction {
    font-size: 48px;
    font-weight: 800;
    color: #60a5fa;
}

.confidence {
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: #f8fafc;
    font-size: 25px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 60px;
    padding: 25px;
    border-top: 1px solid #1e3a5f;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-title">
    🚦 STOP Sign <span>Detection</span>
</div>

<div class="subtitle">
    Computer Vision · Deep Learning · Image Classification
</div>
""", unsafe_allow_html=True)


# ============================================================
# 1. PRESENTATION DU PROJET
# ============================================================

st.markdown("""
<div class="section-title">
    <span class="section-number">01.</span> Présentation du projet
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("""
    <div class="card">

        <div class="card-title">
            🎯 Objectif
        </div>

        <div class="card-text">

        Ce projet consiste à développer un système de
        <strong>Computer Vision</strong> capable d'analyser une image
        et de déterminer automatiquement si elle contient un
        <strong>panneau STOP</strong> ou non.

        <br><br>

        Le problème est traité comme une tâche de
        <strong>classification d'images binaire</strong> avec deux classes :

        <br><br>

        <strong>STOP</strong> → présence d'un panneau STOP

        <br>

        <strong>NOT STOP</strong> → absence d'un panneau STOP

        </div>

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">

        <div class="card-title">
            🔎 Problématique
        </div>

        <div class="card-text">

        Comment utiliser les techniques de
        <strong>Deep Learning et de Computer Vision</strong>
        pour permettre à un système informatique de reconnaître
        automatiquement un panneau STOP à partir d'une image ?

        <br><br>

        Le projet met en œuvre un modèle de classification
        capable de produire une prédiction accompagnée de son
        <strong>niveau de confiance</strong>.

        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 2. TRAVAIL REALISE
# ============================================================

st.markdown("""
<div class="section-title">
    <span class="section-number">02.</span> Travail réalisé
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">

<div class="card-title">
    🛠️ Pipeline de développement
</div>

<div class="card-text">

<strong>1. Préparation des données</strong><br>
Organisation des images dans les différentes catégories
nécessaires à l'entraînement, à la validation et au test.

<br><br>

<strong>2. Prétraitement des images</strong><br>
Les images sont préparées avant leur passage dans le réseau
de neurones afin de respecter le format attendu par le modèle.

<br><br>

<strong>3. Transfer Learning</strong><br>
Utilisation d'une architecture ResNet-18 pré-entraînée afin
de bénéficier de représentations visuelles déjà apprises.

<br><br>

<strong>4. Adaptation du modèle</strong><br>
La couche finale du réseau est adaptée au problème de
classification à deux classes.

<br><br>

<strong>5. Inférence</strong><br>
Une image peut ensuite être envoyée au modèle afin d'obtenir
une classe prédite, une confiance et le temps nécessaire
à l'inférence.

<br><br>

<strong>6. Déploiement</strong><br>
Une interface Streamlit permet de présenter le modèle et
de tester directement le système avec de nouvelles images.

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# 3. ARCHITECTURE TECHNIQUE
# ============================================================

st.markdown("""
<div class="section-title">
    <span class="section-number">03.</span> Architecture technique
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="architecture">

    <div class="arch-step">
        📷 Image d'entrée
    </div>

    <div class="arrow">↓</div>

    <div class="arch-step">
        🔧 Prétraitement
    </div>

    <div class="arrow">↓</div>

    <div class="arch-step">
        📐 Resize 224 × 224
    </div>

    <div class="arrow">↓</div>

    <div class="arch-step">
        🧠 ResNet-18
    </div>

    <div class="arrow">↓</div>

    <div class="arch-step">
        🔬 Couche de classification
    </div>

    <div class="arrow">↓</div>

    <div class="arch-step">
        🎯 2 classes
    </div>

    <div class="arrow">↓</div>

    <div class="arch-step">
        🚦 STOP / NOT STOP
    </div>

    <div class="arrow">↓</div>

    <div class="arch-step">
        📊 Confiance + temps d'inférence
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# 4. TECHNOLOGIES UTILISEES
# ============================================================

st.markdown("""
<div class="section-title">
    <span class="section-number">04.</span> Technologies utilisées
</div>
""", unsafe_allow_html=True)

technologies = [
    "Python",
    "PyTorch",
    "Torchvision",
    "PIL",
    "Streamlit",
    "Computer Vision",
    "Deep Learning",
    "ResNet-18"
]

tech_html = ""

for tech in technologies:
    tech_html += f'<span class="tech">{tech}</span>'

st.markdown(
    f"""
    <div class="card">
        <div class="card-title">
            💻 Stack technique
        </div>
        <div style="margin-top:15px;">
            {tech_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. MODELE UTILISE
# ============================================================

st.markdown("""
<div class="section-title">
    <span class="section-number">06.</span> Modèle utilisé
</div>
""", unsafe_allow_html=True)

model_col1, model_col2 = st.columns(2)

with model_col1:

    st.markdown("""
    <div class="model-box">

        <div class="model-label">
            Architecture
        </div>

        <div class="model-value">
            ResNet-18
        </div>

        <div class="model-label">
            Approche
        </div>

        <div class="model-value">
            Transfer Learning
        </div>

        <div class="model-label">
            Nombre de classes
        </div>

        <div class="model-value">
            2
        </div>

    </div>
    """, unsafe_allow_html=True)

with model_col2:

    st.markdown("""
    <div class="model-box">

        <div class="model-label">
            Classes
        </div>

        <div class="model-value">
            STOP / NOT STOP
        </div>

        <div class="model-label">
            Taille d'entrée
        </div>

        <div class="model-value">
            224 × 224 pixels
        </div>

        <div class="model-label">
            Sortie
        </div>

        <div class="model-value">
            Classe + confiance
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TEST EN TEMPS REEL
# ============================================================

st.markdown("""
<div class="section-title">
    🚦 Test en temps réel
</div>

<div class="subtitle">
    Importez une image afin de tester directement le modèle de Computer Vision.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="test-box">
    <div class="card-title">
        🧪 Évaluation du modèle
    </div>
    <div class="card-text">
        Le système va analyser l'image fournie et retourner
        la classe prédite, le niveau de confiance et le temps
        d'inférence.
    </div>
</div>
""", unsafe_allow_html=True)

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
# TRAITEMENT DU TEST
# ============================================================

if uploaded_file:

    col_image, col_result = st.columns([1, 1])

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col_image:

        st.markdown("### 📷 Image analysée")

        st.image(
            uploaded_file,
            use_container_width=True
        )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with col_result:

        st.markdown("### 🤖 Résultat du modèle")

        if not INFERENCE_FILE.exists():

            st.error(
                "Le fichier inference.py est introuvable."
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

                st.error(
                    "Erreur pendant l'inférence."
                )

                st.code(stderr)

            else:

                # ------------------------------------------------
                # EXTRACTION DES RESULTATS
                # ------------------------------------------------

                prediction = None
                confidence = None
                inference_time = None

                # Prediction

                prediction_match = re.search(
                    r"(?:Classe prédite|Predicted class|Prediction|Classe)\s*[:=]\s*([A-Za-z_]+)",
                    stdout,
                    re.IGNORECASE
                )

                if prediction_match:
                    prediction = prediction_match.group(1)

                # Confidence

                confidence_match = re.search(
                    r"(?:Confiance|Confidence)\s*[:=]\s*([0-9.,]+)\s*%",
                    stdout,
                    re.IGNORECASE
                )

                if confidence_match:
                    confidence = confidence_match.group(1)

                # Inference time

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

                    st.markdown(
                        f"""
                        <div class="result">

                            <div class="metric-title">
                                CLASSE PRÉDITE
                            </div>

                            <div class="prediction">
                                {prediction.upper()}
                            </div>

                            {
                                f'<div class="confidence">{confidence}%</div>'
                                if confidence
                                else ''
                            }

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------
                # METRIQUES
                # ------------------------------------------------

                metric1, metric2 = st.columns(2)

                with metric1:

                    st.markdown(
                        f"""
                        <div class="card">

                            <div class="metric-title">
                                CONFIANCE
                            </div>

                            <div class="metric-value">
                                {
                                    confidence + '%'
                                    if confidence
                                    else 'Voir sortie'
                                }
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with metric2:

                    st.markdown(
                        f"""
                        <div class="card">

                            <div class="metric-title">
                                TEMPS D'INFÉRENCE
                            </div>

                            <div class="metric-value">
                                {
                                    inference_time + ' ms'
                                    if inference_time
                                    else 'Voir sortie'
                                }
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------
                # SORTIE TECHNIQUE
                # ------------------------------------------------

                with st.expander("Voir la sortie technique du modèle"):

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

st.markdown("""
<div class="footer">

    <strong>STOP Sign Detection</strong><br>

    Computer Vision · Deep Learning · AI / Machine Learning Portfolio

</div>
""", unsafe_allow_html=True)