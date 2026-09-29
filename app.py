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
INFERENCE_FILE = BASE_DIR / "Streamlit_ai" /"src" / "inference.py"

# ============================================================
# STYLE BLEU-NUIT
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
    margin-bottom: 30px;
}

.card {
    background: #0d1b2e;
    border: 1px solid #1e3a5f;
    border-radius: 18px;
    padding: 25px;
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
    margin-top: 50px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-title">
    STOP Sign <span>Detection</span>
</div>

<div class="subtitle">
    Interface web d'évaluation du modèle de Computer Vision
</div>
""", unsafe_allow_html=True)


# ============================================================
# INFORMATIONS MODELE
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <div class="metric-title">MODEL</div>
        <div class="metric-value">ResNet-18</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="metric-title">CLASSES</div>
        <div class="metric-value">2</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="metric-title">CLASS 01</div>
        <div class="metric-value">STOP</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="metric-title">CLASS 02</div>
        <div class="metric-value">NOT STOP</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# UPLOAD
# ============================================================

st.markdown("## Évaluer une image")

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
# TRAITEMENT
# ============================================================

if uploaded_file:

    col_image, col_result = st.columns([1, 1])

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col_image:

        st.markdown("### Image analysée")

        st.image(
            uploaded_file,
            use_container_width=True
        )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with col_result:

        st.markdown("### Résultat")

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
            # RESULTAT
            # ------------------------------------------------

            if returncode != 0:

                st.error("Erreur pendant l'inférence.")

                st.code(stderr)

            else:

                # Affichage brut pour ne jamais cacher
                # les informations produites par inference.py

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.markdown("#### Sortie du modèle")

                st.code(stdout)

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

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
                # RESULTAT VISUEL
                # ------------------------------------------------

                if prediction:

                    st.markdown(f"""
                    <div class="result">

                        <div class="metric-title">
                            CLASSE PRÉDITE
                        </div>

                        <div class="prediction">
                            {prediction.upper()}
                        </div>

                    """, unsafe_allow_html=True)

                    if confidence:

                        st.markdown(f"""
                            <div class="confidence">
                                {confidence}%
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown(
                        "</div>",
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
                                {confidence + '%' if confidence else 'Voir sortie'}
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
                                {inference_time + ' ms' if inference_time else 'Voir sortie'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Nettoyage
            try:
                temp_path.unlink()
            except Exception:
                pass


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    STOP Sign Detection · AI / Machine Learning Portfolio
</div>
""", unsafe_allow_html=True)