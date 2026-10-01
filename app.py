import io
import uuid
import base64

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.explaner import ask_follow_up, generate_explanation, reset_session
from src.gradcam import gradcam_heatmap, overlay_heatmap, summarise_activation_region
from src.model_loader import GRAD_CAM_LAYER_NAME, load_model
from src.predict import predict_image

load_dotenv()

st.set_page_config(
    page_title="Invasive Ductal Carcinoma Classifier and Explainer",
    layout = "wide"
)

st.set_page_config(page_title="Invasive Ductal Carcinoma Classifier and Explainer", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .block-container {
        padding-top: 2.5rem;
        max-width: 1100px;
    }

    /* ── Header ── */
    .app-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.15rem;
    }
    .app-header .dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #5B8DEF;
        display: inline-block;
        box-shadow: 0 0 8px rgba(91, 141, 239, 0.6);
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 600;
        color: #E7E9ED;
        margin: 0;
    }
    .app-subtitle {
        color: #838C9C;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* ── Result cards ── */
    .result-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.75rem;
        animation: fade-up 0.4s ease-out;
    }
    @keyframes fade-up {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .card {
        background: #181C24;
        border: 1px solid #262B36;
        border-radius: 10px;
        padding: 1rem;
    }
    .image-card {
        flex: 1;
        text-align: center;
    }
    .image-card img {
        width: 100%;
        border-radius: 6px;
        image-rendering: pixelated;
    }
    .image-card .caption {
        color: #838C9C;
        font-size: 0.8rem;
        margin-top: 0.6rem;
    }
    .reading-card {
        flex: 1.2;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.6rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .badge.benign {
        background: rgba(79, 169, 140, 0.15);
        color: #4FA98C;
    }
    .badge.malignant {
        background: rgba(193, 87, 63, 0.15);
        color: #C1573F;
    }
    .confidence-readout {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 2.4rem;
        color: #E7E9ED;
        line-height: 1;
    }
    .confidence-label {
        color: #838C9C;
        font-size: 0.8rem;
    }

    /* ── Explanation callout ── */
    .explanation-label {
        color: #838C9C;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.6rem;
    }
    [data-testid="stChatMessage"]:nth-of-type(odd) {
    background: #161A21 !important;
    border-left: 2px solid #5B8DEF;
    border-radius: 4px;
    padding: 0.75rem 1rem !important;
}
    [data-testid="stChatMessage"]:nth-of-type(even) {
    background: transparent !important;
    border-left: none;
    padding: 0.5rem 0 !important;
}

    /* ── Misc widget tweaks ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed #262B36;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stChatInput textarea {
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def _image_to_base64(arr_or_image) -> str:
    """Encode a PIL Image or numpy array as a base64 PNG string for inline <img> use."""
    if isinstance(arr_or_image, np.ndarray):
        arr_or_image = Image.fromarray(arr_or_image)
    buf = io.BytesIO()
    arr_or_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


st.markdown(
    """
    <div class="app-header">
        <span class="dot"></span>
        <h1>Invasive Ductal Carcinoma Classifier and Explainer</h1>
    </div>
    <div class="app-subtitle">
        Upload a 50×50 histopathology patch for a prediction, a Grad-CAM
        activation reading, and a plain-language explanation you can ask
        follow-up questions about.
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()

uploaded_file = st.file_uploader("Upload a patch image", type=["png", "jpg", "jpeg"])

# Ensures context from previous images is excluded
if uploaded_file is not None:
    if st.session_state.get("current_file_name") != uploaded_file.name:
        old_session_id = st.session_state.get("session_id")
        if old_session_id:
            reset_session(old_session_id)

        st.session_state.current_file_name = uploaded_file.name
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_messages = []
        st.session_state.explanation_generated = False

    image = Image.open(uploaded_file)

    with st.spinner("Running prediction..."):
        prediction = predict_image(model, image)

    with st.spinner("Generating Grad-CAM..."):
        heatmap = gradcam_heatmap(
            prediction["input_array"],
            model,
            GRAD_CAM_LAYER_NAME
        )
        original_resized = np.array(image.convert("RGB").resize((50, 50)))
        overlay = overlay_heatmap(heatmap, original_resized)
        activation_summary = summarise_activation_region(heatmap)

    badge_class = "malignant" if prediction["predicted_class"] == 1 else "benign"
    original_b64 = _image_to_base64(image.convert("RGB").resize((200, 200), Image.NEAREST))
    overlay_b64 = _image_to_base64(
        Image.fromarray(overlay).resize((200, 200), Image.NEAREST)
    )

    st.markdown(
        f"""
           <div class="result-row">
               <div class="card image-card">
                   <img src="data:image/png;base64,{original_b64}" />
                   <div class="caption">Original patch</div>
               </div>
               <div class="card image-card">
                   <img src="data:image/png;base64,{overlay_b64}" />
                   <div class="caption">Grad-CAM overlay</div>
               </div>
               <div class="card reading-card">
                   <span class="badge {badge_class}">{prediction['label']}</span>
                   <div>
                       <div class="confidence-readout">{prediction['confidence']:.1%}</div>
                       <div class="confidence-label">model confidence</div>
                   </div>
               </div>
           </div>
           """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="explanation-label">Explanation</div>', unsafe_allow_html=True)

    #Generate the initial explanation only once per image
    if not st.session_state.explanation_generated:
        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(
                prediction,
                activation_summary,
                st.session_state.session_id
            )
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": explanation}
        )
        st.session_state.explanation_generated = True

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    followup = st.chat_input("Ask a follow-up question about this prediction...")
    if followup:
        st.session_state.chat_messages.append({"role": "user", "content": followup})
        with st.chat_message("user"):
            st.write(followup)

        with st.spinner("Thinking..."):
            answer = ask_follow_up(followup, st.session_state.session_id)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
else:
    st.info("Upload a patch image to get started.")