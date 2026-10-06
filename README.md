# Invasive Ductal Carcinoma Classifier with Grad-CAM visualisation and LLM Support

### Overview
An AI-powered support tool intended as a research demo. Combines a previously trained CNN image classifier's prediction and Grad-CAM activation-maps with a Google Gemini-generated explanation all hosted on Streamlit interface.
Intended to be a support tool for oncological clinicians.

## Key Demonstrations in project

* Reloading and implementing a trained TensorFlow/Keras CNN outside the training environment.
* Grad-CAM generation with explainability upon inference.
* Integration of multiple systems:
  - Model output
  - Structured summary leveraging an LLM via LangChain (Gemini backend)
* Follow-up questions can be asked by the user in a conversational interface about the output result. The prediction and Grad-CAM context is preserved throughout the conversation.
* A deployable Streamlit interface with a deliberate visual design for its purpose.

## Setup

Requires **Python 3.11 or 3.12** for compatibility with **Tensorflow 2.17.0**. Essential for model training environment
matching.

### Using `bash`
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env #Add GEMINI_API_KEY
```

### Using `cmd`
```commandline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Model

The app loads `model/idc_cnn_final.h5`, a legacy format of the .keras file suited to personal computational power. Contains model architecture, weights and optimiser state. Using the SavedModel format won't work for this application because the reconstructable layer structure is required for the Grad-CAM.

The model had some custom layers built into the solution so loading the model requires those to be registered. The custom objects (`CastToFloat16` and `CastToFloat32`) from the original notebook were loaded with the model using `load_model()` in `src\model_loader.py` with `compile=False` to skip reconstructing config aspects not required during inference. The `GRAD_CAM_LAYER_NAME` is set to `"relu4b"`. This is the last activation before model's global average pooling layer.

The prediction step in `predict.py` needs only raw 0-255 pixel values passed into it as normalisation is handled implicitly by the model's own built-in Augmentation and Normalisation layers.

## UI Design

Dark, cool-toned colour palette intended to suit the purpose of the system implementation as a support tool for an oncology clinician. The CSS targets Streamlit's `data-testid` attributes for chat\file-uploader styling, which can shift between Streamlit versions.

## Running using `bash` or `cmd`

```bash
streamlit run app.py
```

## Project structure

```
model/                          # The model saved in various formats
    idc_cnn_final.keras
    idc_cnn_final.h5 
app.py                          # Streamlit entry point and UI Styling
.streamlit/
    config.toml                 # Base theme colour configuration
src/
    __init__.py
    model_loader.py             # Loads the model, cached across reruns
    predict.py                  # Image preprocessing and inference
    explainer.py                # LangChain-based explanation and conversation set-up
    gradcam.py                  # Heatmap generation and overlay placement
          
```

## Known Limitations

* Not a diagnostic tool, intended for support alone
* Patch resolution presents constraint, input patches are 50x50 pixels which yield a coarse 6x6px Grad-CAM grid
* The LLM has no access to the training dataset so it cannot answer any questions to that effect
* Free-tier LLM rate limits should be noted if using the model currently in the solution
