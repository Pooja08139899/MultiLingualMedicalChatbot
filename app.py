# app.py
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from datetime import datetime

# Argos translate (offline). If translation packages (.argosmodel) are provided in /translation_models,
# they will be installed automatically at startup.
try:
    import argostranslate.package as argopkg
    import argostranslate.translate as argotrans
    ARGOS_AVAILABLE = True
except Exception:
    ARGOS_AVAILABLE = False

# -------- CONFIG --------
# You can set MODEL_REPO to either:
#  - a HuggingFace repo id (e.g. "username/medical_flan_t5_lora"), or
#  - a folder included in the Space named "medical_flan_t5_lora"
MODEL_REPO = os.environ.get("MODEL_REPO", "medical_flan_t5_lora")
CSV_PATH = "chat_history.csv"
TRANSLATION_MODELS_FOLDER = "translation_models"  # optional: upload .argosmodel files here

# create CSV if not exists
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["timestamp","user_lang","user_text","response"]).to_csv(CSV_PATH, index=False)

# -------- Argos install helper (optional) --------
def install_argos_models_from_folder(folder):
    if not ARGOS_AVAILABLE:
        return False
    if not os.path.isdir(folder):
        return False
    installed = False
    for filename in os.listdir(folder):
        if filename.endswith(".argosmodel"):
            path = os.path.join(folder, filename)
            try:
                argopkg.install_from_path(path)
                installed = True
            except Exception as e:
                st.sidebar.error(f"Failed to install {filename}: {e}")
    return installed

# Attempt to install any uploaded argosmodels (only if argos is available)
if ARGOS_AVAILABLE:
    install_argos_models_from_folder(TRANSLATION_MODELS_FOLDER)

# -------- Translation wrapper using Argos (offline) --------
def translate_text(text, from_code, to_code):
    """
    Translate using argostranslate if available, otherwise return original text.
    from_code/to_code are language codes like 'en', 'hi', 'es', 'fr', 'de', 'ta'
    """
    if not ARGOS_AVAILABLE:
        return text
    try:
        # find installed translation pair
        for pkg in argotrans.get_installed_languages():
            pass
        # argostranslate API: translate(text, from_code, to_code)
        return argotrans.translate(text, from_code, to_code)
    except Exception:
        return text

# -------- Model loader (cached) --------
@st.cache_resource
def load_model(model_repo):
    # Load tokenizer and model from either a repo id or local folder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_repo)
    model.to(device)
    return tokenizer, model, device

# Attempt to load model (Spaces will either have the folder or you set MODEL_REPO to HF repo id)
with st.spinner("Loading model (this can take a minute)..."):
    try:
        tokenizer, model, device = load_model(MODEL_REPO)
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_REPO}: {e}")
        st.stop()

# -------- Streamlit UI --------
st.set_page_config(page_title="Multilingual Medical Chatbot", layout="wide")
st.title("🌐 Multilingual Medical Chatbot (HF Space)")

col1, col2 = st.columns([3,1])
with col2:
    st.markdown("**Settings**")
    lang_options = {"English":"en","Hindi":"hi","Spanish":"es","French":"fr","German":"de","Tamil":"ta"}
    user_lang_label = st.selectbox("Your language", list(lang_options.keys()), index=0)
    user_lang = lang_options[user_lang_label]

with col1:
    user_input = st.text_area("Ask a medical question", height=120)

if st.button("Send") and user_input:
    st.session_state = getattr(st, "session_state", st.session_state)
    with st.spinner("Translating & generating answer..."):
        # Translate to English (model trained to answer in English)
        if user_lang != "en" and ARGOS_AVAILABLE:
            user_input_en = translate_text(user_input, user_lang, "en")
        else:
            user_input_en = user_input

        # Prepare prompt for T5-flan style
        prompt = "question: " + user_input_en
        inputs = tokenizer([prompt], return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)

        # Generate
        gen_kwargs = {"max_length":128, "num_beams":2, "early_stopping":True}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        response_en = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Translate back to user language if needed
        if user_lang != "en" and ARGOS_AVAILABLE:
            response_user = translate_text(response_en, "en", user_lang)
        else:
            response_user = response_en

        # Save to CSV
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_lang": user_lang,
            "user_text": user_input,
            "response": response_user
        }
        try:
            pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)
        except Exception:
            # ignore write errors on Spaces storage
            pass

        st.markdown("**Answer:**")
        st.write(response_user)

# Show recent history
st.markdown("---")
st.markdown("### Recent Chat History")
try:
    hist = pd.read_csv(CSV_PATH).tail(20).iloc[::-1]
    for _, r in hist.iterrows():
        st.markdown(f"**User ({r.user_lang}):** {r.user_text}")
        st.markdown(f"**Bot:** {r.response}")
        st.markdown("---")
except Exception:
    st.info("No chat history yet.")
