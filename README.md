# MultiLingualMedicalChatbot
# Multilingual Medical Chatbot (Hugging Face Space)

This Streamlit app is configured to run on **Hugging Face Spaces** with GPU runtime.
It uses a Flan-T5 style seq2seq model (LoRA fine-tuned recommended) and Argos Translate for offline translation.

## Files
- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies
- Optional folder: `translation_models/` — upload any `.argosmodel` packages here for offline translation

## How to deploy
1. Create a new **Space** on Hugging Face: choose **Streamlit** and **GPU** runtime.
2. Push this repository to your new Space (or upload files via the web UI).
3. If your fine-tuned model is large, you can:
   - **Option A**: upload the model folder (`medical_flan_t5_lora/`) into the repo directly (small → ok).
   - **Option B** (recommended): push your model to the Hugging Face Hub as a model repo, then set `MODEL_REPO` in Space settings (or the `app.py` environment) to `username/medical_flan_t5_lora`. Example: `MODEL_REPO = "yourname/medical_flan_t5_lora"`.
4. (Optional) Put Argos Translate packages (`.argosmodel` files) into the repo under `translation_models/`. The app will install them at startup.
5. Wait for the Space to build — once it’s ready, you’ll get a public URL.

## Notes
- The app uses Argos Translate for offline translation. If no argos packages are provided, the app gracefully falls back to no translation (i.e., English-only).
- If you use a model hosted on the HF Hub, ensure the Space has permission/access to download it.
