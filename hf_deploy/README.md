---
title: ParkSense AI — Bengaluru Congestion Intelligence
emoji: 🚦
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ParkSense AI
Inference-only dashboard for the Bengaluru Traffic Police (ASTraM) parking congestion model.
Loads the pre-trained `parksense_model.pkl` — no training happens here.

See `app/inference.py` for the prediction logic and `app/app.py` for the Gradio UI.
