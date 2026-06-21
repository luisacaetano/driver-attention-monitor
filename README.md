# Driver Attention Monitor (DMS)

A real-time **Driver Monitoring System** that estimates a **Driver Attention
Score (0–100)** from a regular webcam, combining classic computer vision with
deep learning. It is the final project for the **Computer Vision** course
(IFMG – Campus Formiga).

> The system does not measure only fatigue — it measures **attention**, fusing
> eye state, yawning, microsleep, head pose, gaze direction and hand-held
> objects into a single score.

## Highlights

- 🧠 **Two CNNs trained from scratch** — eye state (open/closed, **96%**) and
  yawning (**98%**), each with full metrics (precision / recall / F1).
- 🔬 **Architecture study** — the custom CNN vs. ResNet18, MobileNetV3,
  EfficientNet-B0 and a small ViT (transfer learning), comparing accuracy,
  speed and size.
- 🎯 **Decision fusion** — the CNN reports a confidence; below 70% it falls back
  to the geometric EAR.
- ⚖️ **Ensemble** — CNN + EAR + PERCLOS combined with weights into one
  drowsiness signal (a normal blink does **not** lower the score; a long,
  drowsy blink does).
- 🔥 **Grad-CAM** — a heatmap proving the CNN looks at the eyelid, not the
  background (`gradcam.py`).
- 📊 **Driver Attention Score** with bands: Excellent → Good → Distracted →
  Alert → Critical.
- 🚨 **Smart alerts** following the **NHTSA 2-second rule**: a glance off the
  road of up to 2 s is fine; beyond that it is flagged. Spoken alerts
  ("look at the road", "look ahead", "stop and rest").
- 📼 **Black box** (events CSV + video clips) and a **trip summary** screen.

## How it works

| Signal | Method |
|---|---|
| Eye open/closed | **CNN** (deep learning) with EAR fallback |
| Yawning | **CNN** of the mouth (compared with the MAR) |
| Microsleep | eyes closed for more than 2 s |
| PERCLOS | % of time with eyes closed (gold-standard fatigue metric) |
| Head pose | yaw / pitch via `solvePnP` |
| Gaze | iris position (looking down = phone) |
| Hand-held object | EfficientDet (phone, book, cup, bottle, remote) |
| Exit gesture | open palm |

## Run

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

python detector_dms.py              # full system (live)
python gradcam.py                   # Grad-CAM demo (press 's' to save a figure)
```

### Train the models (optional — pre-trained weights are included)

```bash
python coletar_dados.py             # collect eye dataset (auto-labelled by EAR)
python treino.py                    # train the eye CNN -> cnn.pt

python coletar_boca.py              # collect mouth dataset (auto-labelled by MAR)
python treino_boca.py               # train the yawn CNN -> cnn_bocejo.pt

python comparar_arquiteturas.py     # CNN vs ResNet/MobileNet/EfficientNet/ViT
```

## Results

| Model | Accuracy | Inference | Size |
|---|---|---|---|
| **Custom CNN** | **96.0%** | **0.46 ms** | **1.1 MB** |
| ViT (small) | 96.2% | 1.68 ms | 5.0 MB |
| ResNet18 | 94.6% | 5.37 ms | 44.7 MB |
| EfficientNet-B0 | 91.5% | 219 ms | 16.0 MB |
| MobileNetV3 | 85.8% | 81 ms | 6.1 MB |

The small custom CNN matches the accuracy of the much larger pre-trained models
while being **~40× smaller** and **far faster** — ideal for real-time, on-device
driver monitoring. Full numbers in `comparacao_arquiteturas.txt`,
`resultados.txt` and `resultados_boca.txt`.

## Scientific basis

- **2-second rule** — off-road glances over 2 s roughly double crash risk
  (NHTSA / VTTI 100-Car Study).
- **PERCLOS** — eyes closed > 15% of the time indicates drowsiness (NHTSA
  gold standard).

## Author

Luisa Caetano Araújo — Computer Vision, IFMG Campus Formiga
(Prof. Me. Fernando Paim Lima).
