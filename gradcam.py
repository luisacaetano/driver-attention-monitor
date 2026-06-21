"""
Grad-CAM da CNN do olho: mostra um mapa de calor de "onde a CNN olhou" para
decidir aberto/fechado. O heatmap deve acender sobre a palpebra/olho -- prova
visual de que a rede aprendeu a regiao certa. Otimo para a apresentacao.

Rode:  python gradcam.py
  - ESC sai
  - tecla 's' salva a figura atual (gradcam_<hora>.png) para slides/artigo
"""

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp

from detector_dms import (
    garantir_permissao_camera, criar_landmarker, coords, recortar_olho,
    OLHO_DIR, TAM_OLHO,
)
from modelos import CNN

CLASSES = ["ABERTO", "FECHADO"]   # ordem ImageFolder: 0=aberto, 1=fechado

# ---- modelo + hooks no ultimo Conv2d ----
modelo = CNN()
modelo.load_state_dict(torch.load("cnn.pt", map_location="cpu"))
modelo.eval()

_cache = {}
_conv = modelo.features[6]   # ultima camada convolucional (64 x 16 x 16)
_conv.register_forward_hook(lambda m, i, o: _cache.__setitem__("a", o))
_conv.register_full_backward_hook(
    lambda m, gi, go: _cache.__setitem__("g", go[0].detach()))


def grad_cam(crop_bgr):
    """Recebe recorte BGR 64x64 do olho. Devolve (heatmap 0..1, classe, conf)."""
    cinza = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    x = torch.from_numpy(cinza).float().div(255.0).unsqueeze(0).unsqueeze(0)
    x.requires_grad_(True)

    saida = modelo(x)                       # forward (dispara o hook)
    classe = int(saida.argmax(1))
    conf = float(torch.softmax(saida, 1)[0, classe].detach())

    modelo.zero_grad()
    saida[0, classe].backward()             # backward (dispara o hook)

    ativ = _cache["a"][0]                    # 64 x 16 x 16
    grad = _cache["g"][0]                    # 64 x 16 x 16
    pesos = grad.mean(dim=(1, 2))            # importancia de cada filtro
    cam = F.relu((pesos[:, None, None] * ativ).sum(0))   # 16 x 16
    cam = cam / (cam.max() + 1e-6)
    cam = cv2.resize(cam.detach().numpy(), (TAM_OLHO, TAM_OLHO))
    return cam, classe, conf


def painel(crop_bgr, cam, classe, conf, lado=260):
    """Monta o painel: olho ampliado | heatmap sobreposto, com rotulo."""
    olho = cv2.resize(crop_bgr, (lado, lado), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (lado, lado), interpolation=cv2.INTER_LINEAR)
    over = cv2.addWeighted(olho, 0.55, heat, 0.45, 0)

    sep = np.zeros((lado, 8, 3), np.uint8)
    comp = np.hstack([olho, sep, over])
    faixa = np.zeros((52, comp.shape[1], 3), np.uint8)
    comp = np.vstack([faixa, comp])
    cv2.putText(comp, "olho", (lado // 2 - 28, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    cor = (90, 90, 255) if classe == 1 else (120, 230, 120)
    cv2.putText(comp, f"CNN olhou aqui -> {CLASSES[classe]} {conf*100:.0f}%",
                (lado + 16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)
    return comp


def main():
    garantir_permissao_camera()
    landmarker = criar_landmarker()
    cap = cv2.VideoCapture(0)
    frame_id = 0
    ultimo = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_id += 1
        res = landmarker.detect_for_video(mp_image, frame_id * 33)

        if res.face_landmarks:
            lm = res.face_landmarks[0]
            crop = recortar_olho(frame, coords(lm, OLHO_DIR, w, h))
            if crop is not None:
                cam, classe, conf = grad_cam(crop)
                ultimo = painel(crop, cam, classe, conf)

        if ultimo is not None:
            cv2.imshow("Grad-CAM - onde a CNN olha (s=salvar, ESC=sai)", ultimo)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        if tecla == ord("s") and ultimo is not None:
            nome = time.strftime("gradcam_%H%M%S.png")
            cv2.imwrite(nome, ultimo)
            print(f"figura salva: {nome}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
