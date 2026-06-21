"""
Coleta automatica de dados para treinar o classificador
de estado do olho. Usa o EAR para rotular cada recorte de olho como
'aberto' ou 'fechado' -- sem nenhuma anotacao manual.

Como usar:
  python coletar_dados.py
  -> Mantenha os olhos ABERTOS por ~30s, depois FECHADOS por ~30s,
     variando um pouco a posicao da cabeca. ESC para sair.

Resultado: imagens salvas em dataset/aberto/ e dataset/fechado/.
"""

import os

import cv2
import numpy as np
import mediapipe as mp

# Reaproveita o que ja funciona no DMS
from detector_dms import (
    garantir_permissao_camera, criar_landmarker, coords, calcular_ear,
    OLHO_DIR, OLHO_ESQ,
)

EAR_LIMIAR = 0.21  # limiar para rotular fechado x aberto durante a coleta
TAM = 64          # tamanho do recorte salvo (TAM x TAM pixels)
MARGEM = 0.6      # margem extra ao redor do olho
PASTA = "dataset"
GAP = 0.07        # zona morta entre 'fechado' e 'aberto' (evita rotulo ambiguo)


def recortar_olho(frame, pts):
    """Recorta a regiao do olho a partir dos 6 pontos do EAR."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    mx = (x2 - x1) * MARGEM + 5
    my = (y2 - y1) * MARGEM + 5
    h, w = frame.shape[:2]
    x1 = max(0, int(x1 - mx)); x2 = min(w, int(x2 + mx))
    y1 = max(0, int(y1 - my)); y2 = min(h, int(y2 + my))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], (TAM, TAM))


def main():
    garantir_permissao_camera()
    os.makedirs(os.path.join(PASTA, "aberto"), exist_ok=True)
    os.makedirs(os.path.join(PASTA, "fechado"), exist_ok=True)

    landmarker = criar_landmarker()
    cap = cv2.VideoCapture(0)
    n_aberto = len(os.listdir(os.path.join(PASTA, "aberto")))
    n_fechado = len(os.listdir(os.path.join(PASTA, "fechado")))
    frame_id = 0

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
            for idxs in (OLHO_DIR, OLHO_ESQ):
                pts = coords(lm, idxs, w, h)
                ear = calcular_ear(pts)
                rec = recortar_olho(frame, pts)
                if rec is None:
                    continue
                # Anotacao automatica pelo EAR
                if ear < EAR_LIMIAR:
                    cv2.imwrite(f"{PASTA}/fechado/{n_fechado:05d}.png", rec)
                    n_fechado += 1
                elif ear > EAR_LIMIAR + GAP:
                    cv2.imwrite(f"{PASTA}/aberto/{n_aberto:05d}.png", rec)
                    n_aberto += 1

            cv2.putText(frame, f"aberto: {n_aberto}   fechado: {n_fechado}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Olhos ABERTOS, depois FECHADOS. ESC p/ sair.",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Coleta de dados (anotacao automatica)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nColetado: {n_aberto} abertos + {n_fechado} fechados em '{PASTA}/'")


if __name__ == "__main__":
    main()
