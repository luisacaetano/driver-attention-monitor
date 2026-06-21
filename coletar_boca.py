"""
Coleta automatica de dados para treinar a CNN de BOCEJO.
Usa o MAR (Mouth Aspect Ratio) para rotular cada recorte de boca como
'bocejando' (boca bem aberta) ou 'normal' -- sem anotacao manual.

Como usar:
  python coletar_boca.py
  -> Faca varios BOCEJOS (boca bem aberta) por ~30s; depois fique NORMAL
     (boca fechada, falando, sorrindo) por ~30s, variando a cabeca. ESC sai.

Resultado: imagens em dataset_boca/bocejando/ e dataset_boca/normal/.
"""

import os

import cv2
import mediapipe as mp

# Reaproveita o que ja existe no DMS
from detector_dms import (
    garantir_permissao_camera, criar_landmarker, coords, calcular_mar,
    recortar_boca, BOCA_VERT, BOCA_HORIZ, BOCA_BOX,
)

MAR_BOCEJO = 0.60   # acima disso = bocejando
MAR_NORMAL = 0.35   # abaixo disso = normal (entre os dois = ignorado)
PASTA = "dataset_boca"


def main():
    garantir_permissao_camera()
    os.makedirs(os.path.join(PASTA, "bocejando"), exist_ok=True)
    os.makedirs(os.path.join(PASTA, "normal"), exist_ok=True)

    landmarker = criar_landmarker()
    cap = cv2.VideoCapture(0)
    n_boc = len(os.listdir(os.path.join(PASTA, "bocejando")))
    n_norm = len(os.listdir(os.path.join(PASTA, "normal")))
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
            mar = calcular_mar(coords(lm, BOCA_VERT, w, h),
                               coords(lm, BOCA_HORIZ, w, h))
            rec = recortar_boca(frame, coords(lm, BOCA_BOX, w, h))
            if rec is not None:
                if mar > MAR_BOCEJO:
                    cv2.imwrite(f"{PASTA}/bocejando/{n_boc:05d}.png", rec)
                    n_boc += 1
                elif mar < MAR_NORMAL:
                    cv2.imwrite(f"{PASTA}/normal/{n_norm:05d}.png", rec)
                    n_norm += 1

            cv2.putText(frame, f"bocejando: {n_boc}   normal: {n_norm}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Boceje (boca aberta), depois fique normal. ESC sai.",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Coleta de bocejo (anotacao automatica)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nColetado: {n_boc} bocejando + {n_norm} normal em '{PASTA}/'")


if __name__ == "__main__":
    main()
