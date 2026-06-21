"""
DMS - Driver Monitoring System (projeto final, autossuficiente).

Recursos:
  - CNN (deep learning) classifica o olho aberto/fechado (cai no EAR se nao houver cnn.pt)
  - PERCLOS / bocejo (MAR)
  - Pose da cabeca (yaw/pitch via solvePnP) -> celular (cabeca baixa) e retrovisor (virar)
  - Indice de fadiga ponderado com niveis (Normal/Atencao/Alerta/Critico)
  - Microssono: olhos fechados por mais de 2s -> CRITICO imediato
  - Calibracao automatica do EAR por motorista
  - Alerta escalonado: som leve -> som forte -> voz "Pare e descanse"
  - Caixa-preta: registra eventos em CSV e salva um clipe do video

Pre-requisitos: face_landmarker.task e gesture_recognizer.task nesta pasta.
Rode:  python detector_dms.py   (mao aberta ou ESC para sair)
"""

import os
import time
import math
import platform
import subprocess
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# A CNN (deep learning) classifica o estado do olho; e opcional para o detector
# continuar funcionando (cai no EAR) caso o modelo ainda nao tenha sido treinado.
try:
    import torch
    from modelos import CNN
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

# Pillow renderiza texto com acentos e fonte bonita (o OpenCV nao faz acento).
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

_FONTES = {}
_CAMINHOS_FONTE = [
    "/System/Library/Fonts/SFNSRounded.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _fonte(tam):
    f = _FONTES.get(tam)
    if f is None:
        for caminho in _CAMINHOS_FONTE:
            try:
                f = ImageFont.truetype(caminho, tam)
                break
            except Exception:
                continue
        _FONTES[tam] = f
    return f


def desenhar_textos(frame, itens):
    """itens: lista de (texto, x, y_topo, cor_bgr, tamanho_px[, stroke]). Usa
    Pillow se disponivel (com acento); senao cai no OpenCV sem acento."""
    if not itens:
        return frame
    if not _PIL_OK or _fonte(20) is None:
        for it in itens:
            texto, x, y, cor, tam = it[0], it[1], it[2], it[3], it[4]
            cv2.putText(frame, texto, (x, y + tam), cv2.FONT_HERSHEY_SIMPLEX,
                        tam / 32.0, cor, 2)
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for it in itens:
        texto, x, y, cor, tam = it[0], it[1], it[2], it[3], it[4]
        sw = it[5] if len(it) > 5 else 0   # contorno escuro (0 = sem)
        d.text((x, y), texto, font=_fonte(tam), fill=(cor[2], cor[1], cor[0]),
               stroke_width=sw, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def larg_texto(texto, tam):
    """Largura do texto em pixels na fonte do tamanho dado."""
    f = _fonte(tam)
    if f is None:
        return int(len(texto) * tam * 0.55)
    return int(f.getlength(texto))


# ============================================================
# Paleta "instrument cluster" (BGR)
# ============================================================
CARBON = (18, 14, 11)       # #0B0E12  fundo do painel
SLATE = (166, 151, 138)     # #8A97A6  rotulos discretos
MIST = (244, 238, 232)      # #E8EEF4  valores
TRILHO = (54, 46, 40)       # fundo do gauge
VERDE = (160, 224, 55)      # #37E0A0  NORMAL
AMARELO = (46, 176, 255)    # #FFB02E  ATENCAO
LARANJA = (47, 122, 255)    # #FF7A2F  ALERTA
VERMELHO = (59, 59, 255)    # #FF3B3B  CRITICO


NIVEL_LABEL = {"NORMAL": "NORMAL", "ATENCAO": "ATENÇÃO",
               "ALERTA": "ALERTA", "CRITICO": "CRÍTICO"}

# Driver Attention Score: 100 = atencao maxima, 0 = critico. Faixas (ordem do
# pior para o melhor usada para comparar severidade).
ORDEM_BANDA = ["EXCELENTE", "BOM", "DISTRAIDO", "ALERTA", "CRITICO"]
BANDA_LABEL = {"EXCELENTE": "EXCELENTE", "BOM": "BOM", "DISTRAIDO": "DISTRAÍDO",
               "ALERTA": "ALERTA", "CRITICO": "CRÍTICO"}
NIVEL_DA_BANDA = {"EXCELENTE": "NORMAL", "BOM": "NORMAL", "DISTRAIDO": "ATENCAO",
                  "ALERTA": "ALERTA", "CRITICO": "CRITICO"}
COR_BANDA = {"EXCELENTE": VERDE, "BOM": VERDE, "DISTRAIDO": AMARELO,
             "ALERTA": LARANJA, "CRITICO": VERMELHO}


def _rrect(img, p1, p2, r, cor, thickness=-1):
    """Retangulo de cantos arredondados (preenchido se thickness<0)."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), cor, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), cor, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                       (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, cor, -1)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, thickness)


def painel(frame, x1, y1, x2, y2, alpha=0.62, r=18, cor=CARBON):
    """Painel translucido de cantos arredondados (contraste para o texto)."""
    over = frame.copy()
    _rrect(over, (x1, y1), (x2, y2), r, cor, -1)
    cv2.addWeighted(over, alpha, frame, 1 - alpha, 0, frame)


def gauge(frame, x1, y, x2, h, frac, cor):
    """Barra horizontal: trilho escuro + preenchimento colorido (0..1)."""
    r = h // 2
    _rrect(frame, (x1, y), (x2, y + h), r, TRILHO, -1)
    fx = x1 + int((x2 - x1) * max(0.03, min(1.0, frac)))
    _rrect(frame, (x1, y), (fx, y + h), r, cor, -1)


def formatar_tempo(seg):
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    if h > 0:
        return f"{h}h{m:02d}"
    if m > 0:
        return f"{m}min{s:02d}"
    return f"{s}s"


def tela_resumo(d):
    """Monta a tela de Estatisticas da Viagem (mostrada ao encerrar)."""
    W, H = 900, 640
    img = np.full((H, W, 3), 16, np.uint8)
    cor_at = (VERDE if d["atencao_min"] >= 60 else
              AMARELO if d["atencao_min"] >= 40 else
              LARANJA if d["atencao_min"] >= 15 else VERMELHO)
    cv2.rectangle(img, (0, 0), (W, 6), VERDE, -1)
    textos = [("ESTATÍSTICAS DA VIAGEM", 56, 40, MIST, 40),
              ("resumo da sessão de monitoramento", 58, 96, SLATE, 22)]
    linhas = [
        ("Tempo dirigindo", d["tempo"], MIST),
        ("Bocejos", str(d["bocejos"]), MIST),
        ("Microssonos", str(d["micro"]), VERMELHO if d["micro"] else MIST),
        ("PERCLOS médio", f"{d['perclos']*100:.0f}%", MIST),
        ("Eventos de celular/objeto", str(d["celular"]),
         LARANJA if d["celular"] else MIST),
        ("Piscadas", str(d["piscadas"]), MIST),
        ("Menor atenção (pior momento)", f"{d['atencao_min']:.0f}/100", cor_at),
    ]
    y = 168
    for rot, val, cval in linhas:
        cv2.line(img, (56, y + 46), (W - 56, y + 46), (44, 40, 34), 1)
        textos.append((rot, 58, y, SLATE, 27))
        textos.append((val, W - 58 - larg_texto(val, 32), y - 3, cval, 32))
        y += 66
    textos.append(("pressione qualquer tecla para fechar", 58, H - 42, SLATE, 20))
    return desenhar_textos(img, textos)

# ============================================================
# Funcoes auxiliares (visao)
# ============================================================
OLHO_DIR = [33, 160, 158, 133, 153, 144]
OLHO_ESQ = [362, 385, 387, 263, 373, 380]
BOCA_VERT = [13, 14]
BOCA_HORIZ = [78, 308]
BOCA_BOX = [61, 291, 0, 17, 13, 14, 78, 308]  # cantos + topo/base dos labios
TAM_OLHO = 64        # recorte 64x64 (mesmo tamanho usado no treino da CNN)
MARGEM_OLHO = 0.6    # margem ao redor do olho (igual ao coletar_dados.py)
MARGEM_BOCA = 0.35   # margem ao redor da boca
IDX_FECHADO = 1      # classes ImageFolder em ordem: 0=aberto, 1=fechado
IDX_BOCEJO = 0       # classes ImageFolder em ordem: 0=bocejando, 1=normal


def recortar_olho(frame, pts):
    """Recorta a regiao do olho (64x64) a partir dos 6 pontos do EAR."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    mx = (x2 - x1) * MARGEM_OLHO + 5
    my = (y2 - y1) * MARGEM_OLHO + 5
    h, w = frame.shape[:2]
    x1 = max(0, int(x1 - mx)); x2 = min(w, int(x2 + mx))
    y1 = max(0, int(y1 - my)); y2 = min(h, int(y2 + my))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], (TAM_OLHO, TAM_OLHO))


def prob_fechado_cnn(modelo, crops):
    """Roda a CNN nos recortes dos olhos e devolve a prob. media de 'fechado'."""
    crops = [c for c in crops if c is not None]
    if not crops:
        return None
    lote = []
    for c in crops:
        cinza = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)          # 1 canal
        t = torch.from_numpy(cinza).float().div(255.0).unsqueeze(0)  # 1x64x64
        lote.append(t)
    with torch.no_grad():
        logits = modelo(torch.stack(lote))                  # N x 2
        probs = torch.softmax(logits, dim=1)[:, IDX_FECHADO]
    return float(probs.mean())


def recortar_boca(frame, pts):
    """Recorta a regiao da boca (64x64) a partir dos pontos dos labios."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    mx = (x2 - x1) * MARGEM_BOCA + 5
    my = (y2 - y1) * MARGEM_BOCA + 5
    h, w = frame.shape[:2]
    x1 = max(0, int(x1 - mx)); x2 = min(w, int(x2 + mx))
    y1 = max(0, int(y1 - my)); y2 = min(h, int(y2 + my))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], (TAM_OLHO, TAM_OLHO))


def prob_bocejo_cnn(modelo, crop):
    """Roda a CNN de bocejo no recorte da boca; prob. de 'bocejando'."""
    if crop is None:
        return None
    cinza = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    t = torch.from_numpy(cinza).float().div(255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(modelo(t), dim=1)[:, IDX_BOCEJO]
    return float(probs[0])


def distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calcular_ear(pts):
    vertical = distancia(pts[1], pts[5]) + distancia(pts[2], pts[4])
    horizontal = 2.0 * distancia(pts[0], pts[3])
    return vertical / horizontal


def calcular_mar(vert, horiz):
    return distancia(vert[0], vert[1]) / distancia(horiz[0], horiz[1])


def coords(landmarks, indices, w, h):
    return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]


def gaze_vertical(landmarks):
    """Posicao vertical media da iris dentro do olho (0=topo, 1=base).
    Acima de ~0.6 indica olhar dirigido para baixo. Precisa dos pontos de iris
    (478 landmarks); devolve None se nao houver."""
    if len(landmarks) < 478:
        return None

    def razao(iris, topo, base):
        ty, by, iy = landmarks[topo].y, landmarks[base].y, landmarks[iris].y
        d = by - ty
        return (iy - ty) / d if d > 1e-6 else 0.5

    # iris dir=468 (topo 159, base 145) | iris esq=473 (topo 386, base 374)
    return (razao(468, 159, 145) + razao(473, 386, 374)) / 2.0


def garantir_permissao_camera():
    """No macOS, pede a permissao da camera e espera o usuario responder."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        from Foundation import NSRunLoop, NSDate, NSDefaultRunLoopMode
    except Exception:
        return
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    if status == 3:
        return
    if status in (1, 2):
        print("\n[!] Camera bloqueada. Ative o Terminal em Ajustes do Sistema > "
              "Privacidade e Seguranca > Camera, de Cmd+Q e rode de novo.\n")
        return
    box = {}

    def handler(ok):
        box["ok"] = bool(ok)

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, handler)
    print("\n[i] Clique em OK no pop-up para liberar a camera...\n")
    loop = NSRunLoop.currentRunLoop()
    while "ok" not in box:
        loop.runMode_beforeDate_(NSDefaultRunLoopMode,
                                 NSDate.dateWithTimeIntervalSinceNow_(0.1))


def criar_landmarker():
    base = python.BaseOptions(model_asset_path="face_landmarker.task")
    opts = vision.FaceLandmarkerOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


def criar_reconhecedor_gestos():
    base = python.BaseOptions(model_asset_path="gesture_recognizer.task")
    opts = vision.GestureRecognizerOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=1)
    return vision.GestureRecognizer.create_from_options(opts)


# Objetos "de mao" do COCO que indicam descuido se a pessoa estiver segurando.
OBJETOS_MAO = ["cell phone", "book", "cup", "bottle", "remote", "wine glass"]
NOME_PT = {"cell phone": "celular", "book": "livro/papel", "cup": "copo",
           "bottle": "garrafa", "remote": "controle", "wine glass": "taca"}


def criar_detector_objetos():
    """Detector de objetos (EfficientDet) p/ objetos de mao do COCO."""
    base = python.BaseOptions(model_asset_path="efficientdet.tflite")
    opts = vision.ObjectDetectorOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO,
        score_threshold=0.30, category_allowlist=OBJETOS_MAO)
    return vision.ObjectDetector.create_from_options(opts)


# ============================================================
# Parametros do DMS (ajustaveis)
# ============================================================
JANELA_SEG = 15
MAR_LIMIAR = 0.50        # abertura da boca p/ contar bocejo (menor = mais sensivel)
FRAMES_BOCEJO = 12       # frames de boca aberta p/ confirmar 1 bocejo
MICROSSONO_SEG = 2.0
CONF_MIN = 0.70          # confianca minima da CNN; abaixo disso usa o EAR (fusao)
# Ensemble: combina CNN + EAR + PERCLOS (pesos somam 1) numa decisao de sonolencia
W_ENS_CNN = 0.4
W_ENS_EAR = 0.2
W_ENS_PERCLOS = 0.4
EAR_ABERTO = 0.26        # EAR tipico de olho aberto (normaliza o sinal do EAR)
EAR_FECHADO = 0.12       # EAR tipico de olho fechado
ENS_SUAVIZA = 0.15       # suavizacao (media movel) p/ a piscada nao dar pico
PISCADA_MAX_SEG = 0.4    # fechamento mais curto que isso conta como piscada
BOCEJOS_SONO = 3         # >= este nro de bocejos por minuto = sinal de sono
YAW_LIMIAR = 22
# Regra NHTSA/VTTI: olhada fora da pista de ate 2s e aceitavel (velocimetro,
# retrovisor, radio); acima de 2s o risco dobra -> imprudencia -> alerta.
TEMPO_GLANCE = 2.0       # seg. de olhada fora da pista ate virar imprudencia
TOLERANCIA_ROSTO = 2.0   # seg. sem achar o rosto (virou de costas) ate alertar
PITCH_LIMIAR = 12        # cabeca inclinada para baixo (instantaneo; o tempo e o filtro)
GAZE_BAIXO_LIMIAR = 0.60  # iris abaixo do centro do olho = olhando p/ baixo
EAR_OLHANDO_MIN = 0.13   # acima disso o olho ainda esta entreaberto (olhando, nao dormindo)
N_CALIBRACAO = 40

MODELO_3D = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
IDX_POSE = [1, 152, 263, 33, 291, 61]

DUR_AVISO = 3.0          # segundos que cada aviso fica na tela

_alerta = {"proc": None}


def pose_cabeca(landmarks, w, h):
    pts2d = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in IDX_POSE],
                     dtype=np.float64)
    foco = float(w)
    cam = np.array([[foco, 0, w / 2.0], [0, foco, h / 2.0], [0, 0, 1]],
                   dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODELO_3D, pts2d, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-R[2, 0], sy))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    return pitch, yaw, roll


def acionar_alerta(tipo):
    """tipo: 'fadiga' (voz), 'celular' (voz), 'forte' (som), 'leve' (toque) ou None."""
    if tipo is None:
        return
    if _alerta["proc"] is not None and _alerta["proc"].poll() is None:
        return
    if platform.system() != "Darwin":
        print("\a", end="", flush=True)
        return
    if tipo == "fadiga":
        _alerta["proc"] = subprocess.Popen(
            ["say", "-v", "Luciana", "-r", "230", "Pare e descanse, motorista"])
    elif tipo == "celular":
        _alerta["proc"] = subprocess.Popen(
            ["say", "-v", "Luciana", "-r", "230", "Olhe para a estrada, motorista"])
    elif tipo == "frente":
        _alerta["proc"] = subprocess.Popen(
            ["say", "-v", "Luciana", "-r", "230", "Olhe para a frente, motorista"])
    elif tipo == "forte":
        _alerta["proc"] = subprocess.Popen(
            ["afplay", "/System/Library/Sounds/Sosumi.aiff"])
    else:  # leve / toquinho
        _alerta["proc"] = subprocess.Popen(
            ["afplay", "/System/Library/Sounds/Tink.aiff"])


def registrar_evento(path, banda, atencao, perclos):
    with open(path, "a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S},{banda},"
                f"{atencao:.0f},{perclos:.2f}\n")


def salvar_clipe(buffer, w, h):
    nome = datetime.now().strftime("eventos/evento_%Y%m%d_%H%M%S.mp4")
    vw = cv2.VideoWriter(nome, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for f in list(buffer):
        vw.write(f)
    vw.release()
    return nome


def main():
    garantir_permissao_camera()
    os.makedirs("eventos", exist_ok=True)
    csv_path = "eventos/eventos.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("data_hora,banda,atencao,perclos\n")

    landmarker = criar_landmarker()
    reconhecedor = criar_reconhecedor_gestos()
    detector_obj = criar_detector_objetos()

    # Carrega a CNN treinada (deep learning) para classificar olho aberto/fechado.
    # Se ainda nao houver cnn.pt, o detector cai no EAR geometrico.
    modelo_cnn = None
    if _TORCH_OK and os.path.exists("cnn.pt"):
        modelo_cnn = CNN()
        modelo_cnn.load_state_dict(torch.load("cnn.pt", map_location="cpu"))
        modelo_cnn.eval()
        print("[i] CNN carregada (cnn.pt) — classificando o olho com a rede.")
    else:
        print("[i] Sem cnn.pt — usando EAR. Treine a CNN (coletar_dados.py + "
              "treino.py) para o sistema usar a rede.")

    # CNN de bocejo (opcional): classifica a boca (bocejando/normal). Sem ela,
    # o detector usa o MAR geometrico.
    modelo_bocejo = None
    if _TORCH_OK and os.path.exists("cnn_bocejo.pt"):
        modelo_bocejo = CNN()
        modelo_bocejo.load_state_dict(torch.load("cnn_bocejo.pt",
                                                 map_location="cpu"))
        modelo_bocejo.eval()
        print("[i] CNN de bocejo carregada (cnn_bocejo.pt).")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    historico = deque()
    buffer_frames = deque(maxlen=150)
    bocejos_recentes = deque()
    ear_calib = []
    ear_limiar = 0.21
    calibrado = False
    contador_bocejo = 0
    total_bocejos = 0
    total_piscadas = 0
    fechado_anterior = False   # estado do olho no frame anterior (p/ piscada)
    fechou_em = 0.0            # instante em que o olho fechou
    lado_desde = None          # instante em que comecou a olhar para o lado
    baixo_desde = None         # instante em que comecou a olhar para baixo/celular
    rosto_perdido_desde = None  # instante em que o rosto sumiu (virou de costas)
    ens_ema = 0.0              # ensemble suavizado (media movel)
    # Estatisticas da viagem (mostradas ao encerrar)
    inicio = None
    micro_anterior = False
    celular_anterior = False
    total_micro = 0
    total_celular_ev = 0
    perclos_soma = 0.0
    perclos_n = 0
    atencao_min = 100.0
    frames_mao_aberta = 0
    grave_anterior = False
    avisos = {}            # tipo -> {"t": texto, "c": cor, "ate": timestamp}
    celular_visto_em = 0.0  # ultimo instante em que um objeto de mao foi detectado
    caixa_celular = None    # bounding box do objeto para desenhar
    objeto_nome = ""        # nome (PT) do objeto detectado na mao
    frame_id = 0
    FRAMES_PARA_SAIR = 35

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        buffer_frames.append(frame.copy())
        textos = []          # (texto, x, y_topo, cor_bgr, tamanho_px)
        # escala do HUD conforme a resolucao real (ancorada em 1500px ~ 30% da tela)
        S = w / 1500.0
        e = lambda v: max(1, int(round(v * S)))  # noqa: E731
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_id += 1
        res = landmarker.detect_for_video(mp_image, frame_id * 33)
        agora = time.time()
        if inicio is None:
            inicio = agora

        # Detecta objeto de mao na imagem (a cada 3 frames p/ desempenho).
        # So conta se o objeto for grande o bastante (perto/na mao, nao no fundo).
        if frame_id % 3 == 0:
            det_obj = detector_obj.detect_for_video(mp_image, frame_id * 33)
            caixa_celular = None
            for det in det_obj.detections:
                bb = det.bounding_box
                if bb.width > 0.10 * w or bb.height > 0.14 * h:
                    caixa_celular = (bb.origin_x, bb.origin_y, bb.width, bb.height)
                    objeto_nome = NOME_PT.get(det.categories[0].category_name,
                                              det.categories[0].category_name)
                    celular_visto_em = agora
                    break
        celular_visivel = (agora - celular_visto_em) < 1.0

        nivel = "NORMAL"
        if res.face_landmarks:
            rosto_perdido_desde = None        # rosto voltou
            lm = res.face_landmarks[0]
            ear = (calcular_ear(coords(lm, OLHO_DIR, w, h)) +
                   calcular_ear(coords(lm, OLHO_ESQ, w, h))) / 2.0
            mar = calcular_mar(coords(lm, BOCA_VERT, w, h),
                               coords(lm, BOCA_HORIZ, w, h))

            if modelo_cnn is None and not calibrado:
                # Calibracao do EAR so e necessaria no modo geometrico (sem CNN)
                ear_calib.append(ear)
                painel(frame, e(24), e(24), e(24) + e(410), e(24) + e(80))
                _rrect(frame, (e(24), e(30)), (e(32), e(96)), e(4), AMARELO, -1)
                textos.append(("CALIBRANDO", e(52), e(36), SLATE, e(15)))
                textos.append(("Mantenha os olhos abertos", e(52), e(56),
                               MIST, e(22)))
                if len(ear_calib) >= N_CALIBRACAO:
                    base = float(np.median(ear_calib))
                    ear_limiar = max(0.12, base * 0.72)
                    calibrado = True
            else:
                # 1) Pose da cabeca (cabeca baixa = celular; virar = retrovisor)
                pitch, yaw, roll = pose_cabeca(lm, w, h)
                cabeca_para_baixo = pitch > PITCH_LIMIAR

                # 2) Estado do olho: CNN (deep learning) + FUSAO com o EAR.
                #    A CNN decide; se a confianca dela for baixa (<CONF_MIN),
                #    cai no EAR como apoio (fusao de decisoes).
                prob_cnn = None
                conf_cnn = None
                usou_ear = False
                if modelo_cnn is not None:
                    prob_cnn = prob_fechado_cnn(modelo_cnn, [
                        recortar_olho(frame, coords(lm, OLHO_DIR, w, h)),
                        recortar_olho(frame, coords(lm, OLHO_ESQ, w, h))])
                if prob_cnn is not None:
                    conf_cnn = max(prob_cnn, 1 - prob_cnn)
                    if conf_cnn >= CONF_MIN:
                        olho_fechado = prob_cnn > 0.5
                    else:
                        olho_fechado = ear < ear_limiar   # CNN incerta -> EAR
                        usou_ear = True
                else:
                    olho_fechado = ear < ear_limiar

                # 3) Direcao do olhar (iris) apontando p/ baixo = celular.
                #    Vale mesmo com o olho ESTREITADO (EAR acima do piso): de
                #    cabeca/olhos baixos no celular o olho fica entreaberto;
                #    so quando fecha de vez (EAR muito baixo) e que vira sono.
                gv = gaze_vertical(lm)
                olhos_para_baixo = (gv is not None and gv > GAZE_BAIXO_LIMIAR
                                    and ear > EAR_OLHANDO_MIN)

                # 4) Distracao "olhos fora da pista" pela regra das 2 segundos.
                #    Olhar p/ BAIXO (cabeca baixa / olhos baixos / celular visivel):
                fora_baixo = (cabeca_para_baixo or celular_visivel
                              or olhos_para_baixo)
                if fora_baixo:
                    if baixo_desde is None:
                        baixo_desde = agora
                    dur_baixo = agora - baixo_desde
                else:
                    baixo_desde = None
                    dur_baixo = 0.0
                olhando_celular = dur_baixo >= TEMPO_GLANCE     # > 2s = imprudencia

                #    Olhar p/ o LADO (retrovisor curto ok; > 2s = olhe para a frente):
                if abs(yaw) > YAW_LIMIAR:
                    if lado_desde is None:
                        lado_desde = agora
                    dur_lado = agora - lado_desde
                else:
                    lado_desde = None
                    dur_lado = 0.0
                desvio_prolongado = (dur_lado >= TEMPO_GLANCE) and not olhando_celular

                # 5) Olho fechado so conta como sono se a pessoa NAO estiver com a
                #    cabeca/olhos baixos (de cabeca baixa o olho parece fechado)
                olho_fechado_sono = olho_fechado and not fora_baixo

                # Piscadas (fechamento curto) e microssono (fechamento longo)
                if olho_fechado_sono:
                    if not fechado_anterior:
                        fechou_em = agora
                    dur_fechado = agora - fechou_em
                else:
                    if fechado_anterior and (agora - fechou_em) < PISCADA_MAX_SEG:
                        total_piscadas += 1
                    dur_fechado = 0.0
                fechado_anterior = olho_fechado_sono
                microssono = dur_fechado >= MICROSSONO_SEG

                # Bocejo: CNN da boca (deep learning) se houver; senao MAR.
                prob_bocejo = None
                if modelo_bocejo is not None:
                    prob_bocejo = prob_bocejo_cnn(
                        modelo_bocejo, recortar_boca(frame,
                                                     coords(lm, BOCA_BOX, w, h)))
                bocejando = (prob_bocejo > 0.5) if prob_bocejo is not None \
                    else (mar > MAR_LIMIAR)
                if bocejando:
                    contador_bocejo += 1
                else:
                    if contador_bocejo >= FRAMES_BOCEJO:
                        bocejos_recentes.append(agora)
                        total_bocejos += 1
                        avisos["boc"] = {"t": "Bocejo detectado",
                                         "c": AMARELO, "ate": agora + DUR_AVISO}
                    contador_bocejo = 0
                while bocejos_recentes and agora - bocejos_recentes[0] > 60:
                    bocejos_recentes.popleft()
                if len(bocejos_recentes) >= BOCEJOS_SONO:
                    avisos["sono"] = {"t": "Muitos bocejos - sinal de sono!",
                                      "c": LARANJA, "ate": agora + DUR_AVISO}

                historico.append((agora, olho_fechado_sono))
                while historico and agora - historico[0][0] > JANELA_SEG:
                    historico.popleft()
                perclos = (sum(1 for _, c in historico if c) / len(historico)
                           if historico else 0.0)

                # Ensemble (fusao ponderada): CNN + EAR + PERCLOS -> sonolencia.
                # CNN e EAR so entram quando o olho fica fechado MAIS que uma
                # piscada (>=PISCADA_MAX_SEG): piscada normal NAO baixa a atencao;
                # piscada demorada/sonolenta sim. PERCLOS ja e temporal.
                fechado_lento = olho_fechado_sono and dur_fechado >= PISCADA_MAX_SEG
                s_cnn = (prob_cnn if prob_cnn is not None else 1.0) \
                    if fechado_lento else 0.0
                s_ear = (min(1.0, max(0.0, (EAR_ABERTO - ear) /
                         (EAR_ABERTO - EAR_FECHADO)))) if fechado_lento else 0.0
                s_perclos = perclos
                ensemble = (W_ENS_CNN * s_cnn + W_ENS_EAR * s_ear +
                            W_ENS_PERCLOS * s_perclos)
                ens_ema = (1 - ENS_SUAVIZA) * ens_ema + ENS_SUAVIZA * ensemble

                # ===== Driver Attention Score (0-100, 100 = atencao maxima) =====
                # Cada sinal de desatencao retira pontos da atencao plena.
                penal = min(55.0, ens_ema * 100)               # fadiga (CNN+EAR+PERCLOS)
                penal += min(20.0, len(bocejos_recentes) * 7.0)  # bocejos
                if olhando_celular:
                    penal += 50.0                              # > 2s olhando p/ baixo/celular
                if desvio_prolongado:
                    penal += 45.0                              # > 2s olhando p/ o lado
                if microssono:
                    penal = 100.0                              # microssono = critico
                atencao = max(0.0, 100.0 - penal)

                if atencao >= 85:
                    banda = "EXCELENTE"
                elif atencao >= 60:
                    banda = "BOM"
                elif atencao >= 40:
                    banda = "DISTRAIDO"
                elif atencao >= 15:
                    banda = "ALERTA"
                else:
                    banda = "CRITICO"
                # garante severidade visivel das causas agudas (distracoes > 2s)
                if (olhando_celular or desvio_prolongado) and \
                        ORDEM_BANDA.index(banda) < ORDEM_BANDA.index("ALERTA"):
                    banda = "ALERTA"
                if microssono:
                    banda = "CRITICO"

                nivel = NIVEL_DA_BANDA[banda]
                grave = microssono or olhando_celular or banda == "CRITICO"

                # Estatisticas da viagem
                if microssono and not micro_anterior:
                    total_micro += 1
                micro_anterior = microssono
                if olhando_celular and not celular_anterior:
                    total_celular_ev += 1
                celular_anterior = olhando_celular
                perclos_soma += perclos
                perclos_n += 1
                atencao_min = min(atencao_min, atencao)

                # Avisos persistentes (~3s na tela)
                if microssono:
                    avisos["micro"] = {"t": "MICROSSONO - olhos fechados!",
                                       "c": VERMELHO, "ate": agora + DUR_AVISO}
                if olhando_celular:
                    avisos["cel"] = {"t": "Olhe para a estrada!",
                                     "c": VERMELHO, "ate": agora + DUR_AVISO}
                if desvio_prolongado:
                    avisos["frente"] = {"t": "Olhe para a frente!",
                                        "c": LARANJA, "ate": agora + DUR_AVISO}

                # Som (prioridade: microssono/critico > celular > desvio > alerta)
                if microssono or banda == "CRITICO":
                    acionar_alerta("fadiga")
                elif olhando_celular:
                    acionar_alerta("celular")
                elif desvio_prolongado:
                    acionar_alerta("frente")
                elif banda == "ALERTA":
                    acionar_alerta("forte")

                # Caixa-preta: grava na transicao para um evento grave
                if grave and not grave_anterior:
                    registrar_evento(csv_path, banda, atencao, perclos)
                    salvar_clipe(buffer_frames, w, h)
                grave_anterior = grave

                # Cor do tema = cor da faixa de atencao (verde alto -> vermelho baixo)
                cor = COR_BANDA[banda]
                # ===== Cluster de instrumentos (escala com a resolucao) =====
                PH_BASE = 342 if modelo_bocejo is None else 374
                PX, PY, PW, PH = e(24), e(24), e(440), e(PH_BASE)
                painel(frame, PX, PY, PX + PW, PY + PH)
                _rrect(frame, (PX, PY + e(6)), (PX + e(8), PY + PH - e(6)),
                       e(4), cor, -1)
                X = PX + e(28)                      # margem interna de conteudo
                # eyebrow + farol de status
                textos.append(("ATENÇÃO DO MOTORISTA", X, PY + e(16), SLATE, e(16)))
                cv2.circle(frame, (PX + PW - e(28), PY + e(26)), e(7), cor, -1)
                # numero gigante (Driver Attention Score) + /100 + faixa
                fn = e(74)
                textos.append((f"{atencao:.0f}", X - e(2), PY + e(34), cor, fn))
                wnum = larg_texto(f"{atencao:.0f}", fn)
                textos.append(("/100", X + wnum + e(14), PY + e(44), SLATE, e(21)))
                textos.append((BANDA_LABEL[banda],
                               X + wnum + e(14), PY + e(74), cor, e(27)))
                # gauge = medidor de atencao (cheio/verde = otimo)
                gauge(frame, X, PY + e(134), PX + PW - e(26), e(15),
                      atencao / 100.0, cor)
                # divisor
                cv2.line(frame, (X, PY + e(166)), (PX + PW - e(26), PY + e(166)),
                         (70, 62, 54), 1)
                # telemetria (rotulo SLATE + valor) em 2 colunas
                # Estado do olho + confianca da CNN + fusao (linha cheia)
                if prob_cnn is not None:
                    estado = "FECHADO" if olho_fechado else "ABERTO"
                    olho_txt = f"{estado}  ·  conf {conf_cnn*100:.0f}%"
                    if usou_ear:
                        olho_txt += "  ·  EAR"
                    cor_olho = AMARELO if usou_ear else \
                        (VERMELHO if olho_fechado else VERDE)
                else:
                    olho_txt = "EAR (sem CNN)"
                    cor_olho = VERMELHO if olho_fechado else VERDE
                cab = f"yaw {yaw:.0f}  ·  pitch {pitch:.0f}"
                if gv is not None:
                    cab += f"  ·  olhar {gv:.2f}"
                colL, colR, voff = X, X + e(214), e(108)
                rl, rv = e(14), e(20)
                linhas = [
                    (colL, PY + e(182), "PERCLOS", f"{perclos*100:.0f}%", MIST),
                    (colR, PY + e(182), "PISCADAS", f"{total_piscadas}", MIST),
                    (colL, PY + e(214), "EAR", f"{ear:.2f}", MIST),
                    (colR, PY + e(214), "BOCEJOS", f"{total_bocejos}", MIST),
                ]
                for lx, ly, rot, val, cval in linhas:
                    textos.append((rot, lx, ly, SLATE, rl))
                    textos.append((val, lx + voff, ly - e(3), cval, rv))
                textos.append(("OLHO", colL, PY + e(246), SLATE, rl))
                textos.append((olho_txt, colL + voff, PY + e(243), cor_olho, e(18)))
                textos.append(("CABEÇA", colL, PY + e(278), SLATE, rl))
                textos.append((cab, colL + voff, PY + e(275), MIST, e(18)))
                # Ensemble: fusao ponderada CNN + EAR + PERCLOS
                ens_cor = VERMELHO if ens_ema > 0.6 else \
                    (AMARELO if ens_ema > 0.3 else VERDE)
                textos.append(("ENSEMBLE", colL, PY + e(308), SLATE, rl))
                textos.append((f"{ens_ema*100:.0f}%  ·  C{s_cnn*100:.0f} "
                               f"E{s_ear*100:.0f} P{s_perclos*100:.0f}",
                               colL + voff, PY + e(305), ens_cor, e(18)))
                if prob_bocejo is not None:    # comparacao MAR x CNN
                    bclass = "bocejando" if bocejando else "normal"
                    bcol = AMARELO if bocejando else MIST
                    textos.append(("BOCA", colL, PY + e(338), SLATE, rl))
                    textos.append((f"MAR {mar:.2f}  ·  CNN {bclass} {prob_bocejo:.2f}",
                                   colL + voff, PY + e(335), bcol, e(18)))
                if nivel != "NORMAL" or grave:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), cor, e(10))
        else:
            # Sem rosto: provavelmente virou de costas. Apos um tempo, alerta.
            if rosto_perdido_desde is None:
                rosto_perdido_desde = agora
            sem_rosto = agora - rosto_perdido_desde
            cor_sr = LARANJA if sem_rosto >= TOLERANCIA_ROSTO else VERMELHO
            painel(frame, e(24), e(24), e(24) + e(400), e(24) + e(58))
            _rrect(frame, (e(24), e(30)), (e(32), e(76)), e(4), cor_sr, -1)
            textos.append(("Rosto não detectado", e(52), e(36), cor_sr, e(24)))
            if sem_rosto >= TOLERANCIA_ROSTO:
                acionar_alerta("frente")
                avisos["frente"] = {"t": "Olhe para a frente!",
                                    "c": LARANJA, "ate": agora + DUR_AVISO}
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), LARANJA, e(10))

        # Caixa em volta do celular detectado
        if caixa_celular is not None:
            cx, cy, cw, ch = caixa_celular
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), VERMELHO, 3)
            textos.append((objeto_nome.upper(), cx, max(2, cy - e(28)),
                           VERMELHO, e(20), 2))

        # Avisos = farois sólidos empilhados a partir da base
        for k in [k for k, v in avisos.items() if v["ate"] < agora]:
            del avisos[k]
        py2 = h - e(24)
        for v in sorted(avisos.values(), key=lambda a: a["ate"]):
            ts = e(24)
            tw = larg_texto(v["t"], ts)
            px1, px2, py1 = e(24), e(24) + tw + e(82), py2 - e(50)
            painel(frame, px1, py1, px2, py2, alpha=0.82, r=e(14), cor=v["c"])
            cy_b = (py1 + py2) // 2
            cv2.circle(frame, (px1 + e(30), cy_b), e(15), MIST, -1)
            textos.append(("!", px1 + e(25), py1 + e(11), v["c"], e(26)))
            textos.append((v["t"], px1 + e(56), py1 + e(12), MIST, ts, 2))
            py2 = py1 - e(10)

        gesto = reconhecedor.recognize_for_video(mp_image, frame_id * 33)
        mao_aberta = bool(gesto.gestures) and \
            gesto.gestures[0][0].category_name == "Open_Palm"
        frames_mao_aberta = frames_mao_aberta + 1 if mao_aberta else 0
        if frames_mao_aberta > 0:
            pct = min(100, int(100 * frames_mao_aberta / FRAMES_PARA_SAIR))
            txt = f"Encerrando  {pct}%"
            ts = e(22)
            pw = larg_texto(txt, ts) + e(36)
            painel(frame, w - e(24) - pw, h - e(68), w - e(24), h - e(22),
                   alpha=0.6)
            gauge(frame, w - e(24) - pw + e(16), h - e(34), w - e(40), e(7),
                  pct / 100.0, VERDE)
            textos.append((txt, w - e(24) - pw + e(16), h - e(60), MIST, ts))

        frame = desenhar_textos(frame, textos)
        cv2.imshow("DMS - Monitoramento do Motorista", frame)
        if (cv2.waitKey(1) & 0xFF == 27) or frames_mao_aberta >= FRAMES_PARA_SAIR:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ===== Estatisticas da viagem =====
    if inicio is not None:
        dur = agora - inicio
        perclos_med = perclos_soma / perclos_n if perclos_n else 0.0
        stats = {
            "tempo": formatar_tempo(dur),
            "bocejos": total_bocejos,
            "micro": total_micro,
            "perclos": perclos_med,
            "celular": total_celular_ev,
            "piscadas": total_piscadas,
            "atencao_min": atencao_min,
        }
        with open("relatorio_viagem.txt", "a") as f:
            f.write(f"\n=== Viagem {datetime.now():%Y-%m-%d %H:%M} ===\n")
            f.write(f"Tempo dirigindo: {stats['tempo']}\n")
            f.write(f"Bocejos: {stats['bocejos']}\n")
            f.write(f"Microssonos: {stats['micro']}\n")
            f.write(f"PERCLOS medio: {perclos_med*100:.0f}%\n")
            f.write(f"Eventos de celular/objeto: {stats['celular']}\n")
            f.write(f"Piscadas: {stats['piscadas']}\n")
            f.write(f"Menor atencao: {atencao_min:.0f}/100\n")
        tela = tela_resumo(stats)
        cv2.imshow("Estatisticas da Viagem", tela)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Eventos em {csv_path} | relatorio em relatorio_viagem.txt")


if __name__ == "__main__":
    main()
