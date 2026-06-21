"""
Comparacao de arquiteturas para classificar o estado do olho (aberto/fechado),
no MESMO dataset e split (seed 42):

  - CNN propria      (do zero)
  - ResNet18         (transfer learning - pre-treinada ImageNet)
  - MobileNetV3      (transfer learning)
  - EfficientNet-B0  (transfer learning)
  - ViT pequeno      (do zero)

Metricas (material direto para o artigo): acuracia, precisao, recall e F1
(macro), tempo de inferencia por imagem, numero de parametros e tamanho (MB).
Salva comparacao_arquiteturas.txt e comparacao_arquiteturas.csv.

Rode:  python comparar_arquiteturas.py   (leva alguns minutos na CPU)
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from modelos import CNN
import metricas

PASTA = "dataset"
EPOCAS = 6
BATCH = 64
LR = 1e-3
DEVICE = "cpu"
TAM = 64

# Entrada em tons de cinza (CNN propria) e RGB normalizado (modelos pre-treinados)
tf_cinza = transforms.Compose([
    transforms.Grayscale(), transforms.Resize((TAM, TAM)), transforms.ToTensor()])
tf_rgb = transforms.Compose([
    transforms.Resize((TAM, TAM)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def loaders(tf):
    ds = datasets.ImageFolder(PASTA, transform=tf)
    n_val = max(1, int(0.2 * len(ds)))
    n_tr = len(ds) - n_val
    tr, val = random_split(ds, [n_tr, n_val],
                           generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr, batch_size=BATCH, shuffle=True),
            DataLoader(val, batch_size=BATCH), ds.classes, len(ds))


# ---------- construtores de cada arquitetura ----------
def build_cnn():
    return CNN(), tf_cinza, 1


def build_resnet():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m, tf_rgb, 3


def build_mobilenet():
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    return m, tf_rgb, 3


def build_efficientnet():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m, tf_rgb, 3


def build_vit():
    m = models.VisionTransformer(image_size=TAM, patch_size=8, num_layers=4,
                                 num_heads=4, hidden_dim=192, mlp_dim=384,
                                 num_classes=2)
    return m, tf_rgb, 3


ARQS = [
    ("CNN propria", build_cnn),
    ("ResNet18", build_resnet),
    ("MobileNetV3", build_mobilenet),
    ("EfficientNet-B0", build_efficientnet),
    ("ViT pequeno", build_vit),
]


def treinar(m, tr):
    m.to(DEVICE)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(EPOCAS):
        m.train()
        soma = 0.0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(m(x), y)
            loss.backward()
            opt.step()
            soma += loss.item()
        print(f"    epoca {ep+1}/{EPOCAS}  loss={soma/len(tr):.3f}")
    return m


def inferencia_ms(m, canais, n=80):
    m.eval()
    x = torch.randn(1, canais, TAM, TAM)
    with torch.no_grad():
        for _ in range(8):
            m(x)
        t0 = time.time()
        for _ in range(n):
            m(x)
    return (time.time() - t0) / n * 1000


def tamanho_mb(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6


def macro(conf):
    pc = metricas.por_classe(conf)
    prec = sum(p for p, _, _ in pc) / len(pc)
    rec = sum(r for _, r, _ in pc) / len(pc)
    f1 = sum(f for _, _, f in pc) / len(pc)
    return prec, rec, f1


def main():
    resultados = []
    for nome, build in ARQS:
        print(f"\n=== {nome} ===")
        try:
            m, tf, canais = build()
            tr, val, classes, n = loaders(tf)
            t0 = time.time()
            treinar(m, tr)
            conf = metricas.matriz_confusao(m, val, len(classes), DEVICE)
            acc = metricas.acuracia(conf)
            prec, rec, f1 = macro(conf)
            ms = inferencia_ms(m, canais)
            params = sum(p.numel() for p in m.parameters())
            mb = tamanho_mb(m)
            print(f"  acc={acc*100:.1f}%  F1={f1*100:.1f}%  {ms:.2f}ms  "
                  f"{params:,} params  {mb:.1f}MB  ({time.time()-t0:.0f}s)")
            resultados.append((nome, acc, prec, rec, f1, ms, params, mb))
        except Exception as ex:
            print(f"  [!] falhou: {ex}")

    # ---------- tabela ----------
    cab = (f"{'Arquitetura':>16}  {'Acuracia':>8}  {'Precisao':>8}  "
           f"{'Recall':>7}  {'F1':>6}  {'Tempo/img':>9}  {'Parametros':>11}  "
           f"{'Tamanho':>8}")
    linhas = ["=== Comparacao de Arquiteturas (olho aberto/fechado) ===",
              f"Dataset: {n} imagens | epocas: {EPOCAS} | entrada: {TAM}x{TAM}",
              "", cab, "-" * len(cab)]
    for nome, acc, prec, rec, f1, ms, params, mb in resultados:
        linhas.append(f"{nome:>16}  {acc*100:7.1f}%  {prec*100:7.1f}%  "
                      f"{rec*100:6.1f}%  {f1*100:5.1f}%  {ms:8.2f}ms  "
                      f"{params:>11,}  {mb:6.1f}MB")
    texto = "\n".join(linhas)
    with open("comparacao_arquiteturas.txt", "w") as f:
        f.write(texto + "\n")
    with open("comparacao_arquiteturas.csv", "w") as f:
        f.write("Arquitetura,Acuracia(%),Precisao(%),Recall(%),F1(%),"
                "Tempo_ms,Parametros,Tamanho_MB\n")
        for nome, acc, prec, rec, f1, ms, params, mb in resultados:
            f.write(f"{nome},{acc*100:.1f},{prec*100:.1f},{rec*100:.1f},"
                    f"{f1*100:.1f},{ms:.2f},{params},{mb:.1f}\n")
    print("\n" + texto)
    print("\nSalvo em comparacao_arquiteturas.txt e .csv")


if __name__ == "__main__":
    main()
