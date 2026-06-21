"""
Treina a CNN no dataset de olhos (aberto/fechado) e avalia a acuracia.

Pre-requisito: rodar 'python coletar_dados.py' antes, para gerar dataset/.
Rode:  python treino.py
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from modelos import CNN
import metricas

PASTA = "dataset"
EPOCAS = 12
BATCH = 64
LR = 1e-3
DEVICE = "cpu"   # CPU e suficiente (modelos pequenos) e 100% estavel


def carregar_dados():
    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(PASTA, transform=tf)
    n_val = max(1, int(0.2 * len(ds)))
    n_tr = len(ds) - n_val
    tr, val = random_split(ds, [n_tr, n_val],
                           generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr, batch_size=BATCH, shuffle=True),
            DataLoader(val, batch_size=BATCH), ds)


def avaliar(modelo, val):
    modelo.eval()
    certos = total = 0
    with torch.no_grad():
        for x, y in val:
            x, y = x.to(DEVICE), y.to(DEVICE)
            certos += (modelo(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return certos / total


def treinar(modelo, tr, val, nome):
    modelo.to(DEVICE)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(EPOCAS):
        modelo.train()
        soma = 0.0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(modelo(x), y)
            loss.backward()
            opt.step()
            soma += loss.item()
        acc = avaliar(modelo, val)
        print(f"  [{nome}] epoca {ep+1:>2}/{EPOCAS}  loss={soma/len(tr):.3f}  val_acc={acc*100:.1f}%")
    return acc


def velocidade_ms(modelo, n=200):
    """Tempo medio de inferencia por imagem (ms)."""
    modelo.eval()
    x = torch.randn(1, 1, 64, 64).to(DEVICE)
    with torch.no_grad():
        for _ in range(10):
            modelo(x)                      # aquecimento
        t0 = time.time()
        for _ in range(n):
            modelo(x)
        return (time.time() - t0) / n * 1000


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    # precisa ter coletado dados antes
    for c in ("aberto", "fechado"):
        p = os.path.join(PASTA, c)
        if not os.path.isdir(p) or len(os.listdir(p)) < 10:
            print(f"[!] Poucas imagens em '{p}'. "
                  f"Rode 'python coletar_dados.py' primeiro.")
            return

    tr, val, ds = carregar_dados()
    classes = ds.classes
    counts = [ds.targets.count(i) for i in range(len(classes))]
    print(f"Dataset: {len(ds)} imagens | classes: {classes}\n")

    print("=== Treinando CNN ===")
    modelo = CNN()
    treinar(modelo, tr, val, "CNN")
    ms = velocidade_ms(modelo)
    params = n_params(modelo)
    torch.save(modelo.state_dict(), "cnn.pt")

    conf = metricas.matriz_confusao(modelo, val, len(classes), DEVICE)
    texto = metricas.salvar("resultados.txt", "CNN de Olho (aberto/fechado)",
                            classes, conf, params, ms, counts)
    print("\n" + texto)
    print("\nModelo salvo (cnn.pt) e metricas em resultados.txt")


if __name__ == "__main__":
    main()
