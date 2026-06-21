"""
Metricas de classificacao compartilhadas pelos treinos das CNNs (olho e boca):
acuracia, precisao, recall, F1 por classe e matriz de confusao.
Material direto para o artigo.
"""

import torch


def matriz_confusao(modelo, loader, n_classes, device="cpu"):
    """Conta acertos/erros: conf[verdade, previsto]."""
    modelo.eval()
    conf = torch.zeros(n_classes, n_classes, dtype=torch.int)
    with torch.no_grad():
        for x, y in loader:
            p = modelo(x.to(device)).argmax(1).cpu()
            for t, pr in zip(y, p):
                conf[int(t), int(pr)] += 1
    return conf


def por_classe(conf):
    """Lista de (precisao, recall, f1) para cada classe."""
    out = []
    for i in range(conf.size(0)):
        tp = conf[i, i].item()
        fp = conf[:, i].sum().item() - tp
        fn = conf[i, :].sum().item() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append((prec, rec, f1))
    return out


def acuracia(conf):
    return conf.diag().sum().item() / max(1, conf.sum().item())


def relatorio(titulo, classes, conf, params, ms, counts):
    """Monta um texto formatado pronto para o artigo."""
    L = []
    L.append(f"=== {titulo} ===")
    dist = ", ".join(f"{c}: {counts[i]}" for i, c in enumerate(classes))
    L.append(f"Dataset: {sum(counts)} imagens ({dist})")
    L.append(f"Acuracia geral: {acuracia(conf) * 100:.1f}%")
    L.append(f"Parametros: {params:,}")
    L.append(f"Tempo/imagem: {ms:.2f} ms")
    L.append("")
    L.append("Metricas por classe:")
    L.append(f"{'classe':>14}  {'precisao':>9}  {'recall':>7}  {'F1':>6}")
    for c, (p, r, f) in zip(classes, por_classe(conf)):
        L.append(f"{c:>14}  {p * 100:8.1f}%  {r * 100:6.1f}%  {f * 100:5.1f}%")
    L.append("")
    L.append("Matriz de confusao (linha = verdade, coluna = previsto):")
    L.append(" " * 16 + "  ".join(f"{c:>12}" for c in classes))
    for i, c in enumerate(classes):
        vals = "  ".join(f"{conf[i, j].item():>12}" for j in range(len(classes)))
        L.append(f"{c:>14}  {vals}")
    return "\n".join(L)


def salvar(caminho, titulo, classes, conf, params, ms, counts):
    texto = relatorio(titulo, classes, conf, params, ms, counts)
    with open(caminho, "w") as f:
        f.write(texto + "\n")
    return texto
