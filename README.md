# Driver Attention Monitor (DMS)

Um **sistema de monitoramento do motorista** em tempo real que estima uma
**Pontuação de Atenção do Motorista (0–100)** a partir de uma webcam comum,
combinando visão computacional clássica com deep learning. É o projeto final da
disciplina de **Visão Computacional** (IFMG – Campus Formiga).

> O sistema não mede apenas fadiga — ele mede **atenção**, fundindo estado dos
> olhos, bocejo, microssono, pose da cabeça, direção do olhar e objetos na mão
> em uma única pontuação.

## Destaques

- 🧠 **Duas CNNs treinadas do zero** — estado do olho (aberto/fechado, **96%**)
  e bocejo (**98%**), cada uma com métricas completas (precisão / recall / F1).
- 🔬 **Comparação de arquiteturas** — a CNN própria vs. ResNet18, MobileNetV3,
  EfficientNet-B0 e um ViT pequeno (transfer learning), comparando acurácia,
  velocidade e tamanho.
- 🎯 **Fusão de decisões** — a CNN informa uma confiança; abaixo de 70% ela cai
  no EAR geométrico como apoio.
- ⚖️ **Ensemble** — CNN + EAR + PERCLOS combinados com pesos em um único sinal
  de sonolência (uma piscada normal **não** baixa a pontuação; uma piscada
  demorada/sonolenta sim).
- 🔥 **Grad-CAM** — um mapa de calor que prova que a CNN olha para a pálpebra,
  e não para o fundo (`gradcam.py`).
- 📊 **Pontuação de Atenção** com faixas: Excelente → Bom → Distraído → Alerta
  → Crítico.
- 🚨 **Alertas inteligentes** seguindo a **regra das 2 segundos da NHTSA**: uma
  olhada para fora da pista de até 2 s é aceitável; acima disso é sinalizada.
  Alertas por voz ("olhe para a estrada", "olhe para a frente", "pare e
  descanse").
- 📼 **Caixa-preta** (CSV de eventos + clipes de vídeo) e uma tela de
  **resumo da viagem**.

## Como funciona

| Sinal | Método |
|---|---|
| Olho aberto/fechado | **CNN** (deep learning) com fallback no EAR |
| Bocejo | **CNN** da boca (comparada com o MAR) |
| Microssono | olhos fechados por mais de 2 s |
| PERCLOS | % do tempo com olhos fechados (métrica padrão-ouro de fadiga) |
| Pose da cabeça | yaw / pitch via `solvePnP` |
| Direção do olhar | posição da íris (olhar para baixo = celular) |
| Objeto na mão | EfficientDet (celular, livro, copo, garrafa, controle) |
| Gesto de sair | mão aberta |

## Como executar

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

python detector_dms.py              # sistema completo (ao vivo)
python gradcam.py                   # demo do Grad-CAM (tecla 's' salva a figura)
```

### Treinar os modelos (opcional — os pesos treinados já estão incluídos)

```bash
python coletar_dados.py             # coleta o dataset de olhos (rótulo automático por EAR)
python treino.py                    # treina a CNN do olho -> cnn.pt

python coletar_boca.py              # coleta o dataset da boca (rótulo automático por MAR)
python treino_boca.py               # treina a CNN de bocejo -> cnn_bocejo.pt

python comparar_arquiteturas.py     # CNN vs ResNet/MobileNet/EfficientNet/ViT
```

## Resultados

| Modelo | Acurácia | Inferência | Tamanho |
|---|---|---|---|
| **CNN própria** | **96,0%** | **0,46 ms** | **1,1 MB** |
| ViT (pequeno) | 96,2% | 1,68 ms | 5,0 MB |
| ResNet18 | 94,6% | 5,37 ms | 44,7 MB |
| EfficientNet-B0 | 91,5% | 219 ms | 16,0 MB |
| MobileNetV3 | 85,8% | 81 ms | 6,1 MB |

A CNN própria, pequena, empata em acurácia com os modelos pré-treinados muito
maiores, sendo **~40× menor** e **muito mais rápida** — ideal para monitoramento
em tempo real, embarcado. Números completos em `comparacao_arquiteturas.txt`,
`resultados.txt` e `resultados_boca.txt`.

## Base científica

- **Regra das 2 segundos** — olhadas para fora da pista acima de 2 s praticamente
  dobram o risco de acidente (NHTSA / estudo VTTI das 100 viagens).
- **PERCLOS** — olhos fechados por mais de 15% do tempo indicam sonolência
  (padrão-ouro da NHTSA).

## Autora

Luisa Caetano Araújo — Visão Computacional, IFMG Campus Formiga
(Prof. Me. Fernando Paim Lima).
