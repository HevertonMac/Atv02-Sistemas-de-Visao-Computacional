# Atividade Prática 02 — Pirâmides & Blending de Imagens

**Disciplina:** Sistemas de Visão Computacional  
**Autor:** Heverton Macedo — VipLab UFPI  
**Técnica:** Burt & Adelson (1983)

---

## Como executar

```bash
pip install pillow numpy
python atv02.py
```

As imagens padrão da atividade (`512_burt_apple.png`, `512_burt_orange.png`, `512_mask.jpg`)
são carregadas automaticamente ao abrir o programa, desde que estejam na mesma pasta.

---

## Funcionalidades implementadas

| Controle | Descrição |
|---|---|
| Carregar Imagem A / B | Abre qualquer PNG/JPG como imagem de entrada |
| Carregar Máscara | Imagem em tons de cinza: branco → usa A, preto → usa B |
| Níveis da pirâmide | Slider de 2 a 8 (padrão: 6) |
| ▶ Executar Blending | Aplica o algoritmo completo de Burt & Adelson |
| Resultado — Pirâmide | Exibe o resultado do blending multiresolução |
| Justaposição direta | Exibe o corte simples pela máscara (para comparação) |
| Comparação lado a lado | Ambos os resultados na mesma tela com rótulos |
| Salvar Resultado | Salva a visualização atual em PNG/JPEG |

---

## Matemática implementada

### Kernel Gaussiano separável (Burt & Adelson)

```
w = [1, 4, 6, 4, 1] / 16
K(i,j) = w(i) · w(j)
```

A convolução 2-D é realizada como dois passes 1-D separados (horizontal e vertical),
com reflexão de borda (`mode="reflect"`).

### Pirâmide Gaussiana

```
G[0] = imagem original
G[k] = REDUCE(G[k-1])   onde   REDUCE = blur + subsample ×2
```

### Pirâmide Laplaciana

```
L[k]   = G[k] − EXPAND(G[k+1])    para k = 0 … N−2
L[N−1] = G[N−1]                   (topo: imagem de baixíssima frequência)

EXPAND(G) = blur(upsample(G)) × 4
```

### Blending nível a nível

```
LS[k] = GM[k] · LA[k]  +  (1 − GM[k]) · LB[k]
```

onde `GM[k]` é o k-ésimo nível da **pirâmide Gaussiana da máscara**.
A máscara suavizada em cada nível garante que a mistura de baixas frequências
seja gradual e que as altas frequências sejam preservadas localmente.

### Reconstrução (colapso)

```
resultado = EXPAND(LS[N−1]) + LS[N−2]
          = EXPAND(…) + LS[N−3]
          = …
```

### Justaposição direta (comparação)

```
pixel = A   se mask > 0.5
pixel = B   caso contrário
```

---

## Por que pirâmides funcionam melhor?

| Aspecto | Justaposição direta | Blending por pirâmide |
|---|---|---|
| Baixas frequências (cores/tons) | Corte abrupto | Transição suave (máscara borrada) |
| Altas frequências (texturas) | Corte abrupto | Preservadas de cada imagem separadamente |
| Resultado visual | "Fantasma" / borda dura | Natural, sem artefatos |

---

## Estrutura do código

```
atv02.py
├── _W                       kernel Gaussiano 1-D
├── _gauss_blur()            filtro Gaussiano 5×5 separável (manual)
├── _reduce()                REDUCE: blur + subsample ×2
├── _expand()                EXPAND: upsample ×2 + blur ×4
├── build_gaussian_pyramid() pirâmide Gaussiana
├── build_laplacian_pyramid() pirâmide Laplaciana
├── blend_pyramids()         combinação nível a nível pela máscara
├── collapse_pyramid()       reconstrução (colapso)
├── pyramid_blend()          pipeline completo de blending
├── direct_cut()             justaposição direta (referência)
├── _ScrollFrame             frame com barra de rolagem (≡ atv01)
└── Atv02App(tk.Tk)
    ├── _build_menu()
    ├── _build_layout()
    ├── _try_load_defaults()   carrega imagens padrão ao iniciar
    ├── cmd_load_a/b/mask()
    ├── cmd_run()              executa o blending
    ├── cmd_save()
    ├── _refresh_canvas()      exibe resultado no canvas
    └── _add_compare_labels()  rótulos na visualização de comparação
```
