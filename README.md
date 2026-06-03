# genetic-pathfinder

Route optimization using A* and genetic algorithms, visualized in real time with pygame.

Given a map with checkpoints, the program finds the most efficient route order and team assignment to minimize total cost — combining terrain cost and time.

---

### Authors

- Matheus Figueiredo
- João Marcelo Onofre
- Enzo Farina

---

### Project structure

- `main.py` — all the core logic
- `BUILD.md` — instructions to run the project
- `mapa.txt` — the map used as input

---

### How it works

**1. Map reading**

The program reads `mapa.txt` and extracts the starting position, destination, and all checkpoints the character must visit.

```python
mapa = leMapa('mapa.txt')

posicao_inicial = encontraPosicaoNoMapa(mapa, '0')
posicao_destino = encontraPosicaoNoMapa(mapa, 'Z')

checkpoints = coletaCheckpointsNoMapa(mapa)
```

**2. Cost matrix precomputation**

Before searching for routes, the full cost matrix is precomputed by running A* once for every relevant pair of nodes. This avoids redundant pathfinding calls and significantly reduces runtime. The process is rendered visually in the pygame interface.

```python
matriz = preComputarMatriz(mapa, todos_nos, tela)
```

**3. Route optimization — genetic algorithm**

A genetic algorithm finds the optimal order to visit all checkpoints, minimizing total route cost.

```python
ordem_otima, custo_rota, historico_rotas = algoritmoGeneticoRotas(
    checkpoints, posicao_inicial, posicao_destino, matriz, CONFIG_AG_ROTAS
)
```

**4. Team optimization — genetic algorithm**

A second genetic algorithm determines the best team assignment for each checkpoint stage, minimizing total time.

```python
melhor_equipes, melhor_tempo, historico_equipe = algoritmoGeneticoEquipe(
    ordem_otima, CONFIG_AG_EQUIPE
)
```

**5. Visualization**

The final path is animated step by step using pygame.

```python
caminho_percorrido, custo_terreno, tempo_total = animarCaminhoDoAEstrela(
    mapa, ordem_otima, melhor_equipes,
    posicao_inicial, posicao_destino, tela
)
```

---

### Benchmark results (50 seeds)

| Metric | Value |
|---|---|
| Best seed | 4 |
| Terrain cost | 1808 |
| Total time | 1874.65 |
| Combined cost | 3682.65 |

---

### Demo

[Watch the presentation video](https://drive.google.com/file/d/1Nk-mrZz6GAYbWyvFxI3af7umLL5li5b-/view?usp=sharing)
