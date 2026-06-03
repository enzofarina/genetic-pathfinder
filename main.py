"""
Pathfinder Avatar — A* + Algoritmo Genético (Rotas + Equipe)
=============================================================
Fluxo em 4 fases:

  Fase 1 — A* silencioso
      Pré-computa a matriz de custos reais de terreno entre todos os pares
      (inicio, checkpoints, destino).

  Fase 2 — AG de Rotas
      Evolui permutações dos checkpoints para minimizar:
          custo_terreno_total + soma(dificuldades_etapas)
      Operadores: elitismo, torneio, crossover OX, mutação swap e inversão.

  Fase 3 — AG de Equipe
      Para cada etapa na ordem definida pela Fase 2, decide quais personagens
      participam, minimizando o TEMPO total:
          Tempo_etapa = dificuldade / soma(agilidades_dos_escolhidos)
      Restrições hard:
          - Cada personagem possui 8 pontos de energia (−1 por etapa usada).
          - Ao menos 1 personagem deve terminar com energia > 0.
          - Nenhuma etapa pode ter subconjunto vazio (penalidade infinita).
      Representação do indivíduo: lista de bitmasks (um por etapa),
      onde cada bit indica se o personagem i participa da etapa j.

  Fase 4 — A* animado
      Percorre o caminho na ordem da Fase 2 com animação pygame,
      imprimindo custos e tempos de cada etapa.
"""

import random
import argparse
import pygame
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configurações do mapa
# ─────────────────────────────────────────────────────────────────────────────

custos_terrenos: dict[str, int] = {
    'M': 200,
    'A': 15,
    'F': 10,
    'R': 5,
    '.': 1,
    '0': 1,
    'Z': 1,
}

dificuldades_etapas: dict[str, int] = {
    '1': 10,  '2': 20,  '3': 30,  '4': 40,  '5': 50,
    '6': 60,  '7': 70,  '8': 80,  '9': 90,
    'B': 100, 'C': 110, 'D': 120, 'E': 130, 'G': 140,
    'H': 150, 'I': 160, 'J': 170, 'K': 180, 'L': 190,
    'N': 200, 'O': 210, 'P': 220, 'Q': 230, 'S': 240,
    'T': 250, 'U': 260, 'V': 270, 'W': 280, 'X': 290,
    'Y': 300, 'Z': 310,
}

# Checkpoints intermediários — todos exceto 'Z' (destino)
IDENTIFICADORES_CHECKPOINTS: set[str] = set(dificuldades_etapas.keys()) - {'Z'}

# ─────────────────────────────────────────────────────────────────────────────
# Personagens
# ─────────────────────────────────────────────────────────────────────────────

PERSONAGENS: list[str] = ['Aang', 'Zukko',
                          'Toph', 'Katara', 'Sokka', 'Appa', 'Momo']

AGILIDADES: dict[str, float] = {
    'Aang':   1.8,
    'Zukko':  1.6,
    'Toph':   1.6,
    'Katara': 1.6,
    'Sokka':  1.4,
    'Appa':   0.9,
    'Momo':   0.7,
}

ENERGIA_MAXIMA: int = 8      # pontos de energia iniciais por personagem
NUM_PERSONAGENS: int = len(PERSONAGENS)

# ─────────────────────────────────────────────────────────────────────────────
# Configurações dos Algoritmos Genéticos
# ─────────────────────────────────────────────────────────────────────────────

# AG de Rotas (Fase 2) — otimiza a ordem dos checkpoints
CONFIG_AG_ROTAS: dict[str, int | float] = {
    'tamanho_populacao':               80,
    'max_rodadas_por_geracao':         300,
    'taxa_crossover':                  0.85,
    'taxa_mutacao_swap':               0.15,
    'taxa_mutacao_inversao':           0.08,
    'candidatos_selecao_torneio':      5,
    'individuos_preservados_elitismo': 4,
    'max_rodadas_sem_melhora':         60,
}

# AG de Equipe (Fase 3) — otimiza quais personagens participam de cada etapa
CONFIG_AG_EQUIPE: dict[str, int | float] = {
    'tamanho_populacao':               100,
    'max_rodadas_por_geracao':         400,
    'taxa_crossover':                  0.80,
    # probabilidade de flipar 1 bit por etapa
    'taxa_mutacao_bit':                0.12,
    'candidatos_selecao_torneio':      5,
    'individuos_preservados_elitismo': 5,
    'max_rodadas_sem_melhora':         80,
}

PENALIDADE_INFINITA: float = 1e9   # Custo para soluções inviáveis

# ─────────────────────────────────────────────────────────────────────────────
# Configurações do pygame
# ─────────────────────────────────────────────────────────────────────────────

cores_dos_identificadores: dict[str, tuple[int, int, int]] = {
    'M': (139, 69,  19),
    'A': (0,   0,   255),
    'F': (34,  139, 34),
    'R': (128, 128, 128),
    '.': (124, 252, 0),
    '*': (255, 255, 255),
    '@': (0,   255, 255),
    '?': (255, 165, 0),
    'V': (75,  0,   130),
}
for _ck in list(IDENTIFICADORES_CHECKPOINTS) + ['Z', '0']:
    cores_dos_identificadores[_ck] = (255, 0, 255)

pygame.init()
PYGAME_TAMANHO_CELULA: int = 4
pygame_fonte = pygame.font.SysFont('Arial', 18)


def pygame_exibeTexto(tela,
                      texto: str,
                      posicao: tuple[int, int],
                      cor: tuple[int, int, int] = (255, 255, 255)) -> None:
    superficie = pygame_fonte.render(texto, True, cor)
    tela.blit(superficie, posicao)


def pygame_desenhaMapa(tela,
                       mapa: list[list[str]],
                       agente_pos,
                       fronteira,
                       visitados,
                       caminho_final,
                       custo_total,
                       info_extra: str = '') -> None:
    tela.fill((0, 0, 0))
    fronteira_set = set(fronteira)
    for i, row in enumerate(mapa):
        for j, letra in enumerate(row):
            x = j * PYGAME_TAMANHO_CELULA
            y = i * PYGAME_TAMANHO_CELULA
            pos = (i, j)

            if pos in caminho_final and letra not in dificuldades_etapas and letra != '0':
                cor = cores_dos_identificadores['*']
            elif pos == agente_pos:
                cor = cores_dos_identificadores['@']
            elif pos in fronteira_set:
                cor = cores_dos_identificadores['?']
            elif pos in visitados:
                cor = cores_dos_identificadores['V']
            elif letra in cores_dos_identificadores:
                cor = cores_dos_identificadores[letra]
            else:
                cor = (255, 255, 255)

            pygame.draw.rect(tela, cor, pygame.Rect(
                x, y, PYGAME_TAMANHO_CELULA, PYGAME_TAMANHO_CELULA))

    pygame_exibeTexto(tela, f"Custo Total: {custo_total:.1f}", (10, 10))
    if info_extra:
        pygame_exibeTexto(tela, info_extra, (10, 32), cor=(255, 220, 100))
    pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────────────
# I/O do mapa
# ─────────────────────────────────────────────────────────────────────────────

def leMapa(nome_do_arquivo: str) -> list[list[str]]:
    with open(nome_do_arquivo, 'r') as arquivo:
        return [list(linha.strip()) for linha in arquivo.readlines()]


def encontraPosicaoNoMapa(mapa: list[list[str]],
                          identificador: str) -> Optional[tuple[int, int]]:
    for indice, linha in enumerate(mapa):
        if identificador in linha:
            return (indice, linha.index(identificador))
    return None


def coletaCheckpointsNoMapa(mapa: list[list[str]]) -> list[tuple[tuple[int, int], str]]:
    checkpoints: list[tuple[tuple[int, int], str]] = []
    for x, linha in enumerate(mapa):
        for y, letra in enumerate(linha):
            if letra in IDENTIFICADORES_CHECKPOINTS:
                checkpoints.append(((x, y), letra))
    return checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# Núcleo do A*
# ─────────────────────────────────────────────────────────────────────────────

def heuristica(ponto_a: tuple[int, int], ponto_b: tuple[int, int]) -> int:
    """Distância de Manhattan entre dois pontos."""
    return abs(ponto_a[0] - ponto_b[0]) + abs(ponto_a[1] - ponto_b[1])


def aEstrela(mapa: list[list[str]],
             no_inicio: tuple[int, int],
             no_destino: tuple[int, int],
             tela=None,
             nos_visitados_externo: Optional[set] = None,
             intervalo_entre_atualizacoes: int = 1000,
             caminho_pintado: Optional[set] = None,
             custo_acumulado: float = 0
             ) -> tuple[list[tuple[int, int]], int]:
    """
    Algoritmo A*.
    Retorna (caminho, custo_terreno).
    Se tela for fornecida, anima a busca em pygame.
    """
    fila_de_nos: list[tuple[int, tuple[int, int]]] = [(0, no_inicio)]
    custo_ate_cada_no: dict[tuple[int, int], int] = {no_inicio: 0}
    no_pai: dict[tuple[int, int], Optional[tuple[int, int]]] = {
        no_inicio: None}
    nos_visitados: set[tuple[int, int]] = set()
    quantidade_de_passos: int = 0

    while fila_de_nos:
        if tela:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        fila_de_nos.sort(key=lambda x: x[0])
        _, no_atual = fila_de_nos.pop(0)

        if no_atual == no_destino:
            break
        if no_atual in nos_visitados:
            continue

        nos_visitados.add(no_atual)
        if nos_visitados_externo is not None:
            nos_visitados_externo.add(no_atual)

        quantidade_de_passos += 1
        if tela and quantidade_de_passos % intervalo_entre_atualizacoes == 0:
            fronteira = [no[1] for no in fila_de_nos]
            pygame_desenhaMapa(
                tela, mapa, no_atual, fronteira,
                nos_visitados_externo if nos_visitados_externo is not None else nos_visitados,
                caminho_pintado or set(),
                custo_acumulado,
            )

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            no_vizinho: tuple[int, int] = (no_atual[0] + dx, no_atual[1] + dy)
            if not (0 <= no_vizinho[0] < len(mapa) and 0 <= no_vizinho[1] < len(mapa[0])):
                continue

            identificador_vizinho: str = mapa[no_vizinho[0]][no_vizinho[1]]
            custo_movimento: int = custos_terrenos.get(
                identificador_vizinho, 1)
            novo_custo: int = custo_ate_cada_no[no_atual] + custo_movimento

            if no_vizinho not in custo_ate_cada_no or novo_custo < custo_ate_cada_no[no_vizinho]:
                custo_ate_cada_no[no_vizinho] = novo_custo
                estimativa_total: int = novo_custo + \
                    heuristica(no_vizinho, no_destino)
                fila_de_nos.append((estimativa_total, no_vizinho))
                no_pai[no_vizinho] = no_atual

    # Reconstrói caminho
    caminho, no_atual = [], no_destino
    while no_atual is not None:
        caminho.append(no_atual)
        no_atual = no_pai.get(no_atual)
    caminho.reverse()

    return caminho, custo_ate_cada_no.get(no_destino, float('inf'))


# ─────────────────────────────────────────────────────────────────────────────
# Fase 1 — Matriz de custos A*
# ─────────────────────────────────────────────────────────────────────────────

def preComputarMatriz(mapa: list[list[str]],
                      nos: list[tuple[int, int]],
                      tela
                      ) -> dict[tuple[tuple[int, int], tuple[int, int]], int]:
    """
    Roda A* silencioso entre todos os pares.
    Retorna {(a, b): custo_terreno}.
    """
    n = len(nos)
    print()
    print(f"[Fase 1] Pré-computando matriz {n}x{n} — {n*n} execuções de A*...")
    matriz: dict = {}
    for i, a in enumerate(nos):
        for b in nos:
            if a == b:
                matriz[(a, b)] = 0
            else:
                _, custo = aEstrela(mapa, a, b, tela=tela)
                matriz[(a, b)] = custo
        print(f"  {i+1:3d}/{n} nós processados", end='\r')
    print()
    return matriz


# ─────────────────────────────────────────────────────────────────────────────
# Fase 2 — AG de Rotas (ordem dos checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

def fitness_rotas(individuo: list[int],
                  posicao_inicial: tuple[int, int],
                  posicao_destino: tuple[int, int],
                  matriz: dict,
                  checkpoints: dict[int, tuple[tuple[int, int], str]]) -> float:
    """
    Fitness da rota = custo_terreno + soma(dificuldades).
    """
    rota = ([posicao_inicial]
            + [checkpoints[i][0] for i in individuo]
            + [posicao_destino])
    custo_terreno = sum(
        matriz.get((rota[k], rota[k + 1]), float('inf'))
        for k in range(len(rota) - 1)
    )
    custo_dificuldade = sum(
        dificuldades_etapas.get(checkpoints[i][1], 0) for i in individuo
    )
    return custo_terreno + custo_dificuldade


def selecaoPorTorneio(populacao: list, pontuacoes: list[float], k: int) -> list:
    candidatos = random.sample(range(len(populacao)), k)
    vencedor = min(candidatos, key=lambda i: pontuacoes[i])
    return populacao[vencedor][:]


def crossoverOX(pai: list[int], mae: list[int]) -> list[int]:
    """Order Crossover (OX) — garante permutação válida."""
    n = len(pai)
    a, b = sorted(random.sample(range(n), 2))
    filho = [None] * n
    filho[a:b + 1] = pai[a:b + 1]
    segmento = set(pai[a:b + 1])
    restantes = [g for g in mae if g not in segmento]
    idx = 0
    for i in range(n):
        if filho[i] is None:
            filho[i] = restantes[idx]
            idx += 1
    return filho


def mutacaoPorSwap(individuo: list[int], taxa: float) -> list[int]:
    ind = individuo[:]
    if random.random() < taxa:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind


def mutacaoPorInversao(individuo: list[int], taxa: float) -> list[int]:
    ind = individuo[:]
    if random.random() < taxa:
        i, j = sorted(random.sample(range(len(ind)), 2))
        ind[i:j + 1] = ind[i:j + 1][::-1]
    return ind


def algoritmoGeneticoRotas(posicoes_checkpoints: list[tuple[tuple[int, int], str]],
                           posicao_inicial: tuple[int, int],
                           posicao_destino: tuple[int, int],
                           matriz: dict,
                           config: dict
                           ) -> tuple[list[tuple[tuple[int, int], str]], float, list[float]]:
    """
    AG Fase 2: evolui a ordem de visita dos checkpoints.
    Retorna (melhor_ordem, melhor_custo, historico).
    """
    n = len(posicoes_checkpoints)
    checkpoints = {i: posicoes_checkpoints[i] for i in range(n)}

    populacao = [random.sample(range(n), n)
                 for _ in range(config['tamanho_populacao'])]
    pontuacoes = [fitness_rotas(ind, posicao_inicial, posicao_destino, matriz, checkpoints)
                  for ind in populacao]

    idx_melhor = min(range(len(pontuacoes)), key=lambda i: pontuacoes[i])
    melhor_ind = populacao[idx_melhor][:]
    melhor_custo = pontuacoes[idx_melhor]
    historico = [melhor_custo]
    sem_melhora = 0

    print()
    print("[Fase 2] AG de Rotas")
    print(f"  Geração   0 | Melhor custo: {melhor_custo:.0f}")

    for rodada in range(1, config['max_rodadas_por_geracao'] + 1):
        elite_idx = sorted(range(len(pontuacoes)), key=lambda i: pontuacoes[i])
        nova_pop = [populacao[i][:]
                    for i in elite_idx[:config['individuos_preservados_elitismo']]]

        while len(nova_pop) < config['tamanho_populacao']:
            pai = selecaoPorTorneio(
                populacao, pontuacoes, config['candidatos_selecao_torneio'])
            mae = selecaoPorTorneio(
                populacao, pontuacoes, config['candidatos_selecao_torneio'])
            filho = crossoverOX(pai, mae) if random.random(
            ) < config['taxa_crossover'] else pai[:]
            filho = mutacaoPorSwap(filho, config['taxa_mutacao_swap'])
            filho = mutacaoPorInversao(filho, config['taxa_mutacao_inversao'])
            nova_pop.append(filho)

        populacao = nova_pop
        pontuacoes = [fitness_rotas(ind, posicao_inicial, posicao_destino, matriz, checkpoints)
                      for ind in populacao]

        idx_novo_melhor = min(range(len(pontuacoes)),
                              key=lambda i: pontuacoes[i])
        novo_melhor = pontuacoes[idx_novo_melhor]

        if novo_melhor < melhor_custo:
            melhor_custo = novo_melhor
            melhor_ind = populacao[idx_novo_melhor][:]
            sem_melhora = 0
        else:
            sem_melhora += 1

        historico.append(melhor_custo)

        if rodada % 20 == 0:
            media = sum(pontuacoes) / len(pontuacoes)
            print(f"  Geração {rodada:3d} | Melhor: {melhor_custo:.0f} "
                  f"| Média: {media:.0f} | Sem melhora: {sem_melhora}")

        if sem_melhora >= config['max_rodadas_sem_melhora']:
            print(f"  Parada antecipada na geração {rodada} "
                  f"({sem_melhora} gerações sem melhora).")
            break

    melhor_ordem = [checkpoints[i] for i in melhor_ind]
    return melhor_ordem, melhor_custo, historico


# ─────────────────────────────────────────────────────────────────────────────
# Fase 3 — AG de Equipe (personagens por etapa)
# ─────────────────────────────────────────────────────────────────────────────

def bitmask_para_lista(bitmask: int) -> list[str]:
    """Converte bitmask inteiro → lista de nomes dos personagens selecionados."""
    return [PERSONAGENS[i] for i in range(NUM_PERSONAGENS) if bitmask & (1 << i)]


def calcula_tempo_etapa(dificuldade: float, personagens_escolhidos: list[str]) -> float:
    """Tempo = dificuldade / soma(agilidades). Retorna inf se nenhum personagem."""
    soma_agi = sum(AGILIDADES[p] for p in personagens_escolhidos)
    if soma_agi == 0:
        return float('inf')
    return dificuldade / soma_agi


def verifica_viabilidade(individuo: list[int],
                         dificuldades_sequencia: list[int]
                         ) -> tuple[bool, list[int]]:
    """
    Verifica se o indivíduo respeita todas as restrições:
      1. Nenhuma etapa com subconjunto vazio.
      2. Nenhum personagem excede 8 usos.
      3. Ao menos 1 personagem sobrevive com energia > 0 ao final.

    Retorna (viavel, energia_final_de_cada_personagem).
    """
    energia = [ENERGIA_MAXIMA] * NUM_PERSONAGENS

    for etapa_idx, bitmask in enumerate(individuo):
        if bitmask == 0:
            return False, energia   # etapa sem nenhum personagem

        for p_idx in range(NUM_PERSONAGENS):
            if bitmask & (1 << p_idx):
                energia[p_idx] -= 1
                if energia[p_idx] < 0:
                    return False, energia   # personagem esgotado

    # Ao menos 1 sobrevivente
    if all(e <= 0 for e in energia):
        return False, energia

    return True, energia


def fitness_equipe(individuo: list[int],
                   dificuldades_sequencia: list[int]) -> float:
    """
    Fitness = soma dos tempos de cada etapa.
    Soluções inviáveis recebem penalidade proporcional às violações.
    """
    viavel, energia = verifica_viabilidade(individuo, dificuldades_sequencia)
    if not viavel:
        return PENALIDADE_INFINITA

    tempo_total = 0.0
    for bitmask, dif in zip(individuo, dificuldades_sequencia):
        personagens = bitmask_para_lista(bitmask)
        tempo_total += calcula_tempo_etapa(dif, personagens)

    return tempo_total


def gera_individuo_equipe_aleatorio(num_etapas: int) -> list[int]:
    """
    Gera um indivíduo aleatório viável para o AG de Equipe.
    Cada etapa recebe um subconjunto aleatório não-vazio de personagens,
    respeitando o limite de 8 usos por personagem.
    """
    for _ in range(200):   # tenta até conseguir um viável
        energia = [ENERGIA_MAXIMA] * NUM_PERSONAGENS
        individuo = []
        viavel = True

        for _ in range(num_etapas):
            # Personagens disponíveis com energia > 0
            disponiveis = [i for i in range(NUM_PERSONAGENS) if energia[i] > 0]
            if not disponiveis:
                viavel = False
                break
            # Escolhe subconjunto aleatório não-vazio
            k = random.randint(1, len(disponiveis))
            escolhidos = random.sample(disponiveis, k)
            bitmask = sum(1 << i for i in escolhidos)
            individuo.append(bitmask)
            for i in escolhidos:
                energia[i] -= 1

        if viavel and any(e > 0 for e in energia):
            return individuo

    # Fallback: cada etapa usa apenas o personagem mais ágil disponível
    return [1] * num_etapas   # Aang em todas (aceita penalidade se necessário)


def crossover_equipe_uniforme(pai: list[int], mae: list[int]) -> list[int]:
    """
    Crossover uniforme: para cada etapa, escolhe o bitmask do pai ou da mãe.
    """
    return [pai[i] if random.random() < 0.5 else mae[i] for i in range(len(pai))]


def mutacao_equipe_bit(individuo: list[int], taxa: float) -> list[int]:
    """
    Para cada etapa, com probabilidade `taxa`, flipa um bit aleatório do bitmask
    (adiciona ou remove um personagem daquela etapa).
    Garante que a etapa não fique vazia.
    """
    ind = individuo[:]
    for i in range(len(ind)):
        if random.random() < taxa:
            bit = 1 << random.randint(0, NUM_PERSONAGENS - 1)
            novo = ind[i] ^ bit
            if novo != 0:   # não permite etapa vazia
                ind[i] = novo
    return ind


def repara_individuo(individuo: list[int],
                     dificuldades_sequencia: list[int]) -> list[int]:
    """
    Tenta reparar um indivíduo inviável:
    - Remove personagens esgotados (energia < 0) da etapa que os excedeu.
    - Se a etapa ficar vazia, força o personagem com mais energia restante.
    - Se nenhum sobrevivente, garante que a última etapa preserve o mais descansado.
    """
    ind = individuo[:]
    energia = [ENERGIA_MAXIMA] * NUM_PERSONAGENS

    for etapa_idx in range(len(ind)):
        bitmask = ind[etapa_idx]
        bits_validos = 0

        for p_idx in range(NUM_PERSONAGENS):
            if bitmask & (1 << p_idx):
                if energia[p_idx] > 0:
                    bits_validos |= (1 << p_idx)
                    energia[p_idx] -= 1

        if bits_validos == 0:
            # Força personagem com mais energia
            melhor = max(range(NUM_PERSONAGENS), key=lambda i: energia[i])
            bits_validos = 1 << melhor
            energia[melhor] -= 1

        ind[etapa_idx] = bits_validos

    # Garante ao menos 1 sobrevivente
    if all(e <= 0 for e in energia):
        # Encontra etapa onde podemos liberar um personagem
        melhor_p = max(range(NUM_PERSONAGENS),
                       key=lambda i: sum(1 for b in ind if b & (1 << i)))
        # Remove o personagem mais usado de sua última aparição
        for etapa_idx in range(len(ind) - 1, -1, -1):
            if ind[etapa_idx] & (1 << melhor_p) and bin(ind[etapa_idx]).count('1') > 1:
                ind[etapa_idx] &= ~(1 << melhor_p)
                break

    return ind


def algoritmoGeneticoEquipe(ordem_checkpoints: list[tuple[tuple[int, int], str]],
                            config: dict
                            ) -> tuple[list[list[str]], float, list[float]]:
    """
    AG Fase 3: decide quais personagens participam de cada etapa.

    Representação: list[int] de tamanho num_etapas,
    onde cada inteiro é um bitmask de personagens.

    Retorna (melhor_equipes, melhor_tempo, historico).
    melhor_equipes[i] = lista de nomes dos personagens na etapa i.
    """
    # Inclui o destino Z (etapa final)
    sequencia_completa = list(ordem_checkpoints) + [(None, 'Z')]
    dificuldades_seq = [dificuldades_etapas.get(char, 0)
                        for _, char in sequencia_completa]
    num_etapas = len(dificuldades_seq)

    print()
    print("[Fase 3] AG de Equipe")
    print(f"  Etapas a otimizar: {num_etapas} | "
          f"Personagens: {NUM_PERSONAGENS} | "
          f"Energia máx/personagem: {ENERGIA_MAXIMA}")

    # Geração da população inicial
    populacao = [gera_individuo_equipe_aleatorio(num_etapas)
                 for _ in range(config['tamanho_populacao'])]

    pontuacoes = [fitness_equipe(ind, dificuldades_seq) for ind in populacao]

    idx_melhor = min(range(len(pontuacoes)), key=lambda i: pontuacoes[i])
    melhor_ind = populacao[idx_melhor][:]
    melhor_custo = pontuacoes[idx_melhor]
    historico = [melhor_custo]
    sem_melhora = 0

    print(f"  Geração   0 | Melhor tempo: {melhor_custo:.4f}")

    for rodada in range(1, config['max_rodadas_por_geracao'] + 1):
        elite_idx = sorted(range(len(pontuacoes)), key=lambda i: pontuacoes[i])
        nova_pop = [populacao[i][:]
                    for i in elite_idx[:config['individuos_preservados_elitismo']]]

        while len(nova_pop) < config['tamanho_populacao']:
            pai = selecaoPorTorneio(
                populacao, pontuacoes, config['candidatos_selecao_torneio'])
            mae = selecaoPorTorneio(
                populacao, pontuacoes, config['candidatos_selecao_torneio'])

            if random.random() < config['taxa_crossover']:
                filho = crossover_equipe_uniforme(pai, mae)
            else:
                filho = pai[:]

            filho = mutacao_equipe_bit(filho, config['taxa_mutacao_bit'])
            filho = repara_individuo(filho, dificuldades_seq)
            nova_pop.append(filho)

        populacao = nova_pop
        pontuacoes = [fitness_equipe(ind, dificuldades_seq)
                      for ind in populacao]

        idx_novo_melhor = min(range(len(pontuacoes)),
                              key=lambda i: pontuacoes[i])
        novo_melhor = pontuacoes[idx_novo_melhor]

        if novo_melhor < melhor_custo:
            melhor_custo = novo_melhor
            melhor_ind = populacao[idx_novo_melhor][:]
            sem_melhora = 0
        else:
            sem_melhora += 1

        historico.append(melhor_custo)

        if rodada % 20 == 0:
            media = sum(p for p in pontuacoes if p < PENALIDADE_INFINITA / 2)
            n_viaveis = sum(1 for p in pontuacoes if p <
                            PENALIDADE_INFINITA / 2)
            print(f"  Geração {rodada:3d} | Melhor tempo: {melhor_custo:.4f} "
                  f"| Média viáveis: {media/max(n_viaveis, 1):.4f} "
                  f"| Sem melhora: {sem_melhora}")

        if sem_melhora >= config['max_rodadas_sem_melhora']:
            print(f"  Parada antecipada na geração {rodada} "
                  f"({sem_melhora} gerações sem melhora).")
            break

    melhor_equipes = [bitmask_para_lista(bitmask) for bitmask in melhor_ind]
    return melhor_equipes, melhor_custo, historico


# ─────────────────────────────────────────────────────────────────────────────
# Fase 4 — Animação gráfica do A*
# ─────────────────────────────────────────────────────────────────────────────

def animarCaminhoDoAEstrela(mapa: list[list[str]],
                            ordem_visita: list[tuple[tuple[int, int], str]],
                            equipes_por_etapa: list[list[str]],
                            posicao_inicial: tuple[int, int],
                            posicao_destino: tuple[int, int],
                            tela,
                            intervalo_entre_atualizacoes: int = 1000
                            ) -> tuple[list[tuple[int, int]], int, float]:
    """
    Executa A* animado, percorrendo: inicio → ck[0] → ... → destino.
    Calcula e imprime custo de terreno e tempo de cada etapa.
    Retorna (caminho_percorrido, custo_terreno_total, tempo_total).
    """
    rota: list[tuple[int, int]] = []
    nos_visitados: set[tuple[int, int]] = set()
    custo_terreno_acumulado: int = 0
    tempo_acumulado: float = 0.0
    posicao_atual = posicao_inicial

    # Inclui Z como última etapa na sequência
    rota_checkpoints = list(ordem_visita) + [(posicao_destino, 'Z')]
    tamanho_rota = len(rota_checkpoints)

    print()
    print(f"[Fase 4] Executando caminho final ({tamanho_rota} trechos)...")

    for indice, ((proxima_posicao, proximo_id), equipe) in enumerate(
            zip(rota_checkpoints, equipes_por_etapa)):

        melhor_caminho, custo_trecho = aEstrela(
            mapa, posicao_atual, proxima_posicao,
            tela=tela,
            nos_visitados_externo=nos_visitados,
            intervalo_entre_atualizacoes=intervalo_entre_atualizacoes,
            caminho_pintado=set(rota),
            custo_acumulado=custo_terreno_acumulado + tempo_acumulado,
        )

        if rota:
            rota.extend(melhor_caminho[1:])  # evita duplicar ponto de junção
        else:
            rota.extend(melhor_caminho)

        custo_terreno_acumulado += custo_trecho

        dificuldade = dificuldades_etapas.get(proximo_id, 0)
        tempo_etapa = calcula_tempo_etapa(dificuldade, equipe)
        tempo_acumulado += tempo_etapa

        soma_agi = sum(AGILIDADES[p] for p in equipe)
        print(
            f"  [{indice+1:2d}/{tamanho_rota}] → '{proximo_id}' {proxima_posicao}")
        print(f"    Equipe:          {equipe}")
        print(f"    Soma agilidade:  {soma_agi:.1f}")
        print(f"    Dificuldade:     {dificuldade}")
        print(f"    Tempo da etapa:  {tempo_etapa:.4f}")
        print(f"    Custo terreno:   {custo_trecho}")
        print(
            f"    Acum. terreno:   {custo_terreno_acumulado} | Acum. tempo: {tempo_acumulado:.4f}")

        posicao_atual = proxima_posicao

    return rota, custo_terreno_acumulado, tempo_acumulado


def calculaCustoTerrenoPorMatriz(posicao_inicial: tuple[int, int],
                                 posicao_destino: tuple[int, int],
                                 ordem_visita: list[tuple[tuple[int, int], str]],
                                 matriz: dict[tuple[tuple[int, int],
                                                    tuple[int, int]], int]
                                 ) -> int:
    """
    Soma o custo de terreno da rota (inicio -> checkpoints -> destino)
    usando a matriz pré-computada na Fase 1.
    """
    nos_rota = [posicao_inicial] + \
        [p for p, _ in ordem_visita] + [posicao_destino]
    return sum(matriz[(nos_rota[i], nos_rota[i + 1])] for i in range(len(nos_rota) - 1))


def executarRodadaComSeed(seed: int) -> dict:
    """
    Executa as fases 1, 2 e 3 sem animação para uma seed fixa.
    Retorna métricas para comparação entre seeds.
    """
    random.seed(seed)

    mapa = leMapa('mapa.txt')
    posicao_inicial = encontraPosicaoNoMapa(mapa, '0')
    posicao_destino = encontraPosicaoNoMapa(mapa, 'Z')

    if posicao_inicial is None or posicao_destino is None:
        raise ValueError("Erro: '0' ou 'Z' não encontrado no mapa.")

    checkpoints = coletaCheckpointsNoMapa(mapa)
    todos_nos = ([posicao_inicial]
                 + [p for p, _ in checkpoints]
                 + [posicao_destino])

    matriz = preComputarMatriz(mapa, todos_nos, tela=None)

    ordem_otima, custo_rota, historico_rotas = algoritmoGeneticoRotas(
        checkpoints, posicao_inicial, posicao_destino, matriz, CONFIG_AG_ROTAS
    )

    melhor_equipes, melhor_tempo, historico_equipe = algoritmoGeneticoEquipe(
        ordem_otima, CONFIG_AG_EQUIPE
    )

    custo_terreno = calculaCustoTerrenoPorMatriz(
        posicao_inicial, posicao_destino, ordem_otima, matriz
    )

    sequencia_completa = list(ordem_otima) + [(posicao_destino, 'Z')]
    tempo_total = sum(
        calcula_tempo_etapa(dificuldades_etapas.get(ident, 0), equipe)
        for (_, ident), equipe in zip(sequencia_completa, melhor_equipes)
    )

    return {
        'seed': seed,
        'ordem_otima': ordem_otima,
        'melhor_equipes': melhor_equipes,
        'custo_rota_estimado': custo_rota,
        'custo_terreno': custo_terreno,
        'tempo_total': tempo_total,
        'custo_combinado': custo_terreno + tempo_total,
        'historico_rotas': historico_rotas,
        'historico_equipe': historico_equipe,
    }


def buscarMelhorSeed(seeds: list[int], arquivo_saida: str = 'melhor_seed.txt') -> dict:
    """
    Roda várias seeds e salva em arquivo a seed com menor custo combinado.
    Custo combinado = custo_terreno + tempo_total.
    """
    if not seeds:
        raise ValueError('A lista de seeds não pode estar vazia.')

    print()
    print(f"[Benchmark] Iniciando avaliação de {len(seeds)} seeds...")

    melhor_resultado = None
    for i, seed in enumerate(seeds, start=1):
        resultado = executarRodadaComSeed(seed)
        print(f"  [{i:2d}/{len(seeds)}] seed={seed:4d} | "
              f"terreno={resultado['custo_terreno']} | "
              f"tempo={resultado['tempo_total']:.4f} | "
              f"combinado={resultado['custo_combinado']:.4f}")

        if (melhor_resultado is None
                or resultado['custo_combinado'] < melhor_resultado['custo_combinado']):
            melhor_resultado = resultado

    with open(arquivo_saida, 'w', encoding='utf-8') as arquivo:
        arquivo.write(f"Melhor seed: {melhor_resultado['seed']}\n")
        arquivo.write(f"Custo terreno: {melhor_resultado['custo_terreno']}\n")
        arquivo.write(f"Tempo total: {melhor_resultado['tempo_total']:.4f}\n")
        arquivo.write(
            f"Custo combinado: {melhor_resultado['custo_combinado']:.4f}\n")

    print()
    print(f"[Benchmark] Melhor seed: {melhor_resultado['seed']} "
          f"(combinado={melhor_resultado['custo_combinado']:.4f})")
    print(f"[Benchmark] Resultado salvo em: {arquivo_saida}")

    return melhor_resultado


def parseia_argumentos() -> argparse.Namespace:
    """
    Parseia argumentos de linha de comando para benchmark de seeds.

    Exemplo:
      python main.py --num-seeds 50
    """
    parser = argparse.ArgumentParser(
        description='Pathfinder Avatar — A* + AG Rotas + AG Equipe'
    )
    parser.add_argument(
        '--num-seeds',
        default=0,
        type=int,
        help='Quantidade de seeds para benchmark (testa seeds de 1 ate N).'
    )
    return parser.parse_args()


def seeds_a_partir_dos_argumentos(args: argparse.Namespace) -> list[int]:
    """
    Gera a lista de seeds a partir da quantidade informada em --num-seeds.
    """
    if args.num_seeds < 0:
        raise ValueError('O valor de --num-seeds nao pode ser negativo.')
    if args.num_seeds == 0:
        return []
    return list(range(1, args.num_seeds + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parseia_argumentos()
    seeds_para_benchmark = seeds_a_partir_dos_argumentos(args)

    if seeds_para_benchmark:
        melhor = buscarMelhorSeed(seeds_para_benchmark)
        #random.seed(melhor['seed'])
        random.seed(4)
        print(
            f"\nExecutando animação final com a melhor seed: {melhor['seed']}")

    mapa = leMapa('mapa.txt')

    posicao_inicial = encontraPosicaoNoMapa(mapa, '0')
    posicao_destino = encontraPosicaoNoMapa(mapa, 'Z')

    if posicao_inicial is None or posicao_destino is None:
        print("Erro: '0' ou 'Z' não encontrado no mapa. Encerrando.")
        return 1

    largura_tela = min(1600, len(mapa[0]) * PYGAME_TAMANHO_CELULA)
    altura_tela = min(1400, len(mapa) * PYGAME_TAMANHO_CELULA)
    tela = pygame.display.set_mode((largura_tela, altura_tela))
    pygame.display.set_caption("Pathfinder — A* + AG Rotas + AG Equipe")

    # Coleta checkpoints
    checkpoints = coletaCheckpointsNoMapa(mapa)
    print(f"Checkpoints encontrados: {len(checkpoints)}")
    for pos, ident in checkpoints:
        print(f"  '{ident}' {pos}  dificuldade: {dificuldades_etapas[ident]}")

    # ── Fase 1: Matriz de custos ──────────────────────────────────────────────
    todos_nos = ([posicao_inicial]
                 + [p for p, _ in checkpoints]
                 + [posicao_destino])
    matriz = preComputarMatriz(mapa, todos_nos, tela)

    # ── Fase 2: AG de Rotas ───────────────────────────────────────────────────
    ordem_otima, custo_rota, historico_rotas = algoritmoGeneticoRotas(
        checkpoints, posicao_inicial, posicao_destino, matriz, CONFIG_AG_ROTAS
    )

    print()
    print(f"Melhor ordem (AG Rotas) — custo estimado: {custo_rota:.0f}")
    for idx, (pos, ident) in enumerate(ordem_otima):
        print(
            f"  {idx+1:2d}. '{ident}' {pos}  dif: {dificuldades_etapas[ident]}")

    # ── Fase 3: AG de Equipe ──────────────────────────────────────────────────
    melhor_equipes, melhor_tempo, historico_equipe = algoritmoGeneticoEquipe(
        ordem_otima, CONFIG_AG_EQUIPE
    )

    print()
    print(
        f"Melhor alocação de equipe (AG Equipe) — tempo total estimado: {melhor_tempo:.4f}")
    sequencia_completa = list(ordem_otima) + [(posicao_destino, 'Z')]
    energia_final = [ENERGIA_MAXIMA] * NUM_PERSONAGENS
    for idx, ((pos, ident), equipe) in enumerate(zip(sequencia_completa, melhor_equipes)):
        dif = dificuldades_etapas.get(ident, 0)
        t = calcula_tempo_etapa(dif, equipe)
        print(f"  Etapa {idx+1:2d} '{ident}' | equipe: {equipe} | "
              f"soma_agi: {sum(AGILIDADES[p] for p in equipe):.1f} | "
              f"dif: {dif} | tempo: {t:.4f}")
        for p in equipe:
            energia_final[PERSONAGENS.index(p)] -= 1

    print()
    print("Energia final dos personagens:")
    for p, e_ini in zip(PERSONAGENS, [ENERGIA_MAXIMA] * NUM_PERSONAGENS):
        e_fim = energia_final[PERSONAGENS.index(p)]
        usos = e_ini - e_fim
        status = "VIVO" if e_fim > 0 else "ESGOTADO"
        print(f"  {p:<8} | usou {usos:2d}x | energia restante: {e_fim} | {status}")

    # ── Fase 4: A* animado ────────────────────────────────────────────────────
    caminho_percorrido, custo_terreno, tempo_total = animarCaminhoDoAEstrela(
        mapa, ordem_otima, melhor_equipes,
        posicao_inicial, posicao_destino, tela
    )

    print()
    print(f"{'='*60}")
    print("Resultados finais")
    print(f"  Custo de terreno (A*):          {custo_terreno}")
    print(f"  Tempo total das etapas (AG):    {tempo_total:.4f}")
    print(
        f"  Custo combinado (terreno+tempo):{custo_terreno + tempo_total:.4f}")
    print(f"{'='*60}")

    print()
    print("Evolução AG Rotas (a cada 20 gerações):")
    for i, v in enumerate(historico_rotas):
        if i % 20 == 0:
            print(f"  Geração {i:3d}: {v:.0f}")

    print()
    print("Evolução AG Equipe (a cada 20 gerações):")
    for i, v in enumerate(historico_equipe):
        if i % 20 == 0:
            print(f"  Geração {i:3d}: {v:.4f}")

    # Mantém janela aberta
    caminho_set = set(caminho_percorrido)
    info = (f"Terreno: {custo_terreno} | "
            f"Tempo etapas: {tempo_total:.4f} | "
            f"AG Rotas: {len(historico_rotas)-1} ger. | "
            f"AG Equipe: {len(historico_equipe)-1} ger.")

    running = True
    while running:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                running = False
        pygame_desenhaMapa(tela, mapa, None, [], [], caminho_set,
                           custo_terreno + tempo_total, info_extra=info)
        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    main()
