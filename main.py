# Multiple Allocation - Location-Routing Problem (inspirado em Campbell 1996)
# Fluxo: origem i -> hub k -> hub m -> destino j
# chi * distância(i, k) + alpha * distância(k, m) + delta * distância(m, j) --> chi, aplha e delta são os "pesos" das rotas

import os
import math
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import gurobipy as gp
from gurobipy import GRB
from configuracao import (
    INSTANCE_PATH,
    N_LIMIT,
    OVERRIDE_P,
    TIME_LIMIT_SECONDS,
)

# FUNÇÕES BASE
# Lê a instância AP, reduz os nós se necessário, calcula as distâncias entre eles e retorna os dados prontos para o modelo de otimização.
def load_ap_instance(file_path, n_limit=None, override_p=None):
    with open(file_path, "r") as file:
        tokens = file.read().split()

    idx = 0
    original_n = int(tokens[idx])
    idx += 1
    original_nodes = list(range(1, original_n + 1))
    original_coords = {}

    for i in original_nodes:
        x_coord = float(tokens[idx])
        y_coord = float(tokens[idx + 1])
        idx += 2
        original_coords[i] = (x_coord, y_coord)

    original_flow_matrix = {}

    for i in original_nodes:
        for j in original_nodes:
            original_flow_matrix[(i, j)] = float(tokens[idx])
            idx += 1

    original_p = int(tokens[idx])
    idx += 1

    delta = float(tokens[idx])
    idx += 1

    alpha = float(tokens[idx])
    idx += 1

    chi = float(tokens[idx])
    idx += 1

    if n_limit is None:
        n = original_n
    else:
        n = min(n_limit, original_n)

    nodes = list(range(1, n + 1))

    coords = {
        i: original_coords[i]
        for i in nodes
    }

    flow = {}

    for i in nodes:
        for j in nodes:
            value = original_flow_matrix[(i, j)]

            if i != j and value > 0:
                flow[(i, j)] = value

    if override_p is not None:
        p = override_p
    else:
        p = min(original_p, n)

    distance = {}

    for i in nodes:
        for j in nodes:
            xi, yi = coords[i]
            xj, yj = coords[j]

            distance[(i, j)] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    # print("Instância carregada com sucesso.")
    # print(f"Nós originais: {original_n}")
    # print(f"Nós usados: {n}")
    # print(f"Hubs usados p: {p}")
    # print(f"alpha: {alpha}, chi: {chi}, delta: {delta}")
    # print(f"Quantidade de fluxos considerados: {len(flow)}")

    return nodes, coords, flow, distance, p, alpha, chi, delta


# Resolvendo: cria o modelo no Gurobi > cria as variáveis > cria as restrições > constroi todos os custos e rotas e custos > opta pelo menor custo
# Calculando quase tudo e escolhendo dessa forma (cresce muito rapidamente)
def write_execution_log(log_path, instance_path, nodes, flow, p, event, elapsed=None, detail=None):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    fields = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"evento={event}",
        f"instancia={instance_path}",
        f"nos={len(nodes)}",
        f"hubs={p}",
        f"fluxos={len(flow)}",
    ]

    if elapsed is not None:
        fields.append(f"tempo_total_s={elapsed:.3f}")

    if detail is not None:
        fields.append(detail)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(" | ".join(fields) + "\n")


class ExecutionTimeLimitReached(Exception):
    pass


def _solve_multiple_allocation_p_hub(
    nodes,
    flow,
    distance,
    p,
    alpha,
    chi,
    delta,
    instance_path,
    time_limit=300,
    execution_log_path="Logs/gurobi_execucoes.log",
):
    start_time = time.perf_counter()
    estimated_route_vars = len(flow) * len(nodes) * len(nodes)
    write_execution_log(
        execution_log_path,
        instance_path,
        nodes,
        flow,
        p,
        event="INICIO",
        detail=(
            f"variaveis_rota_estimadas={estimated_route_vars};"
            f"limite_tempo_s={time_limit}"
        ),
    )
    print(f"\nInstância em execução: {instance_path}")
    print(f"Variáveis de rota estimadas: {estimated_route_vars}")

    def finish_log(event, detail=None):
        elapsed = time.perf_counter() - start_time
        write_execution_log(
            execution_log_path,
            instance_path,
            nodes,
            flow,
            p,
            event=event,
            elapsed=elapsed,
            detail=detail,
        )
        print(f"Tempo total da execução: {elapsed:.3f} s")
        print(f"Log de execução: {execution_log_path}")

    def remaining_time(stage):
        elapsed = time.perf_counter() - start_time
        remaining = time_limit - elapsed

        if remaining <= 0:
            finish_log(
                "TIMEOUT_TOTAL",
                detail=f"etapa={stage};limite_tempo_s={time_limit}",
            )
            raise ExecutionTimeLimitReached(
                f"Tempo limite total atingido durante: {stage}."
            )

        return remaining

    try:
        mdl = gp.Model("AP_multiple_allocation_p_hub")
    except gp.GurobiError as error:
        print("\nErro ao iniciar o Gurobi.")
        print("Verifique a instalação e a licença do Gurobi.")
        print(f"Detalhe do erro: {error}")
        finish_log("ERRO_INICIALIZACAO", detail=f"erro={error}")
        return None, [], {}

    os.makedirs("Logs", exist_ok=True)
    mdl.Params.LogFile = "Logs/gurobi.log"
    remaining_time("criacao_variaveis_hub")

    # z[k] = 1 se k é hub
    z = mdl.addVars(nodes, vtype=GRB.BINARY, name="z")

    # x[i,j,k,m] = 1 se fluxo i->j passa pelos hubs k e m
    x = gp.tupledict()
    flow_pairs = list(flow)
    batch_size = 25

    for batch_start in range(0, len(flow_pairs), batch_size):
        remaining_time("criacao_variaveis_rota")
        batch_pairs = flow_pairs[batch_start:batch_start + batch_size]
        batch_keys = [
            (i, j, k, m)
            for (i, j) in batch_pairs
            for k in nodes
            for m in nodes
        ]
        x.update(mdl.addVars(batch_keys, vtype=GRB.BINARY, name="x"))

    # Restrição 1: abrir exatamente p hubs
    remaining_time("criacao_restricoes")
    mdl.addConstr(
        gp.quicksum(z[k] for k in nodes) == p,
        name="number_of_hubs"
    )

    # Restrição 2: cada fluxo origem-destino deve escolher uma única rota via hubs
    for (i, j) in flow:
        remaining_time("criacao_restricoes_atribuicao")
        mdl.addConstr(
            gp.quicksum(x[i, j, k, m] for k in nodes for m in nodes) == 1,
            name=f"assign_{i}_{j}"
        )

    # Restrição 3 compacta:
    # um nó k só pode aparecer como primeiro hub se z[k] = 1
    # um nó k só pode aparecer como segundo hub se z[k] = 1
    #
    # Essa versão usa bem menos restrições do que x[i,j,k,m] <= z[k] para todos k,m.
    for (i, j) in flow:
        remaining_time("criacao_restricoes_hubs")
        for k in nodes:
            mdl.addConstr(
                gp.quicksum(x[i, j, k, m] for m in nodes) <= z[k],
                name=f"use_first_hub_{i}_{j}_{k}"
            )

            mdl.addConstr(
                gp.quicksum(x[i, j, m, k] for m in nodes) <= z[k],
                name=f"use_second_hub_{i}_{j}_{k}"
            )

    # Função objetivo:
    # custo = origem -> primeiro hub + hub -> hub com desconto + segundo hub -> destino
    objective = gp.LinExpr()

    for position, ((i, j, k, m), variable) in enumerate(x.items()):
        if position % 10000 == 0:
            remaining_time("construcao_objetivo")

        coefficient = (
            flow[(i, j)]
            * (
                chi * distance[(i, k)]
                + alpha * distance[(k, m)]
                + delta * distance[(m, j)]
            )
            / 1000
        )
        objective.addTerms(coefficient, variable)

    mdl.setObjective(objective, GRB.MINIMIZE)

    mdl.Params.TimeLimit = remaining_time("inicio_otimizacao")
    mdl.update()

    print("\nResumo do modelo:")
    print(f"Variáveis totais: {mdl.NumVars}")
    print(f"Restrições totais: {mdl.NumConstrs}")

    try:
        mdl.optimize()
    except gp.GurobiError as error:
        print("\nErro ao resolver o modelo.")
        print("Verifique a instalação e a licença do Gurobi.")
        print(f"Detalhe do erro: {error}")
        finish_log("ERRO_OTIMIZACAO", detail=f"erro={error}")
        return mdl, [], {}

    if mdl.SolCount == 0:
        print("\nNenhuma solução encontrada.")
        finish_log(
            "SEM_SOLUCAO",
            detail=f"status={mdl.Status};tempo_gurobi_s={mdl.Runtime:.3f}",
        )
        return mdl, [], {}

    selected_hubs = [
        k for k in nodes
        if z[k].X > 0.5
    ]

    selected_routes = {}

    for (i, j) in flow:
        remaining_time("leitura_solucao")
        found_route = False

        for k in nodes:
            for m in nodes:
                if x[i, j, k, m].X > 0.5:
                    selected_routes[(i, j)] = (k, m)
                    found_route = True
                    break

            if found_route:
                break

    print("\nSolução encontrada.")
    print("Status:", mdl.Status)
    print("Custo objetivo:", mdl.ObjVal)
    print("Hubs escolhidos:", selected_hubs)
    finish_log(
        "SOLUCAO",
        detail=(
            f"status={mdl.Status};tempo_gurobi_s={mdl.Runtime:.3f};"
            f"objetivo={mdl.ObjVal:.6f};hubs_escolhidos={selected_hubs}"
        ),
    )

    print("\nRotas escolhidas:")
    for (i, j), (k, m) in selected_routes.items():
        print(f"{i} -> {j}: {i} -> hub {k} -> hub {m} -> {j}")

    return mdl, selected_hubs, selected_routes


def solve_multiple_allocation_p_hub(*args, **kwargs):
    try:
        return _solve_multiple_allocation_p_hub(*args, **kwargs)
    except ExecutionTimeLimitReached as error:
        print(f"\n{error}")
        return None, [], {}


#plotando a solução
def plot_solution(coords, flow, selected_hubs, selected_routes, output_path):
    """
    Plota a rede com hubs destacados e largura proporcional ao fluxo agregado.
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    hub_set = set(selected_hubs)
    edge_flows = {}

    for (i, j), (k, m) in selected_routes.items():
        for a, b in [(i, k), (k, m), (m, j)]:
            if a == b:
                continue

            edge = tuple(sorted((a, b)))
            edge_flows[edge] = edge_flows.get(edge, 0) + flow[(i, j)]

    min_flow = min(edge_flows.values(), default=0)
    max_flow = max(edge_flows.values(), default=1)
    flow_norm = Normalize(vmin=min_flow, vmax=max_flow)
    flow_cmap = LinearSegmentedColormap.from_list(
        "fluxo_azul",
        ["#d9e3ee", "#93abc1", "#486c8c", "#163a5b"],
    )

    for (a, b), edge_flow in edge_flows.items():
        xa, ya = coords[a]
        xb, yb = coords[b]
        scaled_flow = math.sqrt(edge_flow / max_flow)
        is_hub_link = a in hub_set and b in hub_set

        ax.plot(
            [xa, xb],
            [ya, yb],
            color=flow_cmap(flow_norm(edge_flow)),
            linewidth=0.6 + 5.6 * scaled_flow,
            alpha=0.9 if is_hub_link else 0.72,
            solid_capstyle="round",
            zorder=2 if is_hub_link else 1,
        )

    for node, (x_coord, y_coord) in coords.items():
        if node in hub_set:
            ax.scatter(
                x_coord, y_coord, marker="o", s=250, color="#d62728",
                edgecolors="black", linewidths=1.2, zorder=4,
            )
            ax.annotate(
                f"H{node}", (x_coord, y_coord), xytext=(0, 13),
                textcoords="offset points", ha="center", fontsize=10,
                fontweight="bold", color="#8b0000", zorder=5,
            )
        else:
            ax.scatter(
                x_coord, y_coord, marker="o", s=75, color="#2166f3",
                edgecolors="white", linewidths=0.8, zorder=3,
            )
            ax.annotate(
                str(node), (x_coord, y_coord), xytext=(5, 5),
                textcoords="offset points", fontsize=8, color="#222222",
                zorder=5,
            )

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=9, label="Hub"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2166f3",
               markeredgecolor="white", markersize=7, label="Spoke"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.96,
    )
    scalar_mappable = ScalarMappable(norm=flow_norm, cmap=flow_cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02, fraction=0.045)
    colorbar.set_label("Fluxo agregado na conexão")
    ax.set_title("Solução AP - Rede hub-and-spoke", fontsize=14, pad=12)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigura salva em: {output_path}")

    plt.show()


def main():
    os.makedirs("outputs", exist_ok=True)

    nodes, coords, flow, distance, p, alpha, chi, delta = load_ap_instance(
        file_path=INSTANCE_PATH,
        n_limit=N_LIMIT,
        override_p=OVERRIDE_P,
    )

    model, selected_hubs, selected_routes = solve_multiple_allocation_p_hub(
        nodes=nodes,
        flow=flow,
        distance=distance,
        p=p,
        alpha=alpha,
        chi=chi,
        delta=delta,
        instance_path=INSTANCE_PATH,
        time_limit=TIME_LIMIT_SECONDS,
    )

    if selected_hubs:
        plot_solution(
            coords=coords,
            flow=flow,
            selected_hubs=selected_hubs,
            selected_routes=selected_routes,
            output_path="outputs/ap_solution.png",
        )


if __name__ == "__main__":
    main()

    #stream_lite - biblio p visualização (verse faz sentido usar junto a um mapa)
    #dif hub spoke por cor
    #largura das linhas de conexão proporcionais ao fluxo
    #iterativo com usuario para ajustar parametros
    #imagem de resultado
    #aplicação web - 
