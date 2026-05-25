# Multiple Allocation - Location-Routing Problem (inspirado em Campbell 1996)
# Fluxo: origem i -> hub k -> hub m -> destino j
# chi * distância(i, k) + alpha * distância(k, m) + delta * distância(m, j) --> chi, aplha e delta são os "pesos" das rotas

import os
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import gurobipy as gp
from gurobipy import GRB

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
def solve_multiple_allocation_p_hub(
    nodes,
    flow,
    distance,
    p,
    alpha,
    chi,
    delta,
    time_limit=300,
):
    try:
        mdl = gp.Model("AP_multiple_allocation_p_hub")
    except gp.GurobiError as error:
        print("\nErro ao iniciar o Gurobi.")
        print("Verifique a instalação e a licença do Gurobi.")
        print(f"Detalhe do erro: {error}")
        return None, [], {}

    # z[k] = 1 se k é hub
    z = mdl.addVars(nodes, vtype=GRB.BINARY, name="z")

    # x[i,j,k,m] = 1 se fluxo i->j passa pelos hubs k e m
    x_keys = []

    for (i, j) in flow:
        for k in nodes:
            for m in nodes:
                x_keys.append((i, j, k, m))

    x = mdl.addVars(x_keys, vtype=GRB.BINARY, name="x")

    # Restrição 1: abrir exatamente p hubs
    mdl.addConstr(
        gp.quicksum(z[k] for k in nodes) == p,
        name="number_of_hubs"
    )

    # Restrição 2: cada fluxo origem-destino deve escolher uma única rota via hubs
    for (i, j) in flow:
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
    objective = gp.quicksum(
        flow[(i, j)]
        * (
            chi * distance[(i, k)]
            + alpha * distance[(k, m)]
            + delta * distance[(m, j)]
        )
        / 1000
        * x[i, j, k, m]
        for (i, j, k, m) in x_keys
    )

    mdl.setObjective(objective, GRB.MINIMIZE)

    mdl.Params.TimeLimit = time_limit
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
        return mdl, [], {}

    if mdl.SolCount == 0:
        print("\nNenhuma solução encontrada.")
        return mdl, [], {}

    selected_hubs = [
        k for k in nodes
        if z[k].X > 0.5
    ]

    selected_routes = {}

    for (i, j) in flow:
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

    print("\nRotas escolhidas:")
    for (i, j), (k, m) in selected_routes.items():
        print(f"{i} -> {j}: {i} -> hub {k} -> hub {m} -> {j}")

    return mdl, selected_hubs, selected_routes


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

    # Caminho da instância AP
    instance_path = "data/APdata/20.3"

    #Até qual nó da instância está indo
    N_LIMIT = 20

    # Quantos hubs vamos escolher
    OVERRIDE_P = 3

    nodes, coords, flow, distance, p, alpha, chi, delta = load_ap_instance(
        file_path=instance_path,
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
        time_limit=300,
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
