# Multiple Allocation - Location-Routing Problem (inspirado em Campbell 1996)
# Fluxo: origem i -> hub k -> hub m -> destino j
# chi * distância(i, k) + alpha * distância(k, m) + delta * distância(m, j) --> chi, aplha e delta são os "pesos" das rotas

import os
import math
import matplotlib.pyplot as plt
from docplex.mp.model import Model

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


# Resolvendo: cria o modelo no Cplex > cria as variáveis > cria as restrições > constroi todos os custos e rotas e custos > opta pelo menor custo
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
    mdl = Model(name="AP_multiple_allocation_p_hub")

    # z[k] = 1 se k é hub
    z = mdl.binary_var_dict(nodes, name="z")

    # x[i,j,k,m] = 1 se fluxo i->j passa pelos hubs k e m
    x_keys = []

    for (i, j) in flow:
        for k in nodes:
            for m in nodes:
                x_keys.append((i, j, k, m))

    x = mdl.binary_var_dict(x_keys, name="x")

    # Restrição 1: abrir exatamente p hubs
    mdl.add_constraint(
        mdl.sum(z[k] for k in nodes) == p,
        ctname="number_of_hubs"
    )

    # Restrição 2: cada fluxo origem-destino deve escolher uma única rota via hubs
    for (i, j) in flow:
        mdl.add_constraint(
            mdl.sum(x[i, j, k, m] for k in nodes for m in nodes) == 1,
            ctname=f"assign_{i}_{j}"
        )

    # Restrição 3 compacta:
    # um nó k só pode aparecer como primeiro hub se z[k] = 1
    # um nó k só pode aparecer como segundo hub se z[k] = 1
    #
    # Essa versão usa bem menos restrições do que x[i,j,k,m] <= z[k] para todos k,m.
    for (i, j) in flow:
        for k in nodes:
            mdl.add_constraint(
                mdl.sum(x[i, j, k, m] for m in nodes) <= z[k],
                ctname=f"use_first_hub_{i}_{j}_{k}"
            )

            mdl.add_constraint(
                mdl.sum(x[i, j, m, k] for m in nodes) <= z[k],
                ctname=f"use_second_hub_{i}_{j}_{k}"
            )

    # Função objetivo:
    # custo = origem -> primeiro hub + hub -> hub com desconto + segundo hub -> destino
    objective = mdl.sum(
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

    mdl.minimize(objective)

    mdl.parameters.timelimit = time_limit

    print("\nResumo do modelo:")
    print(f"Variáveis totais: {mdl.number_of_variables}")
    print(f"Restrições totais: {mdl.number_of_constraints}")

    try:
        solution = mdl.solve(log_output=True)
    except Exception as error:
        print("\nErro ao resolver o modelo.")
        print("Possível causa: limite da versão gratuita do CPLEX.")
        print(f"Detalhe do erro: {error}")
        return mdl, [], {}

    if solution is None:
        print("\nNenhuma solução encontrada.")
        return mdl, [], {}

    selected_hubs = [
        k for k in nodes
        if z[k].solution_value > 0.5
    ]

    selected_routes = {}

    for (i, j) in flow:
        found_route = False

        for k in nodes:
            for m in nodes:
                if x[i, j, k, m].solution_value > 0.5:
                    selected_routes[(i, j)] = (k, m)
                    found_route = True
                    break

            if found_route:
                break

    print("\nSolução encontrada.")
    print("Status:", mdl.solve_details.status)
    print("Custo objetivo:", mdl.objective_value)
    print("Hubs escolhidos:", selected_hubs)

    print("\nRotas escolhidas:")
    for (i, j), (k, m) in selected_routes.items():
        print(f"{i} -> {j}: {i} -> hub {k} -> hub {m} -> {j}")

    return mdl, selected_hubs, selected_routes


#plotando a solução
def plot_solution(coords, selected_hubs, selected_routes, output_path):
    """
    Plota a malha da solução.
    """

    plt.figure(figsize=(10, 8))

    # Plotar os nós
    for node, (x_coord, y_coord) in coords.items():
        if node in selected_hubs:
            plt.scatter(x_coord, y_coord, marker="*", s=350)
            plt.text(x_coord + 1, y_coord + 1, f"H{node}", fontsize=11)
        else:
            plt.scatter(x_coord, y_coord, marker="o", s=90)
            plt.text(x_coord + 1, y_coord + 1, str(node), fontsize=9)

    # Plotar as arestas usadas pelas rotas
    plotted_edges = set()

    for (i, j), (k, m) in selected_routes.items():
        route_edges = [
            (i, k),
            (k, m),
            (m, j),
        ]

        for a, b in route_edges:
            if a == b:
                continue

            edge = tuple(sorted((a, b)))

            if edge in plotted_edges:
                continue

            plotted_edges.add(edge)

            xa, ya = coords[a]
            xb, yb = coords[b]

            plt.plot(
                [xa, xb],
                [ya, yb],
                linewidth=1,
                alpha=0.45,
            )

    plt.title("Solução AP - Multiple Allocation p-Hub Median")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"\nFigura salva em: {output_path}")

    plt.show()


def main():
    os.makedirs("outputs", exist_ok=True)

    # Caminho da instância AP
    instance_path = "data/APdata/20.3"

    #Até qual nó da instância está indo
    N_LIMIT = 5

    # Quantos hubs vamos escolher
    OVERRIDE_P = 2

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
            selected_hubs=selected_hubs,
            selected_routes=selected_routes,
            output_path="outputs/ap_solution.png",
        )


if __name__ == "__main__":
    main()