import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from configuracao import INSTANCE_PATH, N_LIMIT, OVERRIDE_P


def load_ap_instance_raw(file_path, n_limit=None, override_p=None):
    """
    Lê uma instância AP e organiza os dados em estruturas mais fáceis de visualizar.
    """

    with open(file_path, "r") as file:
        tokens = file.read().split()

    idx = 0

    n = int(tokens[idx])
    idx += 1

    nodes = list(range(1, n + 1))

    coords = {}

    for i in nodes:
        x_coord = float(tokens[idx])
        y_coord = float(tokens[idx + 1])
        idx += 2
        coords[i] = (x_coord, y_coord)

    flow_matrix = []

    for i in nodes:
        row = []

        for j in nodes:
            value = float(tokens[idx])
            idx += 1
            row.append(value)

        flow_matrix.append(row)

    p = int(tokens[idx])
    idx += 1

    delta = float(tokens[idx])
    idx += 1

    alpha = float(tokens[idx])
    idx += 1

    chi = float(tokens[idx])
    idx += 1

    if n_limit is not None:
        n = min(n_limit, n)
        nodes = list(range(1, n + 1))
        coords = {node: coords[node] for node in nodes}
        flow_matrix = [row[:n] for row in flow_matrix[:n]]

    if override_p is not None:
        p = override_p
    else:
        p = min(p, n)

    return n, nodes, coords, flow_matrix, p, delta, alpha, chi


def create_visual_outputs(
    file_path,
    n_limit=None,
    override_p=None,
    output_folder="outputs/instance_view",
):
    os.makedirs(output_folder, exist_ok=True)

    n, nodes, coords, flow_matrix, p, delta, alpha, chi = load_ap_instance_raw(
        file_path,
        n_limit=n_limit,
        override_p=override_p,
    )

    coords_df = pd.DataFrame([
        {
            "node": node,
            "x": coords[node][0],
            "y": coords[node][1],
        }
        for node in nodes
    ])

    flow_df = pd.DataFrame(
        flow_matrix,
        index=[f"from_{i}" for i in nodes],
        columns=[f"to_{j}" for j in nodes],
    )

    distance_matrix = []

    for i in nodes:
        row = []

        for j in nodes:
            xi, yi = coords[i]
            xj, yj = coords[j]
            distance = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            row.append(distance)

        distance_matrix.append(row)

    distance_df = pd.DataFrame(
        distance_matrix,
        index=[f"from_{i}" for i in nodes],
        columns=[f"to_{j}" for j in nodes],
    )

    coords_df.to_csv(f"{output_folder}/coordenadas_nos.csv", index=False)
    flow_df.to_csv(f"{output_folder}/matriz_fluxos.csv")
    distance_df.to_csv(f"{output_folder}/matriz_distancias.csv")

    print("\nResumo da instância")
    print("-" * 40)
    print(f"Arquivo: {file_path}")
    print(f"Número de nós: {n}")
    print(f"Número de hubs p: {p}")
    print(f"delta: {delta}")
    print(f"alpha: {alpha}")
    print(f"chi: {chi}")
    print(f"Total de valores de fluxo: {n * n}")
    print(f"Fluxo total: {flow_df.values.sum():.4f}")
    print(f"Maior fluxo: {flow_df.values.max():.4f}")
    print(f"Menor fluxo: {flow_df.values.min():.4f}")

    print("\nPrimeiros nós:")
    print(coords_df.head(10).to_string(index=False))

    print("\nPrimeira parte da matriz de fluxos:")
    print(flow_df.iloc[:8, :8].to_string())

    print("\nArquivos CSV gerados:")
    print(f"- {output_folder}/coordenadas_nos.csv")
    print(f"- {output_folder}/matriz_fluxos.csv")
    print(f"- {output_folder}/matriz_distancias.csv")

    plot_nodes(coords_df, output_folder)
    plot_flow_heatmap(flow_df, output_folder)
    plot_distance_heatmap(distance_df, output_folder)


def plot_nodes(coords_df, output_folder):
    """
    Plota a posição dos nós da instância.
    """

    plt.figure(figsize=(10, 8))

    plt.scatter(coords_df["x"], coords_df["y"], s=80)

    for _, row in coords_df.iterrows():
        plt.text(
            row["x"] + 300,
            row["y"] + 300,
            str(int(row["node"])),
            fontsize=9,
        )

    plt.title("Nós da instância AP")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True)
    plt.tight_layout()

    output_path = f"{output_folder}/nos_instancia.png"
    plt.savefig(output_path, dpi=300)

    print(f"- {output_path}")

    plt.show()


def plot_flow_heatmap(flow_df, output_folder):
    """
    Plota um mapa de calor da matriz de fluxos.
    """

    plt.figure(figsize=(10, 8))

    plt.imshow(flow_df.values, aspect="auto")
    plt.colorbar(label="Fluxo")

    plt.title("Mapa de calor da matriz de fluxos")
    plt.xlabel("Destino")
    plt.ylabel("Origem")

    plt.xticks(
        ticks=range(len(flow_df.columns)),
        labels=[col.replace("to_", "") for col in flow_df.columns],
        rotation=90,
    )

    plt.yticks(
        ticks=range(len(flow_df.index)),
        labels=[idx.replace("from_", "") for idx in flow_df.index],
    )

    plt.tight_layout()

    output_path = f"{output_folder}/heatmap_fluxos.png"
    plt.savefig(output_path, dpi=300)

    print(f"- {output_path}")

    plt.show()


def plot_distance_heatmap(distance_df, output_folder):
    """
    Plota um mapa de calor da matriz de distâncias.
    """

    plt.figure(figsize=(10, 8))

    plt.imshow(distance_df.values, aspect="auto")
    plt.colorbar(label="Distância")

    plt.title("Mapa de calor da matriz de distâncias")
    plt.xlabel("Destino")
    plt.ylabel("Origem")

    plt.xticks(
        ticks=range(len(distance_df.columns)),
        labels=[col.replace("to_", "") for col in distance_df.columns],
        rotation=90,
    )

    plt.yticks(
        ticks=range(len(distance_df.index)),
        labels=[idx.replace("from_", "") for idx in distance_df.index],
    )

    plt.tight_layout()

    output_path = f"{output_folder}/heatmap_distancias.png"
    plt.savefig(output_path, dpi=300)

    print(f"- {output_path}")

    plt.show()


def main():
    create_visual_outputs(
        file_path=INSTANCE_PATH,
        n_limit=N_LIMIT,
        override_p=OVERRIDE_P,
        output_folder="outputs/instance_view",
    )


if __name__ == "__main__":
    main()
