# modification of syt3's run_leiden.py that
# sets n_iterations to 5 and seed to 1234
# Author: Minhyuk Park
# 2/19/2023
# Modified to support CSV files with headers
# Modified by: Vikram Ramavarapu with the help of Claude (1/14/2026)

import leidenalg
import igraph
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for running leiden with CPM.')
    # Todo: Can we make the input arguments similar to runleiden to maintain
    #  consistency. For e.g. "-i" is input file in runleiden and here it is
    #  "number of iterations" and "-n" is the input file
    parser.add_argument(
        '-i', metavar='ip_net', type=str, required=True,
        help='input network edge-list path (CSV with header)'
        )
    parser.add_argument(
        '-r', metavar='resolution', type=float, required=True,
        help='resolution parameter (gamma)'
        )
    parser.add_argument(
        '-o', metavar='output', type=str, required=True,
        help='output membership path'
        )
    parser.add_argument(
        '-n', metavar='n_iterations', type=int, required=True,
        help='number of iterations'
        )
    args = parser.parse_args()

    # Read CSV with header
    print(f"Reading edgelist from {args.i}...")
    edges_df = pd.read_csv(args.i)

    # Assume first two columns are source and target
    # Convert to list of tuples
    edges = list(zip(edges_df.iloc[:, 0], edges_df.iloc[:, 1]))

    print(f"Creating graph with {len(edges)} edges...")
    # Create graph from edge list
    net = igraph.Graph.TupleList(edges, directed=False)

    print(f"Running Leiden algorithm (CPM) with resolution={args.r}, n_iterations={args.n}...")
    partition = leidenalg.find_partition(
        net, leidenalg.CPMVertexPartition, resolution_parameter=args.r,
        seed=1234, n_iterations=args.n
        )

    print(f"Quality (CPM): {partition.quality():.4f}")
    print(f"Number of communities: {len(set(partition.membership))}")

    print(f"Writing membership to {args.o}...")
    with open(args.o, "w") as f:
        f.write("node\tcommunity\n")
        for n, m in enumerate(partition.membership):
            f.write(f"{net.vs[n]['name']}\t{m}\n")

    print("Done!")
