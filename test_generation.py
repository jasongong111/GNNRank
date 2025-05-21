import numpy as np
import scipy.sparse as sp
import os

def generate_complicated_dataset(num_nodes=50, noise_level=0.1, num_3_cycles=3, num_4_cycles=2):
    """
    Generates a complicated dataset with a ground truth ranking, cycles, and noise.

    Args:
        num_nodes (int): Number of nodes in the graph.
        noise_level (float): Fraction of comparisons to flip against ground truth.
        num_3_cycles (int): Number of 3-node cycles to introduce.
        num_4_cycles (int): Number of 4-node cycles to introduce.

    Returns:
        None. Saves 'complicated_test_adj.npz' and 'complicated_test_labels.npy' in the 'data/' directory.
    """
    print(f"Generating a complicated dataset with {num_nodes} nodes...")

    # 1. Ground truth ranking (0 is best, num_nodes-1 is worst)
    true_ranks = np.arange(num_nodes)
    # Save labels
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(os.path.join(data_dir, 'complicated_test_labels.npy'), true_ranks)
    print(f"Saved ground truth labels to {os.path.join(data_dir, 'complicated_test_labels.npy')}")

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # 2. Basic comparisons based on true ranks
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if true_ranks[i] < true_ranks[j]:  # Node i is better than node j
                adj_matrix[i, j] = 1
            else:
                adj_matrix[j, i] = 1 # Node j is better than node i

    # 3. Introduce noise (flip some comparisons)
    num_comparisons = num_nodes * (num_nodes - 1) // 2 # Each pair compared once initially
    num_flips = int(noise_level * num_comparisons)
    
    flipped_pairs = set()
    for _ in range(num_flips):
        while True:
            u, v = np.random.choice(num_nodes, 2, replace=False)
            if tuple(sorted((u,v))) not in flipped_pairs: # ensure we don't flip a pair twice
                flipped_pairs.add(tuple(sorted((u,v))))
                break
        
        # Flip the existing comparison
        if adj_matrix[u,v] == 1: # u was beating v
            adj_matrix[u,v] = 0
            adj_matrix[v,u] = 1
        elif adj_matrix[v,u] == 1: # v was beating u
            adj_matrix[v,u] = 0
            adj_matrix[u,v] = 1
    print(f"Introduced noise by flipping {len(flipped_pairs)} comparisons.")

    # 4. Introduce cycles
    # Ensure nodes for cycles are distinct and don't overlap too much for simplicity
    nodes_used_in_cycles = set()

    # 3-cycles (A > B > C > A)
    for i in range(num_3_cycles):
        attempts = 0
        while attempts < 100:
            cycle_nodes = np.random.choice(num_nodes, 3, replace=False)
            if len(nodes_used_in_cycles.intersection(cycle_nodes)) == 0:
                nodes_used_in_cycles.update(cycle_nodes)
                break
            attempts += 1
        if attempts == 100:
            print(f"Warning: Could not find distinct nodes for 3-cycle {i+1}")
            continue
            
        a, b, c = cycle_nodes
        adj_matrix[a, b] = 1; adj_matrix[b, a] = 0 # a > b
        adj_matrix[b, c] = 1; adj_matrix[c, b] = 0 # b > c
        adj_matrix[c, a] = 1; adj_matrix[a, c] = 0 # c > a (completes cycle)
        print(f"Introduced 3-cycle: {a} > {b} > {c} > {a}")

    # 4-cycles (A > B > C > D > A)
    for i in range(num_4_cycles):
        attempts = 0
        while attempts < 100:
            cycle_nodes = np.random.choice(num_nodes, 4, replace=False)
            if len(nodes_used_in_cycles.intersection(cycle_nodes)) == 0:
                nodes_used_in_cycles.update(cycle_nodes)
                break
            attempts +=1
        if attempts == 100:
            print(f"Warning: Could not find distinct nodes for 4-cycle {i+1}")
            continue

        a, b, c, d = cycle_nodes
        adj_matrix[a, b] = 1; adj_matrix[b, a] = 0 # a > b
        adj_matrix[b, c] = 1; adj_matrix[c, b] = 0 # b > c
        adj_matrix[c, d] = 1; adj_matrix[d, c] = 0 # c > d
        adj_matrix[d, a] = 1; adj_matrix[a, d] = 0 # d > a (completes cycle)
        print(f"Introduced 4-cycle: {a} > {b} > {c} > {d} > {a}")

    # Convert to sparse matrix and save
    sparse_adj_matrix = sp.csr_matrix(adj_matrix)
    adj_file_path = os.path.join(data_dir, 'complicated_test_adj.npz')
    sp.save_npz(adj_file_path, sparse_adj_matrix)
    print(f"Saved sparse adjacency matrix to {adj_file_path}")
    print(f"Dataset generation complete. Density: {sparse_adj_matrix.nnz / (num_nodes * num_nodes):.4f}")

if __name__ == '__main__':
    generate_complicated_dataset()