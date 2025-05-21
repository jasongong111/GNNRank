import numpy as np
import scipy.sparse as sp
import os

def generate_complicated_dataset(dataset_name="complicated_test", num_nodes=50, noise_level=0.1, num_3_cycles=3, num_4_cycles=2):
    """
    Generates a complicated dataset with a ground truth ranking, cycles, and noise.

    Args:
        dataset_name (str): The name of the dataset, used for the subfolder in 'data/'.
        num_nodes (int): Number of nodes in the graph.
        noise_level (float): Fraction of comparisons to flip against ground truth.
        num_3_cycles (int): Number of 3-node cycles to introduce.
        num_4_cycles (int): Number of 4-node cycles to introduce.

    Returns:
        None. Saves 'adj.npz' and 'labels.npy' in the 'data/{dataset_name}/' directory.
    """
    print(f"Generating a complicated dataset named '{dataset_name}' with {num_nodes} nodes...")

    # 1. Define directory paths
    data_root_dir = 'data'
    dataset_dir = os.path.join(data_root_dir, dataset_name)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created directory: {dataset_dir}")

    # 2. Ground truth ranking (0 is best, num_nodes-1 is worst)
    true_ranks = np.arange(num_nodes)
    labels_path = os.path.join(dataset_dir, 'labels.npy')
    np.save(labels_path, true_ranks)
    print(f"Saved ground truth labels to {labels_path}")

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # 3. Basic comparisons based on true ranks
    # Initially, create a complete tournament graph where i beats j if true_ranks[i] < true_ranks[j]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if true_ranks[i] < true_ranks[j]:  # Node i is better than node j
                adj_matrix[i, j] = 1
            else:
                adj_matrix[j, i] = 1 # Node j is better than node i (or vice-versa if ranks are equal, though not here)

    # 4. Introduce noise (flip some comparisons)
    # Count unique pairs for noise calculation
    num_unique_comparisons = 0
    for r in range(num_nodes):
        for c in range(r + 1, num_nodes):
            num_unique_comparisons +=1
            
    num_flips = int(noise_level * num_unique_comparisons)
    
    flipped_pairs = set() # To keep track of pairs we've already considered flipping
    flip_count_actual = 0

    for _ in range(num_flips):
        attempts = 0
        while attempts < 100: # Avoid infinite loops if finding a new pair is hard
            u, v = np.random.choice(num_nodes, 2, replace=False)
            # Ensure u < v to make the pair canonical for the set
            if u > v: u, v = v, u 
            if (u,v) not in flipped_pairs:
                flipped_pairs.add((u,v))
                # Flip the existing comparison
                if adj_matrix[u,v] == 1: # u was beating v
                    adj_matrix[u,v] = 0
                    adj_matrix[v,u] = 1
                elif adj_matrix[v,u] == 1: # v was beating u
                    adj_matrix[v,u] = 0
                    adj_matrix[u,v] = 1
                flip_count_actual +=1
                break
            attempts += 1

    print(f"Introduced noise by flipping {flip_count_actual} comparisons (target: {num_flips}).")

    # 5. Introduce cycles
    nodes_used_in_cycles = set()

    # 3-cycles (A > B > C > A)
    for i_cycle in range(num_3_cycles):
        attempts = 0
        selected_cycle_nodes = []
        while attempts < 100:
            potential_cycle_nodes = np.random.choice(num_nodes, 3, replace=False)
            if len(nodes_used_in_cycles.intersection(potential_cycle_nodes)) == 0:
                nodes_used_in_cycles.update(potential_cycle_nodes)
                selected_cycle_nodes = potential_cycle_nodes
                break
            attempts += 1
        
        if not list(selected_cycle_nodes):
            print(f"Warning: Could not find distinct nodes for 3-cycle {i_cycle+1} after {attempts} attempts. Skipping this cycle.")
            continue
            
        a, b, c = selected_cycle_nodes
        adj_matrix[a, b] = 1; adj_matrix[b, a] = 0 # a > b
        adj_matrix[b, c] = 1; adj_matrix[c, b] = 0 # b > c
        adj_matrix[c, a] = 1; adj_matrix[a, c] = 0 # c > a (completes cycle)
        print(f"Introduced 3-cycle: {a} > {b} > {c} > {a}")

    # 4-cycles (A > B > C > D > A)
    for i_cycle in range(num_4_cycles):
        attempts = 0
        selected_cycle_nodes = []
        while attempts < 100:
            potential_cycle_nodes = np.random.choice(num_nodes, 4, replace=False)
            if len(nodes_used_in_cycles.intersection(potential_cycle_nodes)) == 0:
                nodes_used_in_cycles.update(potential_cycle_nodes)
                selected_cycle_nodes = potential_cycle_nodes
                break
            attempts +=1

        if not list(selected_cycle_nodes):
            print(f"Warning: Could not find distinct nodes for 4-cycle {i_cycle+1} after {attempts} attempts. Skipping this cycle.")
            continue

        a, b, c, d = selected_cycle_nodes
        adj_matrix[a, b] = 1; adj_matrix[b, a] = 0 # a > b
        adj_matrix[b, c] = 1; adj_matrix[c, b] = 0 # b > c
        adj_matrix[c, d] = 1; adj_matrix[d, c] = 0 # c > d
        adj_matrix[d, a] = 1; adj_matrix[a, d] = 0 # d > a (completes cycle)
        print(f"Introduced 4-cycle: {a} > {b} > {c} > {d} > {a}")

    # Convert to sparse matrix and save
    sparse_adj_matrix = sp.csr_matrix(adj_matrix)
    adj_file_path = os.path.join(dataset_dir, 'adj.npz')
    sp.save_npz(adj_file_path, sparse_adj_matrix)
    print(f"Saved sparse adjacency matrix to {adj_file_path}")
    print(f"Dataset generation complete. Adjacency matrix density: {sparse_adj_matrix.nnz / (num_nodes * num_nodes):.4f}")

if __name__ == '__main__':
    generate_complicated_dataset() # Uses default dataset_name="complicated_test"