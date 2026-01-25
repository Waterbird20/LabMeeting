"""
PPTMixer: Python implementation for detecting genuine multipartite entanglement

Based on: Bastian Jungnitsch, Tobias Moroder1, and Otfried Gühne
Phys. Rev. Lett. 106, 190502 (2011)
"""

import numpy as np
import cvxpy as cp
from itertools import combinations
from typing import Tuple, Optional, List


def partial_transpose(rho: np.ndarray, partition: List[int], 
                      dimensions: Optional[List[int]] = None) -> np.ndarray:
    """
    Compute partial transpose of density matrix rho.
    
    Parameters
    ----------
    rho : np.ndarray
        Density matrix (must be square and Hermitian)
    partition : List[int]
        Binary list indicating which subsystems to transpose.
        E.g., [1, 0, 1] means transpose subsystems 0 and 2.
    dimensions : List[int], optional
        Dimensions of each subsystem. If None, assumes qubits (dim=2).
    
    Returns
    -------
    np.ndarray
        Partially transposed density matrix
    """
    n = len(partition)
    
    if dimensions is None:
        # Assume qubits
        dimensions = [2] * n
    
    total_dim = int(np.prod(dimensions))
    
    if rho.shape[0] != total_dim or rho.shape[1] != total_dim:
        raise ValueError("Density matrix dimensions don't match specified subsystem dimensions")
    
    # Check Hermiticity
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("Input matrix is not Hermitian")
    
    rho_pt = np.zeros_like(rho, dtype=complex)
    
    # Helper function to convert flat index to multi-index
    def flat_to_multi(idx, dims):
        multi = []
        for d in reversed(dims):
            multi.append(idx % d)
            idx //= d
        return list(reversed(multi))
    
    # Helper function to convert multi-index to flat index
    def multi_to_flat(multi, dims):
        idx = 0
        for i, (m, d) in enumerate(zip(multi, dims)):
            idx = idx * d + m
        return idx
    
    # Compute partial transpose
    for i in range(total_dim):
        for j in range(total_dim):
            multi_i = flat_to_multi(i, dimensions)
            multi_j = flat_to_multi(j, dimensions)
            
            # Swap indices for systems in partition
            new_multi_i = list(multi_i)
            new_multi_j = list(multi_j)
            
            for k in range(n):
                if partition[k] == 1:
                    new_multi_i[k], new_multi_j[k] = multi_j[k], multi_i[k]
            
            new_i = multi_to_flat(new_multi_i, dimensions)
            new_j = multi_to_flat(new_multi_j, dimensions)
            
            rho_pt[new_i, new_j] = rho[i, j]
    
    return rho_pt


def get_all_bipartitions(n: int) -> List[List[int]]:
    """
    Generate all inequivalent bipartitions of n systems.
    
    A bipartition M|complement(M) is equivalent to complement(M)|M,
    so we only need to check half of them.
    
    Parameters
    ----------
    n : int
        Number of subsystems
    
    Returns
    -------
    List[List[int]]
        List of binary vectors representing bipartitions
    """
    bipartitions = []
    
    # Loop from 1 to 2^(n-1) - 1
    # This ensures neither M nor complement(M) is empty
    # and avoids equivalent partitions
    for m in range(1, 2**(n-1)):
        partition = []
        for i in range(n):
            partition.append((m >> (n - 1 - i)) & 1)
        bipartitions.append(partition)
    
    return bipartitions


def fdecwit(rho: np.ndarray, dimensions: Optional[List[int]] = None, 
            verbose: bool = False) -> Tuple[float, np.ndarray]:
    """
    Find the optimal fully decomposable witness for genuine multipartite entanglement.
    
    Solves the SDP:
        min Tr(W @ rho)
        s.t. Tr(W) = 1
             For all bipartitions M: W = P_M + Q_M^{T_M}
             P_M >= 0, Q_M >= 0
    
    Parameters
    ----------
    rho : np.ndarray
        Density matrix to test
    dimensions : List[int], optional
        Dimensions of subsystems. If None, assumes qubits.
    verbose : bool
        If True, print solver output
    
    Returns
    -------
    Tuple[float, np.ndarray]
        (minimum_value, witness)
        If minimum_value < 0, the state is genuinely multipartite entangled.
        witness is the optimal W, or zeros if not detected.
    """
    d = rho.shape[0]
    
    if dimensions is None:
        n = int(np.log2(d))
        if 2**n != d:
            raise ValueError("For non-qubit systems, please specify dimensions")
        dimensions = [2] * n
    else:
        n = len(dimensions)
        if np.prod(dimensions) != d:
            raise ValueError("Dimensions don't match density matrix size")
    
    # Check inputs
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("Density matrix is not Hermitian")
    
    # Get all bipartitions
    bipartitions = get_all_bipartitions(n)
    
    # Define SDP variables
    W = cp.Variable((d, d), hermitian=True)
    
    constraints = []
    
    # Trace normalization
    constraints.append(cp.trace(W) == 1)
    
    # For each bipartition, we need W = P_M + Q_M^{T_M}
    # with P_M >= 0 and Q_M >= 0
    P_vars = {}
    
    for idx, partition in enumerate(bipartitions):
        P = cp.Variable((d, d), hermitian=True)
        P_vars[idx] = P
        
        # P_M >= 0
        constraints.append(P >> 0)
        
        # Q_M = (W - P_M)^{T_M} >= 0
        # We need to express this constraint
        # Since partial transpose is a linear operation, we can handle it
        
        # Create the partial transpose of (W - P)
        # This is tricky in cvxpy because partial transpose is not a standard atom
        # We'll use a parameterized approach
        
        Q = cp.Variable((d, d), hermitian=True)
        constraints.append(Q >> 0)
        
        # Constraint: W - P = Q^{T_M}
        # We need to implement partial transpose as linear constraints
        # For each element, we set up the appropriate equality
        
        for i in range(d):
            for j in range(d):
                # Find where (i,j) maps under partial transpose
                multi_i = []
                multi_j = []
                temp_i, temp_j = i, j
                for dim in reversed(dimensions):
                    multi_i.append(temp_i % dim)
                    multi_j.append(temp_j % dim)
                    temp_i //= dim
                    temp_j //= dim
                multi_i = list(reversed(multi_i))
                multi_j = list(reversed(multi_j))
                
                # Apply partial transpose
                new_multi_i = list(multi_i)
                new_multi_j = list(multi_j)
                for k in range(n):
                    if partition[k] == 1:
                        new_multi_i[k], new_multi_j[k] = multi_j[k], multi_i[k]
                
                # Convert back to flat indices
                new_i, new_j = 0, 0
                for k in range(n):
                    new_i = new_i * dimensions[k] + new_multi_i[k]
                    new_j = new_j * dimensions[k] + new_multi_j[k]
                
                # Constraint: (W - P)[i,j] = Q[new_i, new_j]
                if j >= i:  # Only need upper triangle due to Hermitian
                    constraints.append(W[i, j] - P[i, j] == Q[new_i, new_j])
    
    # Objective: minimize Tr(W @ rho)
    objective = cp.Minimize(cp.real(cp.trace(W @ rho)))
    
    # Solve
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.SCS, verbose=verbose)
    except:
        try:
            prob.solve(solver=cp.CVXOPT, verbose=verbose)
        except:
            prob.solve(verbose=verbose)
    
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print(f"Warning: Solver status = {prob.status}")
    
    min_val = prob.value if prob.value is not None else 0
    
    precision = 1e-8
    if min_val < -precision:
        witness = W.value
    else:
        witness = np.zeros((d, d))
    
    return min_val, witness


def fpptwit(rho: np.ndarray, dimensions: Optional[List[int]] = None,
            verbose: bool = False) -> Tuple[float, np.ndarray]:
    """
    Find the optimal fully PPT witness for genuine multipartite entanglement.
    
    This is simpler than fdecwit: we require W^{T_M} >= 0 for all M,
    which corresponds to P_M = 0 in the fully decomposable case.
    
    Solves the SDP:
        min Tr(W @ rho)
        s.t. Tr(W) = 1
             For all bipartitions M: W^{T_M} >= 0
    
    Parameters
    ----------
    rho : np.ndarray
        Density matrix to test
    dimensions : List[int], optional
        Dimensions of subsystems. If None, assumes qubits.
    verbose : bool
        If True, print solver output
    
    Returns
    -------
    Tuple[float, np.ndarray]
        (minimum_value, witness)
    """
    d = rho.shape[0]
    
    if dimensions is None:
        n = int(np.log2(d))
        if 2**n != d:
            raise ValueError("For non-qubit systems, please specify dimensions")
        dimensions = [2] * n
    else:
        n = len(dimensions)
        if np.prod(dimensions) != d:
            raise ValueError("Dimensions don't match density matrix size")
    
    # Get all bipartitions
    bipartitions = get_all_bipartitions(n)
    
    # Define SDP variables
    W = cp.Variable((d, d), hermitian=True)
    
    constraints = []
    
    # Trace normalization
    constraints.append(cp.trace(W) == 1)
    
    # For each bipartition M, we need W^{T_M} >= 0
    for partition in bipartitions:
        # W^{T_M} is represented by a new variable that equals the partial transpose
        W_pt = cp.Variable((d, d), hermitian=True)
        constraints.append(W_pt >> 0)
        
        # Set up partial transpose constraints
        for i in range(d):
            for j in range(d):
                multi_i = []
                multi_j = []
                temp_i, temp_j = i, j
                for dim in reversed(dimensions):
                    multi_i.append(temp_i % dim)
                    multi_j.append(temp_j % dim)
                    temp_i //= dim
                    temp_j //= dim
                multi_i = list(reversed(multi_i))
                multi_j = list(reversed(multi_j))
                
                new_multi_i = list(multi_i)
                new_multi_j = list(multi_j)
                for k in range(n):
                    if partition[k] == 1:
                        new_multi_i[k], new_multi_j[k] = multi_j[k], multi_i[k]
                
                new_i, new_j = 0, 0
                for k in range(n):
                    new_i = new_i * dimensions[k] + new_multi_i[k]
                    new_j = new_j * dimensions[k] + new_multi_j[k]
                
                if j >= i:
                    constraints.append(W_pt[i, j] == W[new_i, new_j])
    
    # Objective
    objective = cp.Minimize(cp.real(cp.trace(W @ rho)))
    
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.SCS, verbose=verbose)
    except:
        try:
            prob.solve(solver=cp.CVXOPT, verbose=verbose)
        except:
            prob.solve(verbose=verbose)
    
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print(f"Warning: Solver status = {prob.status}")
    
    min_val = prob.value if prob.value is not None else 0
    
    precision = 1e-8
    if min_val < -precision:
        witness = W.value
    else:
        witness = np.zeros((d, d))
    
    return min_val, witness


def entmon(rho: np.ndarray, dimensions: Optional[List[int]] = None,
           verbose: bool = False) -> float:
    """
    Compute the entanglement monotone for genuine multipartite entanglement.
    
    This uses the modified SDP with 0 <= P_M <= I and 0 <= Q_M <= I.
    
    Parameters
    ----------
    rho : np.ndarray
        Density matrix
    dimensions : List[int], optional
        Dimensions of subsystems
    verbose : bool
        If True, print solver output
    
    Returns
    -------
    float
        Entanglement monotone value (0 for biseparable states)
    """
    d = rho.shape[0]
    
    if dimensions is None:
        n = int(np.log2(d))
        if 2**n != d:
            raise ValueError("For non-qubit systems, please specify dimensions")
        dimensions = [2] * n
    else:
        n = len(dimensions)
        if np.prod(dimensions) != d:
            raise ValueError("Dimensions don't match density matrix size")
    
    bipartitions = get_all_bipartitions(n)
    
    W = cp.Variable((d, d), hermitian=True)
    I = np.eye(d)
    
    constraints = []
    
    for idx, partition in enumerate(bipartitions):
        P = cp.Variable((d, d), hermitian=True)
        Q = cp.Variable((d, d), hermitian=True)
        
        # 0 <= P_M <= I
        constraints.append(P >> 0)
        constraints.append(I - P >> 0)
        
        # 0 <= Q_M <= I
        constraints.append(Q >> 0)
        constraints.append(I - Q >> 0)
        
        # W - P = Q^{T_M}
        for i in range(d):
            for j in range(d):
                multi_i = []
                multi_j = []
                temp_i, temp_j = i, j
                for dim in reversed(dimensions):
                    multi_i.append(temp_i % dim)
                    multi_j.append(temp_j % dim)
                    temp_i //= dim
                    temp_j //= dim
                multi_i = list(reversed(multi_i))
                multi_j = list(reversed(multi_j))
                
                new_multi_i = list(multi_i)
                new_multi_j = list(multi_j)
                for k in range(n):
                    if partition[k] == 1:
                        new_multi_i[k], new_multi_j[k] = multi_j[k], multi_i[k]
                
                new_i, new_j = 0, 0
                for k in range(n):
                    new_i = new_i * dimensions[k] + new_multi_i[k]
                    new_j = new_j * dimensions[k] + new_multi_j[k]
                
                if j >= i:
                    constraints.append(W[i, j] - P[i, j] == Q[new_i, new_j])
    
    objective = cp.Minimize(cp.real(cp.trace(W @ rho)))
    
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.SCS, verbose=verbose)
    except:
        try:
            prob.solve(solver=cp.CVXOPT, verbose=verbose)
        except:
            prob.solve(verbose=verbose)
    
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print(f"Warning: Solver status = {prob.status}")
    
    min_val = prob.value if prob.value is not None else 0
    
    # Return negative of minimum (so positive = entangled)
    return -min_val if min_val is not None else 0


# ============== Helper functions for creating common states ==============

def ghz_state(n: int) -> np.ndarray:
    d = 2**n
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[-1] = 1/np.sqrt(2)
    return np.outer(psi, psi.conj())


def w_state(n: int) -> np.ndarray:
    d = 2**n
    psi = np.zeros(d, dtype=complex)
    for i in range(n):
        idx = 2**i
        psi[idx] = 1/np.sqrt(n)
    return np.outer(psi, psi.conj())


def bell_state(which: str = 'phi+') -> np.ndarray:
    """
    Create a Bell state.
    
    Parameters
    ----------
    which : str
        'phi+': (|00> + |11>)/sqrt(2)
        'phi-': (|00> - |11>)/sqrt(2)
        'psi+': (|01> + |10>)/sqrt(2)
        'psi-': (|01> - |10>)/sqrt(2)
    """
    if which == 'phi+':
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    elif which == 'phi-':
        psi = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    elif which == 'psi+':
        psi = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    elif which == 'psi-':
        psi = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    else:
        raise ValueError(f"Unknown Bell state: {which}")
    
    return np.outer(psi, psi.conj())


def dicke_state(n: int, k: int) -> np.ndarray:
    """
    Create n-qubit Dicke state with k excitations.
    
    |D_{k,n}> = normalized sum of all permutations of |0...0 1...1>
                with k ones and (n-k) zeros.
    """
    from math import comb
    d = 2**n
    psi = np.zeros(d, dtype=complex)
    
    # Find all indices with exactly k ones
    for idx in range(d):
        if bin(idx).count('1') == k:
            psi[idx] = 1
    
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def add_white_noise(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Mix state with white noise: (1-p)*rho + p*I/d
    
    Parameters
    ----------
    rho : np.ndarray
        Original density matrix
    p : float
        Noise fraction (0 <= p <= 1)
    """
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d

