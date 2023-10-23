
    python taylor_dtw/setup.py install

    OR

    pip install git+ssh://git@github.com/Tahlor/taylor_dtw

Examples:


    import numpy as np
    from taylor_dtw.custom_dtw import dtw2d_with_backward, constrained_dtw2d

    # Define pred array (timesteps, 2) for demonstration
    pred = np.array([
    [0, 0],
    [0, 1],
    [1, 3],
    [1, 3],
    [4, 2],
    [3, 0]
], dtype=np.float64)
pred = np.ascontiguousarray(pred)

# Define targ array (timesteps, 2)
targ = np.array([
    [0, 0],
    [0, 1],
    [1, 3],
    [4, 2],
    [4, 2],
    [3, 0]
], dtype=np.float64)

targ = np.ascontiguousarray(targ)

# Define reverse_targ (timesteps, 2)
reverse_targ = np.flip(targ, axis=0)
reverse_targ = np.ascontiguousarray(reverse_targ)

# Invoke the dtw2d_with_backward function
dist, cost, a, b = constrained_dtw2d(pred, targ, 3)

# Display results
print(f"Distance: {dist.base}")
print(f"Cost Matrix: {cost}")
print(f"Alignment a: {a}")
print(f"Alignment b: {b}")

# Compares each element of pred to the current idx in the targ and reverse_targ arrays
# Unclear why you would want this and not the minimum cost of either one separately (as opposed to by-element)
# You might get something that only learns to match half of the sequence, and then reverse itself
dist, cost, a, b = dtw2d_with_backward(pred, targ, reverse_targ)

