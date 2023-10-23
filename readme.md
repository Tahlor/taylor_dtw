
    python taylor_dtw/setup.py install

    OR

    pip install git+ssh://git@github.com/Tahlor/taylor_dtw

Examples:


    from taylor_dtw.custom_dtw import dtw2d_with_backward

    x1 = np.ascontiguousarray(pred)  # time step, batch, (x,y)
    x2 = np.ascontiguousarray(targ)
    x3 = np.ascontiguousarray(reverse_targ)

    dist, cost, a, b = dtw2d_with_backward(x1, x2, x3)
