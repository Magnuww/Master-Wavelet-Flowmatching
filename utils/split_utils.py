def get_split_indices(args, shapes, levels, tr_sample):

    """
    Determines the splits and indices for coefficient groups based on CLI args.
    Returns (splits, indx).
    """
    import numpy as np

    splits = args.splits
    no_split = args.no_split

    if no_split:
        splits = [tr_sample.shape[2]]

    if args.half_split:
        cumsum = np.cumsum(shapes)
        splits = [cumsum[-2]]
        print(f"Half split: {splits}")

    if args.level_split is not None:
        cumsum = np.cumsum(shapes)
        assert max(args.level_split) <= levels and min(args.level_split) >= 1, f"Level split indexes must be between 1 and {levels}. Got {args.level_split}"
        assert len(args.level_split) < levels, f"Number of level splits must be less than the number of levels. Got {len(args.level_split)} splits for {levels} levels."
        splits = [cumsum[level - 1] for level in args.level_split]
        print(f"Using level-based splits at levels {args.level_split}: {splits}")

    if splits is not None:
        if splits[-1] < sum(shapes):
            print(f"Warning: Provided splits {splits} do not sum up to the total number of coefficients {sum(shapes)}. Adding the last split to match the total.")
            splits.append(sum(shapes))
        print(f"Using custom splits: {splits}")
        if splits[0] != 0:
            indx = np.concatenate(([0], splits))
        else:
            indx = np.array(splits)
    else:
        splits = shapes
        cumsum = np.cumsum(shapes)
        indx  = np.concatenate(([0], cumsum))


    return splits, indx
