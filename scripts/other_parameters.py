# Don't change these unless you really understand what you are doing with them/are learning how the code works.
# These have been tuned to work on most high resolution forest point clouds without changing them, but you may be able
# to tune these better for your particular data. Almost everything here is a trade-off between different situations, so
# optimisation is not straight-forward.

other_parameters = dict(
    model_filename="model.pth",
    use_amp=True,  # Run semantic segmentation in fp16 on the GPU. Roughly halves both the runtime and the
    # activation memory. Shifts ~0.17% of point labels (points where the top two classes were near-tied);
    # for comparison, the unseeded randomness below used to move ~8.76% between runs. Ignored on CPU.
    # Caveat: fp16 reductions are not bit-deterministic, so even with random_seed set two runs can differ
    # on a handful of points (measured: 2 in 673,517). Set use_amp=False for exactly repeatable output.
    prewarm_worker_pool=True,  # Start the measurement stage's worker processes while the GPU is busy with
    # segmentation, so their start-up (importing numpy/scipy/sklearn/hdbscan, several seconds per worker on
    # Windows) is hidden behind it rather than paid at the start of measurement. Costs roughly 100 MB of RAM
    # per idle worker during segmentation; set to False on a memory-starved machine.
    random_seed=0,  # Makes runs reproducible. Two things in FSCT are random: the subsampling of boxes that
    # exceed max_points_per_box, and the random starting point of the farthest-point-sampling in the
    # segmentation model. Left unseeded they made repeat runs on the same file disagree on roughly 10% of
    # point labels. Set to None for the old non-reproducible behaviour.
    box_dimensions=[6, 6, 6],  # Dimensions of the sliding box used for semantic segmentation.
    box_overlap=[0.5, 0.5, 0.5],  # Overlap of the sliding box used for semantic segmentation.
    min_points_per_box=1000,  # Minimum number of points for input to the model. Too few points and it becomes near impossible to accurately label them (though assuming vegetation class is the safest bet here).
    max_points_per_box=20000,  # Maximum number of points for input to the model. The model may tolerate higher numbers if you decrease the batch size accordingly (to fit on the GPU), but this is not tested.
    noise_class=0,  # Don't change
    terrain_class=1,  # Don't change
    vegetation_class=2,  # Don't change
    cwd_class=3,  # Don't change
    stem_class=4,  # Don't change
    grid_resolution=0.5,  # Resolution of the DTM.
    vegetation_coverage_resolution=0.2,
    num_neighbours=5,
    sorting_search_angle=20,
    sorting_search_radius=1,
    sorting_angle_tolerance=90,
    max_search_radius=3,
    max_search_angle=30,
    min_cluster_size=30,  # Used for HDBSCAN clustering step. Recommend not changing for general use.
    cleaned_measurement_radius=0.2,  # During cleaning, this w
    subsample=0,  # Off by default. The inherited comment here said "generally leave this on"
    # while the value was 0, so it described the opposite of what the code did. Set to 1 to thin
    # the cloud to subsampling_min_spacing before segmentation - worth it on dense scans, and it
    # changes the measurements, so turn it on deliberately rather than by default.
    subsampling_min_spacing=0.01,  # The point cloud will be subsampled such that the closest any 2 points can be is 0.01 m.
    minimum_CCI=0.3,  # Minimum valid Circuferential Completeness Index (CCI) for non-interpolated circle/cylinder fitting. Any measurements with CCI below this are deleted.
    fix_cci_sectors=False,  # The CCI sector angles are built in degrees but consumed as radians, so the 80
    # "evenly spaced sectors" are really 80 arbitrary directions. Setting this to True uses genuinely evenly
    # spaced sectors. It changes CCI values, and CCI decides which cylinders survive, so stem count, DBH and tree
    # height all move with it. Left off by default so results stay comparable with earlier runs.
    assign_unassigned_skeleton_points=False,  # The step that was meant to fold DBSCAN's leftover "noise"
    # skeleton points into their nearest cluster never did anything (it wrote into a temporary copy, and
    # looked for neighbours only among the other unassigned points). Those points are currently dropped.
    # Setting this to True makes the step work as described. It recovers skeleton points, which changes
    # which stems get cylinders fitted and moves stem count, DBH and height with it - so it is off by
    # default and results stay comparable with earlier runs.
    circle_fit_trials=1000,  # Maximum RANSAC trials per fitted circle. The original code used 10,000 with a
    # sample size of 30% of the slice, a combination for which RANSAC's early-stopping rule almost never
    # fires, so nearly every circle ran all 10,000 trials. The trials are now evaluated in batches and the
    # same stopping rule is applied after each batch, so clean slices still finish in a few dozen trials.
    # Measured on data/test/example.las, the best consensus set stops improving well before trial 1000.
    circle_fit_max_points=1500,  # Cap on the number of slice points used to fit a circle (0 = use all).
    # RANSAC cost is linear in the point count for every trial, and a slice with thousands of returns
    # describes the same circle as a well-spread sample of it. The Circumferential Completeness Index is
    # still computed from every point in the slice.
    min_tree_cyls=10,  # Deletes any trees with fewer than 10 cylinders (before the cylinder interpolation step).
    low_resolution_point_cloud_hack_mode=0,
)  # Very ugly hack that can sometimes be useful on point clouds which are on the borderline of having not enough points to be functional with FSCT. Set to a positive integer. Point cloud will be copied this many times (with noise added) to artificially increase point density giving the segmentation model more points.
