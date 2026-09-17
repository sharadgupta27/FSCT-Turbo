import numpy as np
import pandas as pd
from scipy import spatial
import time
import warnings
import os
from tools import load_file, save_file, get_heights_above_DTM
from fsct_exceptions import DataQualityError

warnings.filterwarnings("ignore", category=RuntimeWarning)


class PostProcessing:
    def __init__(self, parameters):
        self.post_processing_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.output_dir = (
            os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/")
            + "/"
            + self.filename.split("/")[-1][:-4]
            + "_FSCT_output/"
        )
        self.filename = self.filename.split("/")[-1]

        self.noise_class_label = parameters["noise_class"]
        self.terrain_class_label = parameters["terrain_class"]
        self.vegetation_class_label = parameters["vegetation_class"]
        self.cwd_class_label = parameters["cwd_class"]
        self.stem_class_label = parameters["stem_class"]
        print("Loading segmented point cloud...")
        self.point_cloud, self.headers_of_interest = load_file(
            self.output_dir + "segmented.las", headers_of_interest=["x", "y", "z", "red", "green", "blue", "label"]
        )
        self.point_cloud = np.hstack(
            (self.point_cloud, np.zeros((self.point_cloud.shape[0], 1)))
        )  # Add height above DTM column
        self.headers_of_interest.append("height_above_DTM")  # Add height_above_DTM to the headers.
        self.label_index = self.headers_of_interest.index("label")
        self.point_cloud[:, self.label_index] = (
            self.point_cloud[:, self.label_index] + 1
        )  # index offset since noise_class was removed from inference.
        self.plot_summary = pd.read_csv(self.output_dir + "plot_summary.csv", index_col=None)

    def make_DTM(self, crop_dtm=False):
        print("Making DTM...")
        full_point_cloud_kdtree = spatial.cKDTree(self.point_cloud[:, :2])
        terrain_kdtree = spatial.cKDTree(self.terrain_points[:, :2])
        xmin = np.floor(np.min(self.terrain_points[:, 0])) - 3
        ymin = np.floor(np.min(self.terrain_points[:, 1])) - 3
        xmax = np.ceil(np.max(self.terrain_points[:, 0])) + 3
        ymax = np.ceil(np.max(self.terrain_points[:, 1])) + 3
        x_points = np.linspace(xmin, xmax, int(np.ceil((xmax - xmin) / self.parameters["grid_resolution"])) + 1)
        y_points = np.linspace(ymin, ymax, int(np.ceil((ymax - ymin) / self.parameters["grid_resolution"])) + 1)
        grid_resolution = self.parameters["grid_resolution"]
        terrain_z = self.terrain_points[:, 2]
        cloud_z = self.point_cloud[:, 2]

        # Bound on the widening search below. Without it, a cloud holding
        # fewer than 100 points in total would loop forever.
        max_radius = max(xmax - xmin, ymax - ymin)

        # All grid cells at once, in the order the old x-outer / y-inner loop
        # visited them, so the DTM rows come out in the same order.
        grid = np.stack(np.meshgrid(x_points, y_points, indexing="ij"), axis=-1).reshape(-1, 2)
        num_cells = grid.shape[0]

        # The per-cell search widened the terrain query radius from 3 to 6
        # grid cells, stopping at the first radius holding more than 100
        # points. That is at most four radii, so the neighbour *counts* at all
        # four can be taken for every cell in four threaded calls, without
        # materialising any index lists, and each cell's stopping radius read
        # off from them. The radii are accumulated by repeated addition exactly
        # as the old loop did (rather than as k * resolution) so the float
        # values - and so the inclusion of any point lying right on a
        # boundary - are identical.
        #
        # This replaced a Python loop of two to eight cKDTree calls per cell -
        # a 100 x 100 m plot at 0.5 m is 44,000 cells - each building and
        # discarding a Python list of point indices just to take its length.
        radii = [grid_resolution * 3]
        for _ in range(3):
            radii.append(radii[-1] + grid_resolution)
        radii = np.array(radii)
        counts = np.stack(
            [terrain_kdtree.query_ball_point(grid, r=r, return_length=True, workers=-1) for r in radii],
            axis=1,
        )
        exceeds = counts > 100
        stop_index = np.where(exceeds.any(axis=1), exceeds.argmax(axis=1), radii.shape[0] - 1)
        stop_radius = radii[stop_index]
        stop_count = counts[np.arange(num_cells), stop_index]
        from_terrain = stop_count >= 100

        grid_z = np.empty(num_cells)
        keep = np.zeros(num_cells, dtype=bool)

        # Cells with enough terrain returns: one batched query at each cell's
        # own stopping radius, then the 20th percentile per cell.
        terrain_cells = np.flatnonzero(from_terrain)
        if terrain_cells.shape[0] > 0:
            index_lists = terrain_kdtree.query_ball_point(
                grid[terrain_cells], r=stop_radius[terrain_cells], workers=-1
            )
            for cell, indices in zip(terrain_cells, index_lists):
                grid_z[cell] = np.percentile(terrain_z[indices], 20)
            keep[terrain_cells] = True

        # Cells without: fall back to the whole cloud, widening from where the
        # terrain search left off. Same loop as before; these are normally the
        # few cells around the plot edge.
        for cell in np.flatnonzero(~from_terrain):
            radius = stop_radius[cell]
            indices = full_point_cloud_kdtree.query_ball_point(grid[cell], r=radius)
            while len(indices) <= 100 and radius <= max_radius:
                radius += grid_resolution
                indices = full_point_cloud_kdtree.query_ball_point(grid[cell], r=radius)

            if len(indices) == 0:
                continue
            grid_z[cell] = np.percentile(cloud_z[indices], 2.5)
            keep[cell] = True

        grid_points = np.column_stack((grid[keep], grid_z[keep])) if keep.any() else np.zeros((0, 3))

        if self.parameters["plot_radius"] > 0:
            plot_centre = [[float(self.plot_summary["Plot Centre X"].iloc[0]), float(self.plot_summary["Plot Centre Y"].iloc[0])]]
            crop_radius = self.parameters["plot_radius"] + self.parameters["plot_radius_buffer"]
            grid_points = grid_points[np.linalg.norm(grid_points[:, :2] - plot_centre, axis=1) <= crop_radius]

        elif crop_dtm:
            distances, _ = full_point_cloud_kdtree.query(grid_points[:, :2], k=[1])
            distances = np.squeeze(distances)
            grid_points = grid_points[distances <= self.parameters["grid_resolution"]]
        print("    DTM Done")
        return grid_points

    def process_point_cloud(self):
        self.terrain_points = self.point_cloud[
            self.point_cloud[:, self.label_index] == self.terrain_class_label
        ]  # -2 is now the class label as we added the height above DTM column.

        try:
            self.DTM = self.make_DTM(crop_dtm=True)
        except ValueError:
            raise DataQualityError("Failed to make DTM. \nThere probably aren't any terrain_points.")

        save_file(self.output_dir + "DTM.las", self.DTM)

        self.convexhull = spatial.ConvexHull(self.DTM[:, :2])
        self.plot_area = self.convexhull.volume / 10000  # volume is area in 2d.
        print("Plot area is approximately", self.plot_area, "ha")

        above_and_below_DTM_trim_dist = 0.2

        self.point_cloud = get_heights_above_DTM(
            self.point_cloud, self.DTM
        )  # Add a height above DTM column to the point clouds.
        self.terrain_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.terrain_class_label]
        self.terrain_points_rejected = np.vstack(
            (
                self.terrain_points[self.terrain_points[:, -1] <= -above_and_below_DTM_trim_dist],
                self.terrain_points[self.terrain_points[:, -1] > above_and_below_DTM_trim_dist],
            )
        )
        self.terrain_points = self.terrain_points[
            np.logical_and(
                self.terrain_points[:, -1] > -above_and_below_DTM_trim_dist,
                self.terrain_points[:, -1] < above_and_below_DTM_trim_dist,
            )
        ]

        save_file(
            self.output_dir + "terrain_points.las",
            self.terrain_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )
        self.stem_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.stem_class_label]
        self.terrain_points = np.vstack(
            (
                self.terrain_points,
                self.stem_points[
                    np.logical_and(
                        self.stem_points[:, -1] >= -above_and_below_DTM_trim_dist,
                        self.stem_points[:, -1] <= above_and_below_DTM_trim_dist,
                    )
                ],
            )
        )
        self.stem_points_rejected = self.stem_points[self.stem_points[:, -1] <= above_and_below_DTM_trim_dist]
        self.stem_points = self.stem_points[self.stem_points[:, -1] > above_and_below_DTM_trim_dist]
        save_file(
            self.output_dir + "stem_points.las",
            self.stem_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )

        self.vegetation_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.vegetation_class_label]
        self.terrain_points = np.vstack(
            (
                self.terrain_points,
                self.vegetation_points[
                    np.logical_and(
                        self.vegetation_points[:, -1] >= -above_and_below_DTM_trim_dist,
                        self.vegetation_points[:, -1] <= above_and_below_DTM_trim_dist,
                    )
                ],
            )
        )
        self.vegetation_points_rejected = self.vegetation_points[
            self.vegetation_points[:, -1] <= above_and_below_DTM_trim_dist
        ]
        self.vegetation_points = self.vegetation_points[self.vegetation_points[:, -1] > above_and_below_DTM_trim_dist]
        save_file(
            self.output_dir + "vegetation_points.las",
            self.vegetation_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )

        self.cwd_points = self.point_cloud[
            self.point_cloud[:, self.label_index] == self.cwd_class_label
        ]  # -2 is now the class label as we added the height above DTM column.
        self.terrain_points = np.vstack(
            (
                self.terrain_points,
                self.cwd_points[
                    np.logical_and(
                        self.cwd_points[:, -1] >= -above_and_below_DTM_trim_dist,
                        self.cwd_points[:, -1] <= above_and_below_DTM_trim_dist,
                    )
                ],
            )
        )

        self.cwd_points_rejected = np.vstack(
            (
                self.cwd_points[self.cwd_points[:, -1] <= above_and_below_DTM_trim_dist],
                self.cwd_points[self.cwd_points[:, -1] >= 10],
            )
        )
        self.cwd_points = self.cwd_points[
            np.logical_and(self.cwd_points[:, -1] > above_and_below_DTM_trim_dist, self.cwd_points[:, -1] < 3)
        ]
        save_file(
            self.output_dir + "cwd_points.las",
            self.cwd_points,
            headers_of_interest=self.headers_of_interest,
            silent=False,
        )

        self.terrain_points[:, self.label_index] = self.terrain_class_label
        self.cleaned_pc = np.vstack((self.terrain_points, self.vegetation_points, self.cwd_points, self.stem_points))
        save_file(
            self.output_dir + "segmented_cleaned.las", self.cleaned_pc, headers_of_interest=self.headers_of_interest
        )

        self.post_processing_time_end = time.time()
        self.post_processing_time = self.post_processing_time_end - self.post_processing_time_start
        print("Post-processing took", self.post_processing_time, "seconds")
        self.plot_summary["Post processing time (s)"] = self.post_processing_time
        self.plot_summary["Num Terrain Points"] = self.terrain_points.shape[0]
        self.plot_summary["Num Vegetation Points"] = self.vegetation_points.shape[0]
        self.plot_summary["Num CWD Points"] = self.cwd_points.shape[0]
        self.plot_summary["Num Stem Points"] = self.stem_points.shape[0]
        self.plot_summary["Plot Area"] = self.plot_area
        self.plot_summary["Post processing time (s)"] = self.post_processing_time
        self.plot_summary.to_csv(self.output_dir + "plot_summary.csv", index=False)
        print("Post processing done.")
