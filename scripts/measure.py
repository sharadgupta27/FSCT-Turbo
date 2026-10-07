import math
import threading
from copy import deepcopy
from multiprocessing import get_context, TimeoutError
import numpy as np
import os
from scipy import spatial  # TODO Test if sklearn kdtree is faster.

# pandas, networkx and skspatial are imported where they are used, in
# methods that only the parent process calls. Every worker in the
# measurement pool re-imports this module from scratch (see the note at
# the top of tools.py), and those three alone were about a third of that
# cost for modules the workers have no use for.
from tools import (
    get_fsct_path,
    load_file,
    save_file,
    low_resolution_hack_mode,
    cluster_hdbscan,
    cluster_dbscan,
    get_heights_above_DTM,
    get_taper,
    DTMInterpolator,
)
from fsct_exceptions import DataQualityError
import time
from sklearn.neighbors import BallTree


def _indexed_call(task):
    """Pool trampoline: run task's function on its item, returning (index, result)."""
    index, func, item = task
    return index, func(item)


def _initialise_worker():
    """
    Pool initializer: runs once in each worker as soon as it starts.

    It does nothing itself. Its job is done by the time it is called: to run
    it, the worker has to unpickle a reference to this function, and that
    means importing this module - and with it tools, scipy.spatial,
    sklearn.neighbors and hdbscan, everything a task will need.

    Without it a spawned worker imports those only when its *first task*
    arrives, since that is the first thing it unpickles that lives here. So
    the pool that run_tools.FSCT starts ahead of segmentation would have sat
    through the whole GPU stage as sixteen bare interpreters and then all
    imported at once the moment measurement began, which is the exact
    start-up cost the early start is meant to hide.
    """


class MeasureTree:
    def __init__(self, parameters):
        self.measure_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.output_dir = (
            os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/")
            + "/"
            + self.filename.split("/")[-1][:-4]
            + "_FSCT_output/"
        )
        self.filename = self.filename.split("/")[-1]

        self.num_cpu_cores = parameters["num_cpu_cores"]
        self.num_neighbours = parameters["num_neighbours"]
        self.slice_thickness = parameters["slice_thickness"]
        self.slice_increment = parameters["slice_increment"]

        import pandas as pd

        self.plot_summary = pd.read_csv(self.output_dir + "plot_summary.csv", index_col=False)
        self.parameters["plot_radius"] = float(self.plot_summary["Plot Radius"].iloc[0])
        self.parameters["plot_radius_buffer"] = float(self.plot_summary["Plot Radius Buffer"].iloc[0])
        self.plot_area = float(self.plot_summary["Plot Area"].iloc[0])
        self.stem_points, headers_of_interest = load_file(
            self.output_dir + "stem_points.las",
            headers_of_interest=["x", "y", "z", "red", "green", "blue", "label", "height_above_DTM"],
        )
        self.stem_points = np.hstack((self.stem_points, np.zeros((self.stem_points.shape[0], 1))))

        print("stempoints", headers_of_interest)
        if self.parameters["low_resolution_point_cloud_hack_mode"]:
            self.stem_points = low_resolution_hack_mode(
                self.stem_points,
                self.parameters["low_resolution_point_cloud_hack_mode"],
                self.parameters["subsampling_min_spacing"],
                self.parameters["num_cpu_cores"],
            )
            save_file(self.output_dir + self.filename[:-4] + "_stem_points_hack_mode_cloud.las", self.stem_points)

        self.DTM, headers_of_interest = load_file(self.output_dir + "DTM.las")
        # Triangulate the DTM once and reuse it. Every height-above-DTM
        # lookup below goes through this; building it per call meant
        # re-triangulating the whole DTM twice for every tree in the plot.
        self.dtm_interpolator = DTMInterpolator(self.DTM)
        self.characters = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "dot",
            "m",
            "space",
            "_",
            "-",
            "semiC",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "_M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]
        self.character_viz = []

        for i in self.characters:
            self.character_viz.append(
                np.genfromtxt(os.path.join(get_fsct_path("tools/numbers"), i + ".csv"), delimiter=",")
            )

        self.cyl_dict = dict(
            x=0,
            y=1,
            z=2,
            nx=3,
            ny=4,
            nz=5,
            radius=6,
            CCI=7,
            branch_id=8,
            parent_branch_id=9,
            tree_id=10,
            tree_volume=11,
            segment_angle_to_horiz=12,
            height_above_dtm=13,
        )
        # Columns of make_cyl_visualisation_array's output: each circle point's
        # x, y, z, then the last eight cylinder fields. Saving those files with
        # the full cyl_dict declared nx, ny and nz too, three extra fields that
        # held nothing but zeros.
        self.cyl_vis_headers = ["x", "y", "z"] + list(self.cyl_dict)[6:]

        self.veg_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, label=6, height_above_dtm=7, tree_id=8)
        self.stem_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, label=6, height_above_dtm=7, tree_id=8)
        self.tree_data_dict = dict(
            PlotId=0,
            TreeId=1,
            x_tree_base=2,
            y_tree_base=3,
            z_tree_base=4,
            DBH=5,
            CCI_at_BH=6,
            Height=7,
            Volume_1=8,
            Volume_2=9,
            Crown_mean_x=10,
            Crown_mean_y=11,
            Crown_top_x=12,
            Crown_top_y=13,
            Crown_top_z=14,
            mean_understory_height_in_5m_radius=15,
        )

        self.terrain_points, headers_of_interest = load_file(
            self.output_dir + "terrain_points.las",
            headers_of_interest=["x", "y", "z", "red", "green", "blue", "label", "height_above_DTM"],
        )
        self.terrain_points = np.hstack((self.terrain_points, np.zeros((self.terrain_points.shape[0], 1))))

        self.vegetation_points, headers_of_interest = load_file(
            self.output_dir + "vegetation_points.las",
            headers_of_interest=["x", "y", "z", "red", "green", "blue", "label", "height_above_DTM"],
        )
        self.vegetation_points = np.hstack((self.vegetation_points, np.zeros((self.vegetation_points.shape[0], 1))))

        # Remove understorey vegetation and save it.
        ground_veg_mask = (
            self.vegetation_points[:, self.veg_dict["height_above_dtm"]] <= self.parameters["ground_veg_cutoff_height"]
        )
        self.ground_veg = self.vegetation_points[ground_veg_mask]
        save_file(self.output_dir + "ground_veg.las", self.ground_veg, headers_of_interest=list(self.veg_dict))
        self.vegetation_points = self.vegetation_points[np.logical_not(ground_veg_mask)]
        veg_kdtree = spatial.cKDTree(self.vegetation_points[:, :2], leafsize=10000)

        try:
            self.cwd_points, headers_of_interest = load_file(
                self.output_dir + "cwd_points.las",
                headers_of_interest=["x", "y", "z", "red", "green", "blue", "label", "height_above_DTM"],
            )
            if self.cwd_points.shape[1] < self.stem_points.shape[1]:
                self.cwd_points = np.hstack(
                    (
                        self.cwd_points,
                        np.zeros((self.cwd_points.shape[0], self.stem_points.shape[1] - self.cwd_points.shape[1])),
                    )
                )

        except FileNotFoundError:
            self.cwd_points = np.zeros((0, self.stem_points.shape[1]))

        cwd_kdtree = spatial.cKDTree(self.cwd_points[:, :2], leafsize=10000)

        self.ground_veg_kdtree = spatial.cKDTree(self.ground_veg[:, :2], leafsize=10000)
        xmin = np.floor(np.min(self.terrain_points[:, 0]))
        ymin = np.floor(np.min(self.terrain_points[:, 1]))
        xmax = np.ceil(np.max(self.terrain_points[:, 0]))
        ymax = np.ceil(np.max(self.terrain_points[:, 1]))
        x_points = np.linspace(
            xmin, xmax, int(np.ceil((xmax - xmin) / self.parameters["vegetation_coverage_resolution"])) + 1
        )
        y_points = np.linspace(
            ymin, ymax, int(np.ceil((ymax - ymin) / self.parameters["vegetation_coverage_resolution"])) + 1
        )

        convexhull = spatial.ConvexHull(self.DTM[:, :2])

        # Coverage fractions over the sample grid, computed as whole arrays.
        #
        # This was a Python double loop over every grid cell - a 100 x 100 m
        # plot at the default 0.2 m resolution is 250,000 of them - and each
        # cell ran inside_conv_hull, which is a Python generator over every
        # facet of the hull, then three separate query_ball_point calls that
        # each built a Python list of point indices only for len() to be taken
        # of it.
        #
        # Identical arithmetic, three orders of magnitude fewer interpreter
        # round trips: the hull test is one matrix product for all cells at
        # once, and cKDTree answers all the cells in a single threaded call
        # with return_length=True, which counts the neighbours without
        # materialising the index lists.
        grid = np.stack(np.meshgrid(x_points, y_points, indexing="ij"), axis=-1).reshape(-1, 2)

        # hull.equations is [normal | offset]; a point is inside when
        # normal . point + offset <= tolerance for every facet. Same test and
        # same tolerance as inside_conv_hull, done for all points at once.
        hull_eq = convexhull.equations
        inside = np.all(grid @ hull_eq[:, :-1].T + hull_eq[:, -1] <= 1e-5, axis=1)
        grid = grid[inside]

        self.ground_area = int(grid.shape[0])  # unitless. Not in m2.
        resolution = self.parameters["vegetation_coverage_resolution"]

        def _covered_cells(kdtree):
            if grid.shape[0] == 0:
                return 0
            counts = kdtree.query_ball_point(grid, r=resolution, p=10, return_length=True, workers=-1)
            return int(np.count_nonzero(counts > 5))

        self.canopy_area = _covered_cells(veg_kdtree)  # unitless. Not in m2.
        self.ground_veg_area = _covered_cells(self.ground_veg_kdtree)  # unitless. Not in m2.
        self.cwd_area = _covered_cells(cwd_kdtree)  # unitless. Not in m2.

        # A plot whose DTM hull contains no grid cell at all used to divide by
        # zero here and take the whole run down with a ZeroDivisionError.
        if self.ground_area == 0:
            print("No sample cells fell inside the DTM hull; coverage fractions are reported as 0.")
        print("Canopy Cover Fraction:", self.canopy_cover_fraction)
        print("Understory Veg Fraction:", self.understory_veg_fraction)
        print("Coarse Woody Debris Fraction:", self.cwd_fraction)
        max_z = np.max(
            np.hstack(
                (
                    self.stem_points[:, 2],
                    self.vegetation_points[:, 2],
                    self.cwd_points[:, 2],
                    self.terrain_points[:, 2],
                )
            )
        )
        min_z = np.min(
            np.hstack(
                (
                    self.stem_points[:, 2],
                    self.vegetation_points[:, 2],
                    self.cwd_points[:, 2],
                    self.terrain_points[:, 2],
                )
            )
        )
        self.z_range = max_z - min_z
        self.text_point_cloud = np.zeros((0, 3))
        self.tree_measurements = np.zeros((0, 8))
        self.text_point_cloud = np.zeros((0, 3))

    def _coverage_fraction(self, covered_area):
        """
        Fraction of the sampled ground cells covered, or 0 when none were sampled.

        Every one of these divisions used to be written out longhand against
        self.ground_area, which is zero whenever the DTM hull contains no
        sample cell - a tiny or degenerate plot. That raised ZeroDivisionError,
        in one case only at the very end of the measurement stage, after all
        the work had been done.
        """
        if not self.ground_area:
            return 0.0
        return covered_area / self.ground_area

    @property
    def canopy_cover_fraction(self):
        return self._coverage_fraction(self.canopy_area)

    @property
    def understory_veg_fraction(self):
        return self._coverage_fraction(self.ground_veg_area)

    @property
    def cwd_fraction(self):
        return self._coverage_fraction(self.cwd_area)

    def interpolate_cyl(self, cyl1, cyl2, resolution):
        """
        Convention to be used
        cyl_1 is child
        cyl_2 is parent
        """
        length = np.linalg.norm(np.array([cyl2[0], cyl2[1], cyl2[2]]) - np.array([cyl1[0], cyl1[1], cyl1[2]]))
        points_per_line = int(np.ceil(length / resolution))
        interpolated = np.zeros((0, 14))
        if cyl1.shape[0] > 0 and cyl2.shape[0] > 0:
            xyzinterp = np.linspace(cyl1[:3], cyl2[:3], points_per_line, axis=0)
            if xyzinterp.shape[0] > 0:
                interpolated = np.zeros((xyzinterp.shape[0], 14))
                interpolated[:, :3] = xyzinterp

                normal = (cyl2[:3] - cyl1[:3]) / np.linalg.norm(cyl2[:3] - cyl1[:3])

                if normal[2] < 0:
                    normal[:3] = normal[:3] * -1

                interpolated[:, 3:6] = normal

                interpolated[:, self.cyl_dict["tree_id"]] = cyl1[self.cyl_dict["tree_id"]]
                interpolated[:, self.cyl_dict["branch_id"]] = cyl1[self.cyl_dict["branch_id"]]
                interpolated[:, self.cyl_dict["parent_branch_id"]] = cyl2[self.cyl_dict["branch_id"]]
                interpolated[:, self.cyl_dict["radius"]] = np.min(
                    [cyl1[self.cyl_dict["radius"]], cyl2[self.cyl_dict["radius"]]]
                )

        return interpolated

    @classmethod
    def compute_angle(cls, normal1, normal2):
        """
        Computes the angle in degrees between two 3D vectors.

        Args:
            normal1:
            normal2:

        Returns:
            theta: angle in degrees
        """
        normal1 = np.atleast_2d(normal1)
        normal2 = np.atleast_2d(normal2)

        norm1 = normal1 / np.atleast_2d(np.linalg.norm(normal1, axis=1)).T
        norm2 = normal2 / np.atleast_2d(np.linalg.norm(normal2, axis=1)).T
        dot = np.clip(np.einsum("ij,ij->i", norm1, norm2), -1, 1)
        theta = np.degrees(np.arccos(dot))
        return theta

    def cylinder_sorting(self, cylinder_array, angle_tolerance, search_angle, distance_tolerance):
        """
        Step 1 of sorting initial cylinders into individual trees.
        For a cylinder to be joined up with another cylinder in this step, it must meet the below conditions.

        All angles are specified in degrees.
            cylinder_array:
                The Numpy array of cylinders created during cylinder fitting.

            angle_tolerance:
                Angle tolerance refers to the angle between major axis vectors of the two cylinders being queried. If
                the angle is less than "angle_tolerance", this condition is satisfied.

            search_angle:
                Search angle refers to the angle between the major axis of cylinder 1, and the vector from cylinder 1's
                centre point to cylinder 2's centre point.

            distance_tolerance:
                Cylinder centre points must be within this distance to meet this condition. Think of a ball of radius
                "distance_tolerance".

        Returns: The sorted cylinder array.
        """

        def within_angle_tolerance(normal1, normal2, angle_tolerance):
            """Checks if normal1 and normal2 are within "angle_tolerance"
            of each other."""
            theta = self.compute_angle(normal1, normal2)
            return abs((theta > 90) * 180 - theta) <= angle_tolerance
            # return theta<=angle_tolerance

        def criteria_check(cyl1, cyl2, angle_tolerance, search_angle):
            """
            Decides if cyl2 should be joined to cyl1 and if they are the same tree.
            angle_tolerance is the maximum angle between normal vectors of cylinders to be considered the same branch.
            """
            vector_array = cyl2[:, :3] - np.atleast_2d(cyl1[:3])
            condition1 = within_angle_tolerance(cyl1[3:6], cyl2[:, 3:6], angle_tolerance)
            condition2 = within_angle_tolerance(cyl1[3:6], vector_array, search_angle)
            # condition3 = cyl2[:, self.cyl_dict['radius']] < cyl1[self.cyl_dict['radius']]*1.05
            # cyl2[np.logical_and(condition1, condition2, condition3), self.cyl_dict['tree_id']] = cyl1[self.cyl_dict['tree_id']]
            # cyl2[np.logical_and(condition1, condition2, condition3), self.cyl_dict['parent_branch_id']] = cyl1[self.cyl_dict['branch_id']]
            cyl2[np.logical_and(condition1, condition2), self.cyl_dict["tree_id"]] = cyl1[self.cyl_dict["tree_id"]]
            cyl2[np.logical_and(condition1, condition2), self.cyl_dict["parent_branch_id"]] = cyl1[
                self.cyl_dict["branch_id"]
            ]

            return cyl2

        max_tree_label = 1

        cylinder_array = cylinder_array[
            cylinder_array[:, self.cyl_dict["radius"]] != 0
        ]  # ignore all points with radius of 0.

        # The loop below repeatedly takes the lowest cylinder not yet dealt
        # with, then tags its still-unsorted neighbours. The original rebuilt
        # the whole cKDTree and copied the whole unsorted array on every
        # iteration, so a plot with n cylinders paid O(n^2 log n) - which is
        # the bulk of the time on anything but a tiny plot.
        #
        # Cylinders are only ever removed from the unsorted set, never added,
        # so "lowest remaining" is just the next entry in one stable sort by
        # height, and one tree over all cylinders answers every neighbour query
        # once the already-sorted ones are filtered out. The processing order,
        # the neighbour sets and the tagging are identical to before.
        total_points = len(cylinder_array)
        if total_points <= 1:
            print("1.000\n")
            return np.zeros((0, cylinder_array.shape[1]))

        order = np.argsort(cylinder_array[:, 2], kind="stable")
        kdtree = spatial.cKDTree(cylinder_array[:, :3], leafsize=1000)
        unsorted = np.ones(total_points, dtype=bool)

        # The original stopped with one cylinder still unsorted and never
        # emitted it; keeping that so the output is unchanged.
        for count, current_index in enumerate(order[:-1]):
            if count % 200 == 0:
                print("\r", np.around(count / total_points, 3), end="")

            unsorted[current_index] = False
            current_point = cylinder_array[current_index]
            if current_point[self.cyl_dict["tree_id"]] == 0:
                current_point[self.cyl_dict["tree_id"]] = max_tree_label
                max_tree_label += 1

            neighbours = np.asarray(
                kdtree.query_ball_point(current_point[:3], r=distance_tolerance), dtype=np.intp
            )
            neighbours = neighbours[unsorted[neighbours]]
            if neighbours.shape[0] > 0:
                cylinder_array[neighbours] = criteria_check(
                    current_point, cylinder_array[neighbours], angle_tolerance, search_angle
                )
        print("1.000\n")
        return cylinder_array[order[:-1]]

    @classmethod
    def make_cyl_visualisation(cls, cyl):
        """Creates a 3D point cloud representation of a circle."""
        p = MeasureTree.create_3d_circles_as_points_flat(cyl[0], cyl[1], cyl[2], cyl[6])
        points = MeasureTree.rodrigues_rot(p - cyl[:3], [0, 0, 1], cyl[3:6])
        points = np.hstack((points + cyl[:3], np.zeros((points.shape[0], 8))))
        points[:, -8:] = cyl[-8:]
        return points

    @classmethod
    def make_cyl_visualisation_array(cls, cyl_array, circle_points=15, chunk=100000):
        """
        Point-cloud representation of every cylinder in cyl_array.

        Same output as stacking make_cyl_visualisation over the rows, but done
        as array arithmetic. The per-cylinder version was dispatched one
        cylinder per task through the process pool, so each of the hundreds of
        thousands of cylinders in a plot paid a pickle round trip to produce 15
        points; the pool overhead dwarfed the arithmetic. On the example plot
        the two visualisation passes were 21 s of a 55 s measurement stage.
        """
        n = cyl_array.shape[0]
        if n == 0:
            return np.zeros((0, 11))

        angles = np.linspace(0, 2 * np.pi, circle_points)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        out = np.empty((n * circle_points, 11))

        for start in range(0, n, chunk):
            block = cyl_array[start : start + chunk]
            m = block.shape[0]

            # Circle in the cylinder's own frame: create_3d_circles_as_points_flat
            # builds it around the centre and the centre is subtracted straight
            # back off, leaving (r cos, r sin, 0).
            r = block[:, 6]
            p = np.zeros((m, circle_points, 3))
            p[:, :, 0] = r[:, None] * cos_a
            p[:, :, 1] = r[:, None] * sin_a

            # Rodrigues rotation from [0, 0, 1] onto each cylinder's axis.
            # k = cross([0, 0, 1], n) = [-n_y, n_x, 0].
            with np.errstate(invalid="ignore", divide="ignore"):
                normal = block[:, 3:6] / np.linalg.norm(block[:, 3:6], axis=1)[:, None]
            k = np.zeros((m, 3))
            k[:, 0] = -normal[:, 1]
            k[:, 1] = normal[:, 0]
            k_norm = np.linalg.norm(k, axis=1)
            nonzero = k_norm > 0
            k[nonzero] /= k_norm[nonzero, None]
            theta = np.arccos(np.clip(normal[:, 2], -1.0, 1.0))
            cos_t = np.cos(theta)[:, None, None]
            sin_t = np.sin(theta)[:, None, None]

            k_cross_p = np.cross(np.broadcast_to(k[:, None, :], p.shape), p)
            k_dot_p = np.einsum("mij,mj->mi", p, k)
            rotated = p * cos_t + k_cross_p * sin_t + k_dot_p[:, :, None] * k[:, None, :] * (1 - cos_t)
            rotated += block[:, None, :3]

            sl = slice(start * circle_points, (start + m) * circle_points)
            out[sl, :3] = rotated.reshape(-1, 3)
            out[sl, 3:] = np.repeat(block[:, -8:], circle_points, axis=0)

        return out

    @classmethod
    def points_along_line(cls, x0, y0, z0, x1, y1, z1, resolution=0.05):
        """Creates a point cloud representation of a line."""
        points_per_line = int(np.linalg.norm(np.array([x1, y1, z1]) - np.array([x0, y0, z0])) / resolution)
        Xs = np.atleast_2d(np.linspace(x0, x1, points_per_line)).T
        Ys = np.atleast_2d(np.linspace(y0, y1, points_per_line)).T
        Zs = np.atleast_2d(np.linspace(z0, z1, points_per_line)).T
        return np.hstack((Xs, Ys, Zs))

    @classmethod
    def create_3d_circles_as_points_flat(cls, x, y, z, r, circle_points=15):
        """Creates a point cloud representation of a horizontal circle at coordinates x, y, z. and of radius r.
        Circle points is the number of points to use to represent each circle."""
        # One array instead of a vstack per point. This runs once per cylinder
        # in the visualisation stage, and a plot has hundreds of thousands.
        angle_between_points = np.linspace(0, 2 * np.pi, circle_points)
        points = np.empty((circle_points, 3))
        points[:, 0] = r * np.cos(angle_between_points) + x
        points[:, 1] = r * np.sin(angle_between_points) + y
        points[:, 2] = z
        return points

    @classmethod
    def rodrigues_rot(cls, points, vector1, vector2):
        """RODRIGUES ROTATION
        - Rotate given points based on a starting and ending vector
        - Axis k and angle of rotation theta given by vectors n0,n1
        P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))"""
        if points.ndim == 1:
            points = points[np.newaxis, :]

        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        k = np.cross(vector1, vector2)

        # Guard on the magnitude of k, not its component sum. k is
        # cross(v1, [0,0,1]) = [v1_y, -v1_x, 0] in the calls made here, whose
        # sum is v1_y - v1_x: the old "np.sum(k) != 0" test skipped
        # normalisation for any axis with v1_x == v1_y, leaving k scaled by
        # sin(theta) and rotating the points wrongly. Genuinely-zero k (a stem
        # already parallel to the target vector) still skips, as before, and
        # gives the identity rotation.
        k_norm = np.linalg.norm(k)
        if k_norm != 0:
            k = k / k_norm
        theta = np.arccos(np.clip(np.dot(vector1, vector2), -1.0, 1.0))

        # Vectorised Rodrigues rotation. This was a Python loop over every
        # point that also recomputed cos(theta) and sin(theta) on each pass -
        # 3.4 ms of a 5.2 ms circle fit, and circle fitting is the bulk of the
        # measurement stage. Same formula, same result.
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return (
            points * cos_t
            + np.cross(k, points) * sin_t
            + np.outer(points @ k, k) * (1 - cos_t)
        )

    # Smallest scale below which a point set is treated as degenerate. Matches
    # the check skimage's CircleModel makes before normalising.
    _CIRCLE_TINY = np.finfo(np.float64).tiny

    @staticmethod
    def _circle_lstsq(points):
        """
        Algebraic least-squares circle fit to an (N, 2) point set.

        This is skimage's CircleModel estimator written out directly: centre and
        scale the data, solve the linear system for [2x, 2y, 1] . C = x^2 + y^2,
        then take the radius as the RMS distance of the points from the fitted
        centre. Returns (xc, yc, r), or None if the points are degenerate.
        """
        origin = points.mean(axis=0)
        d = points - origin
        scale = d.std()
        if scale < MeasureTree._CIRCLE_TINY:
            return None
        d = d / scale

        x = d[:, 0]
        y = d[:, 1]
        f = x * x + y * y
        # Normal equations for A = [2x, 2y, 1]. Solving these 3x3 systems is
        # what makes the batched version below possible; np.linalg.lstsq is not
        # batched, and calling it once per RANSAC trial was the single most
        # expensive thing in the whole measurement stage.
        n = points.shape[0]
        M = np.array(
            [
                [4 * (x * x).sum(), 4 * (x * y).sum(), 2 * x.sum()],
                [4 * (x * y).sum(), 4 * (y * y).sum(), 2 * y.sum()],
                [2 * x.sum(), 2 * y.sum(), float(n)],
            ]
        )
        b = np.array([2 * (x * f).sum(), 2 * (y * f).sum(), f.sum()])
        try:
            C = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(C)):
            return None

        centre = C[:2]
        r = np.sqrt(np.mean((x - centre[0]) ** 2 + (y - centre[1]) ** 2))
        return centre[0] * scale + origin[0], centre[1] * scale + origin[1], r * scale

    @staticmethod
    def _circle_lstsq_batch(samples):
        """
        The same fit as _circle_lstsq, applied to a stack of (T, k, 2) samples.

        Returns (centres, radii, valid) where centres is (T, 2), radii is (T,)
        and valid is a (T,) boolean marking the trials whose normal equations
        were solvable. Invalid trials are the batched equivalent of skimage
        rejecting a sample for having rank < 3.
        """
        origin = samples.mean(axis=1, keepdims=True)
        d = samples - origin
        scale = d.reshape(d.shape[0], -1).std(axis=1)
        valid = scale >= MeasureTree._CIRCLE_TINY
        # Avoid dividing by zero in the degenerate rows; they are masked out.
        safe_scale = np.where(valid, scale, 1.0)
        d = d / safe_scale[:, None, None]

        x = d[:, :, 0]
        y = d[:, :, 1]
        f = x * x + y * y
        k = float(samples.shape[1])

        sxx = np.einsum("tk,tk->t", x, x)
        sxy = np.einsum("tk,tk->t", x, y)
        syy = np.einsum("tk,tk->t", y, y)
        sx = x.sum(axis=1)
        sy = y.sum(axis=1)

        M = np.empty((samples.shape[0], 3, 3))
        M[:, 0, 0] = 4 * sxx
        M[:, 0, 1] = M[:, 1, 0] = 4 * sxy
        M[:, 0, 2] = M[:, 2, 0] = 2 * sx
        M[:, 1, 1] = 4 * syy
        M[:, 1, 2] = M[:, 2, 1] = 2 * sy
        M[:, 2, 2] = k

        b = np.empty((samples.shape[0], 3))
        b[:, 0] = 2 * np.einsum("tk,tk->t", x, f)
        b[:, 1] = 2 * np.einsum("tk,tk->t", y, f)
        b[:, 2] = f.sum(axis=1)

        # A near-zero determinant is the batched stand-in for a rank-deficient
        # design matrix. np.linalg.solve raises on the whole batch if any one
        # matrix is singular, so the bad rows are filtered out first.
        det = np.linalg.det(M)
        valid &= np.abs(det) > 1e-12 * np.maximum(np.abs(M).max(axis=(1, 2)) ** 3, 1e-12)

        centres = np.zeros((samples.shape[0], 2))
        radii = np.zeros(samples.shape[0])
        if np.any(valid):
            # b is a stack of right-hand-side vectors, not one matrix, so it
            # needs the trailing axis for the batched gufunc signature.
            C = np.linalg.solve(M[valid], b[valid][:, :, None])[:, :, 0]
            good = np.all(np.isfinite(C), axis=1)
            idx = np.flatnonzero(valid)[good]
            C = C[good]
            valid[:] = False
            valid[idx] = True

            cx = C[:, 0]
            cy = C[:, 1]
            dx = x[idx] - cx[:, None]
            dy = y[idx] - cy[:, None]
            radii[idx] = np.sqrt(np.mean(dx * dx + dy * dy, axis=1)) * scale[idx]
            centres[idx, 0] = cx * scale[idx] + origin[idx, 0, 0]
            centres[idx, 1] = cy * scale[idx] + origin[idx, 0, 1]

        return centres, radii, valid

    @staticmethod
    def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability=0.99):
        """Trials still needed to see an all-inlier sample with `probability`.

        Same formula skimage uses to shrink max_trials once a good consensus
        set has been found.
        """
        if n_inliers == 0:
            return np.inf
        inlier_ratio = n_inliers / n_samples
        denom = 1 - inlier_ratio**min_samples
        if denom <= 0:
            return 1
        if denom >= 1:
            return np.inf
        return int(np.ceil(np.log(1 - probability) / np.log(denom)))

    @classmethod
    def ransac_circle(cls, points, min_samples, residual_threshold, max_trials, rng, chunk=64):
        """
        RANSAC circle fit, equivalent to skimage's ransac(..., CircleModel, ...)
        but with the trials evaluated in batches instead of one Python loop
        iteration each.

        skimage draws a sample, calls np.linalg.lstsq on it, and scores it
        against every point - all in Python, once per trial. With
        min_samples = 30% of the points the early-stopping criterion almost
        never fires on a noisy stem slice, so it ran the full 10,000 trials:
        measured at 1.7 s for a single circle, and a plot needs thousands of
        them. Here a chunk of trials is sampled, fitted and scored with a
        handful of array operations, and the dynamic stopping rule is applied
        after each chunk. Same model and same scoring as skimage. The stopping
        rule is skimage's formula at probability 0.99, where upstream FSCT
        left skimage's default of 1 - so this is not the same criterion, but
        with min_samples at 30% of the slice neither value ever lets the rule
        fire in practice, and every fit runs to max_trials either way
        (measured: 762 ms against 740 ms per fit on 60 real slices). The
        speed-up over the original comes from batching (2.6x) and from the
        lower max_trials and point cap in other_parameters.py (the rest of
        69x), not from the stopping rule.

        Returns (xc, yc, r) of the model refitted on the best consensus set, or
        None if no consensus set was found.
        """
        n = points.shape[0]
        min_samples = max(1, min(min_samples, n))

        best_inliers = None
        best_count = -1
        best_residual_sum = np.inf
        trials_done = 0
        trial_budget = max_trials

        while trials_done < trial_budget:
            t = int(min(chunk, trial_budget - trials_done))

            # Sample min_samples distinct indices per trial. argpartition over
            # a random matrix is the vectorised equivalent of
            # rng.choice(n, min_samples, replace=False) done t times.
            if min_samples >= n:
                idx = np.tile(np.arange(n), (t, 1))
            else:
                idx = np.argpartition(rng.random((t, n)), min_samples - 1, axis=1)[:, :min_samples]

            centres, radii, valid = cls._circle_lstsq_batch(points[idx])
            trials_done += t

            if np.any(valid):
                cv = centres[valid]
                rv = radii[valid]
                dist = np.sqrt(
                    (points[None, :, 0] - cv[:, None, 0]) ** 2 + (points[None, :, 1] - cv[:, None, 1]) ** 2
                )
                residuals = np.abs(rv[:, None] - dist)
                inlier_mask = residuals < residual_threshold
                counts = np.count_nonzero(inlier_mask, axis=1)
                residual_sums = np.einsum("tn,tn->t", residuals, residuals)

                # skimage keeps the first trial with the most inliers, breaking
                # ties on the smaller total squared residual. lexsort is stable,
                # so taking the first row of this ordering picks the same trial.
                order = np.lexsort((residual_sums, -counts))
                pick = order[0]
                if counts[pick] > best_count or (
                    counts[pick] == best_count and residual_sums[pick] < best_residual_sum
                ):
                    best_count = int(counts[pick])
                    best_residual_sum = float(residual_sums[pick])
                    best_inliers = inlier_mask[pick]
                    trial_budget = min(
                        trial_budget, cls._dynamic_max_trials(best_count, n, min_samples)
                    )

        if best_inliers is None or not best_inliers.any():
            return None
        return cls._circle_lstsq(points[best_inliers])

    @classmethod
    def fit_circle_3D(cls, points, V, fix_cci_sectors=False, fit_options=None):
        """
        Fits a circle using Random Sample Consensus (RANSAC) to a set of points in a plane perpendicular to vector V.

        Args:
            points: Set of points to fit a circle to using RANSAC.
            V: Axial vector of the cylinder you're fitting.
            fit_options: optional dict with "max_trials", "max_points" and "rng"
                         controlling the RANSAC fit. See other_parameters.py.

        Returns:
            cyl_output: numpy array of the format [[x, y, z, x_norm, y_norm, z_norm, radius, CCI, 0, 0, 0, 0, 0, 0]]
        """

        if fit_options is None:
            fit_options = {}
        max_trials = fit_options.get("max_trials", 1000)
        max_points = fit_options.get("max_points", 0)
        rng = fit_options.get("rng")
        if rng is None:
            rng = np.random.default_rng()

        CCI = 0
        r = 0
        P = points[:, :3]
        P_mean = np.mean(P, axis=0)
        P_centered = P - P_mean
        normal = V / np.linalg.norm(V)
        if normal[2] < 0:  # if normal vector is pointing down, flip it around the other way.
            normal = normal * -1

        # Project points to coords X-Y in 2D plane
        P_xy = MeasureTree.rodrigues_rot(P_centered, normal, [0, 0, 1])

        # Fit circle in new 2D coords with RANSAC
        if P_xy.shape[0] >= 20:
            fit_points = P_xy[:, :2]
            # RANSAC cost is linear in the number of points for every trial, and
            # a slice with thousands of returns describes the same circle as a
            # well-spread sample of it. The CCI below still uses every point.
            if 0 < max_points < fit_points.shape[0]:
                fit_points = fit_points[rng.choice(fit_points.shape[0], max_points, replace=False)]

            model = MeasureTree.ransac_circle(
                fit_points,
                min_samples=int(fit_points.shape[0] * 0.3),
                residual_threshold=0.05,
                max_trials=max_trials,
                rng=rng,
            )
            if model is None:
                xc, yc = np.mean(P_xy[:, :2], axis=0)
                r = 0
            else:
                xc, yc, r = model
                CCI = MeasureTree.circumferential_completeness_index(
                    [xc, yc], r, P_xy[:, :2], fix_cci_sectors=fix_cci_sectors
                )

        if CCI < 0.3:
            r = 0
            xc, yc = np.mean(P_xy[:, :2], axis=0)
            CCI = 0

        # Transform circle center back to 3D coords
        cyl_centre = MeasureTree.rodrigues_rot(np.array([[xc, yc, 0]]), [0, 0, 1], normal) + P_mean
        cyl_output = np.array(
            [
                [
                    cyl_centre[0, 0],
                    cyl_centre[0, 1],
                    cyl_centre[0, 2],
                    normal[0],
                    normal[1],
                    normal[2],
                    r,
                    CCI,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ]
        )
        return cyl_output

    def point_cloud_annotations(self, character_size, xpos, ypos, zpos, offset, text):
        """
        Point based text visualisation. Makes text viewable as a point cloud.

        Args:
            character_size:
            xpos: x coord.
            ypos: y coord.
            zpos: z coord.
            offset: Offset for the x coord. Used to shift the text depending on tree radius.
            text: The text to be displayed.

        Returns:
            nx3 point cloud of the text.
        """

        def convert_character_cells_to_points(character):
            character = np.rot90(character, axes=(1, 0))
            # np.argwhere returns the indices of the set cells in row-major
            # order, which is exactly the order the nested loop visited them,
            # so the resulting point cloud is unchanged. The loop it replaces
            # walked every cell of the bitmap in Python and re-allocated the
            # whole point array for each lit one - and this runs seven times
            # per tree, once per line of the label.
            lit = np.argwhere(character == 1)
            points = np.zeros((lit.shape[0], 3))
            points[:, :2] = lit

            roll_mat = np.array(
                [[1, 0, 0], [0, np.cos(-np.pi / 4), -np.sin(-np.pi / 4)], [0, np.sin(-np.pi / 4), np.cos(-np.pi / 4)]]
            )
            points = np.dot(points, roll_mat)
            return points

        def get_character(char):
            if char == ":":
                return self.character_viz[self.characters.index("semiC")]
            elif char == ".":
                return self.character_viz[self.characters.index("dot")]
            elif char == " ":
                return self.character_viz[self.characters.index("space")]
            elif char == "M":
                return self.character_viz[self.characters.index("_M")]
            else:
                return self.character_viz[self.characters.index(char)]

        text_points = np.zeros((11, 0))
        for i in text:
            text_points = np.hstack((text_points, np.array(get_character(str(i)))))
        points = convert_character_cells_to_points(text_points)

        points = points * character_size + [xpos + 0.2 + 0.5 * offset, ypos, zpos]
        return points

    @staticmethod
    def _plane_slice_index(point_cloud, sort_order, sorted_coord, axis, weights, line_centre, half_length):
        """
        Indices of the points within `half_length` of the plane at line_centre.

        The distance used is the one the original code computes,
        ``norm(|v * (p - c)|)``, i.e. ``sqrt(sum_i v_i^2 (p_i - c_i)^2)``, so
        `weights` is ``v ** 2``. Every coordinate contributes a non-negative
        term, so a point can only qualify if its `axis` coordinate alone is
        within ``half_length / |v_axis|`` of the centre. `axis` is the largest
        component of v - for a near-vertical stem that is z, and the stems are
        exactly what this is slicing - so a binary search on the pre-sorted
        coordinate discards nearly the whole cluster before any arithmetic.

        The original scanned all N points of the cluster for each of the M
        skeleton positions; a big stem is tens of thousands of points and
        dozens of positions. Indices are returned in the cluster's original
        order so the slice is identical to the one the scan produced.
        """
        v_axis = abs(weights[axis]) ** 0.5
        if v_axis > 0:
            span = half_length / v_axis
            lo = np.searchsorted(sorted_coord, line_centre[axis] - span, side="left")
            hi = np.searchsorted(sorted_coord, line_centre[axis] + span, side="right")
            candidates = sort_order[lo:hi]
        else:
            candidates = sort_order

        if candidates.shape[0] == 0:
            return candidates

        offsets = point_cloud[candidates] - line_centre
        keep = np.sqrt(np.einsum("ij,ij->i", offsets * offsets, np.broadcast_to(weights, offsets.shape)))
        return np.sort(candidates[keep < half_length])

    @classmethod
    def fit_cylinder(cls, skeleton_points, point_cloud, num_neighbours, fix_cci_sectors=False, fit_options=None):
        """
        Fits a 3D line to the skeleton points cluster provided.
        Uses this line as the major axis/axial vector of the cylinder to be fitted.
        Fits a series of circles perpendicular to this axis to the point cloud of this particular stem segment.

        Args:
            skeleton_points: A single cluster of skeleton points which should represent a segment of a tree/branch.
            point_cloud: The cluster of points belonging to the segment of the branch.
            num_neighbours: The number of skeleton points to use for fitting each circle in the segment. lower numbers
                            have fewer points to fit a circle to, but higher numbers are negatively affected by curved
                            branches. Recommend leaving this as it is.
            fit_options: passed straight through to fit_circle_3D.

        Returns:
            cyl_array: a numpy array based representation of the fitted circles/cylinders.
        """
        point_cloud = point_cloud[:, :3]
        skeleton_points = skeleton_points[:, :3]
        cyl_array = np.zeros((0, 14))
        line_centre = np.mean(skeleton_points[:, :3], axis=0)
        _, _, vh = np.linalg.svd(line_centre - skeleton_points)
        line_v_hat = vh[0] / np.linalg.norm(vh[0])

        # Sorted once per cluster and reused by every plane slice below.
        weights = line_v_hat**2
        axis = int(np.argmax(np.abs(line_v_hat)))
        sort_order = np.argsort(point_cloud[:, axis], kind="stable")
        sorted_coord = point_cloud[sort_order, axis]

        if skeleton_points.shape[0] <= num_neighbours:
            group = skeleton_points
            line_centre = np.mean([np.min(group[:, :3], axis=0), np.max(group[:, :3], axis=0)], axis=0)
            length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
            plane_slice = point_cloud[
                cls._plane_slice_index(
                    point_cloud, sort_order, sorted_coord, axis, weights, line_centre, length / 2
                )
            ]
            if plane_slice.shape[0] > 0:
                cylinder = MeasureTree.fit_circle_3D(plane_slice, line_v_hat, fix_cci_sectors, fit_options)
                cyl_array = np.vstack((cyl_array, cylinder))
        else:
            # The loop walks the skeleton from the bottom up, taking the
            # lowest remaining point and its num_neighbours nearest, then
            # dropping that point. The original rebuilt a whole sklearn
            # NearestNeighbors index on every pass - O(M) index builds for M
            # skeleton points, each with the constructor and validation
            # overhead that dominates at these sizes.
            #
            # Instead: sort by height once, so "lowest remaining" is just the
            # next row, and compute the pairwise distance matrix once. Each
            # pass is then an argpartition over the remaining columns. Same
            # neighbours, same order of processing.
            order = np.argsort(skeleton_points[:, 2], kind="stable")
            sorted_points = skeleton_points[order]
            total = sorted_points.shape[0]

            cyl_list = []
            for start in range(total - num_neighbours):
                # Remaining points are exactly sorted_points[start:], because
                # the lowest point is removed at the end of each pass.
                # Compute just this row of the distance matrix - the full
                # matrix would be M x M and blow up on large skeletons.
                remaining = sorted_points[start:, :3]
                offsets = remaining - sorted_points[start, :3]
                row = np.einsum("ij,ij->i", offsets, offsets)  # squared distances

                nearest = np.argpartition(row, num_neighbours - 1)[:num_neighbours]
                # argpartition does not order within the selection; sort so the
                # neighbour set matches kneighbors' ascending-distance output.
                nearest = nearest[np.argsort(row[nearest], kind="stable")] + start
                group = sorted_points[nearest]

                line_centre = np.mean(group[:, :3], axis=0)
                length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
                plane_slice = point_cloud[
                    cls._plane_slice_index(
                        point_cloud, sort_order, sorted_coord, axis, weights, line_centre, length / 2
                    )
                ]
                if plane_slice.shape[0] > 0:
                    cyl_list.append(
                        MeasureTree.fit_circle_3D(plane_slice, line_v_hat, fix_cci_sectors, fit_options)
                    )

            if cyl_list:
                cyl_array = np.vstack([cyl_array] + cyl_list)
        return cyl_array

    @classmethod
    def cylinder_cleaning_multithreaded(cls, args):
        def compute_frustum_volume(radius_1, radius_2, height):
            """
            Volume of the conical frustum between two cylinders.

            Two separate errors used to compound here, and tree_data's
            "Volume_1" is the sum of one of these per cylinder of the tree:

            1. The formula read "r1**2 + r1 + r2**2 + r2". A frustum is
               (1/3) pi h (r1^2 + r1 r2 + r2^2); adding a length to an area is
               dimensionally meaningless and there is no radius for which the
               two agree. At r1 = 0.5 m, r2 = 0.3 m it overstated the segment
               by 2.3x.
            2. The parameters were named diameter_1/diameter_2 and halved on
               entry, but the only caller - _stacked_frustum_volume - passes
               cyl_dict["radius"], which is a radius (DBH is computed from it
               as radius * 2). So every radius was halved a second time,
               scaling the result by a further 1/4.

            Volume_2, which is estimated from DBH and height rather than from
            the cylinders, never went through this, which is why the two
            volume columns disagreed so widely.
            """
            volume = (1 / 3) * np.pi * height * (radius_1**2 + radius_1 * radius_2 + radius_2**2)
            return volume

        """
        Cylinder Cleaning
        Works on a single tree worth of cylinders at a time.
        Starts at the lowest (z axis) cylinder.
        Finds neighbouring cylinders within "cleaned_measurement_radius".
        If no neighbours are found, cylinder is deleted.
        If neighbours are found, find the neighbour with the highest circumferential completeness index (CCI). This is
        probably the most trustworthy cylinder in the neighbourhood.

        If there are enough neighbours, use those with CCI >= the 30th percentile of CCIs in the neighbourhood.
        Use the medians of x, y, vx, vy, vz, radius as the cleaned cylinder values.
        Use the mean of the z coords of all neighbours for the cleaned cylinder z coord.
        """
        sorted_cylinders, cleaned_measurement_radius, cyl_dict = args
        n_cyls = sorted_cylinders.shape[0]

        tree = BallTree(sorted_cylinders[:, :3])
        _, ind = tree.query(sorted_cylinders[:, :3], k=min(10, n_cyls))
        # One median over the (n, k) gather rather than a Python median per row.
        sorted_cylinders[:, cyl_dict["radius"]] = np.median(sorted_cylinders[ind, cyl_dict["radius"]], axis=1)

        # Both loops below walk the tree's cylinders from the bottom up,
        # consuming a neighbourhood at a time. The originals rebuilt a spatial
        # index and copied the whole array on every pass, which is O(n^2 log n)
        # for a tree of n cylinders - and interpolation gives a tall tree
        # thousands of them. Cylinders are only removed, never added, so one
        # index over all of them plus a stable sort by height reproduces the
        # same processing order, the same neighbourhoods and the same output.
        order = np.argsort(sorted_cylinders[:, 2], kind="stable")
        kdtree = spatial.cKDTree(sorted_cylinders[:, :3])
        active = np.ones(n_cyls, dtype=bool)
        remaining = n_cyls
        cleaned_list = []

        for start_idx in order:
            if remaining <= 2:
                break
            if not active[start_idx]:
                continue

            active[start_idx] = False
            remaining -= 1
            # Copied because the "no better neighbour" branch below writes
            # through best_cylinder, and the row is still part of the array
            # the neighbour queries read from.
            start_point = sorted_cylinders[start_idx].copy()

            results = np.asarray(
                kdtree.query_ball_point(start_point[:3], cleaned_measurement_radius), dtype=np.intp
            )
            results = results[active[results]]
            neighbours = np.vstack((sorted_cylinders[results], start_point))
            best_cylinder = start_point

            if neighbours.shape[0] > 0:
                if np.max(neighbours[:, cyl_dict["CCI"]]) > 0:
                    best_cylinder = neighbours[np.argsort(neighbours[:, cyl_dict["CCI"]])][
                        -1
                    ]  # choose cyl with highest CCI.
                # compute 50th percentile of CCI in neighbourhood
                percentile_thresh = np.percentile(neighbours[:, cyl_dict["CCI"]], 50)
                if neighbours[neighbours[:, cyl_dict["CCI"]] >= percentile_thresh, :2].shape[0] > 0:
                    best_cylinder[:3] = np.median(
                        neighbours[neighbours[:, cyl_dict["CCI"]] >= percentile_thresh, :3], axis=0
                    )
                    best_cylinder[3:6] = np.median(
                        neighbours[neighbours[:, cyl_dict["CCI"]] >= percentile_thresh, 3:6], axis=0
                    )
                    best_cylinder[cyl_dict["radius"]] = np.max(
                        neighbours[neighbours[:, cyl_dict["CCI"]] >= percentile_thresh, cyl_dict["radius"]], axis=0
                    )
            cleaned_list.append(best_cylinder)
            if results.shape[0] > 0:
                active[results] = False
                remaining -= results.shape[0]

        if cleaned_list:
            cleaned_cyls = np.vstack(cleaned_list)
        else:
            cleaned_cyls = np.zeros((0, sorted_cylinders.shape[1]))

        cleaned_cyls[:, cyl_dict["tree_volume"]] = cls._stacked_frustum_volume(
            cleaned_cyls, cyl_dict, compute_frustum_volume
        )
        return cleaned_cyls

    @staticmethod
    def _stacked_frustum_volume(cleaned_cyls, cyl_dict, compute_frustum_volume):
        """
        Sums the frustum volumes between each cylinder and its nearest
        neighbour above it, working up from the lowest.

        Same quantity the original computed, without rebuilding a BallTree per
        cylinder: one index answers all the queries, and the nearest neighbour
        "above" is the first returned neighbour whose height rank is higher.
        The rare case where all k returned neighbours sit below falls back to a
        direct scan of the cylinders above.
        """
        n = cleaned_cyls.shape[0]
        if n <= 2:
            return 0.0

        order = np.argsort(cleaned_cyls[:, 2], kind="stable")
        rank = np.empty(n, dtype=np.intp)
        rank[order] = np.arange(n)

        pts = cleaned_cyls[:, :3]
        radii = cleaned_cyls[:, cyl_dict["radius"]]
        k = min(n, 16)
        dists, idxs = spatial.cKDTree(pts).query(pts, k=k)

        volume = 0.0
        for i in range(n - 2):
            p = order[i]
            above = rank[idxs[p]] > i
            if above.any():
                j = int(np.argmax(above))
                neighbour, distance = idxs[p][j], dists[p][j]
            else:
                suffix = order[i + 1 :]
                d = np.linalg.norm(pts[suffix] - pts[p], axis=1)
                m = int(np.argmin(d))
                neighbour, distance = suffix[m], d[m]
            volume += compute_frustum_volume(radii[p], radii[neighbour], distance)
        return volume

    @staticmethod
    def _rows_by_value(values):
        """
        Map each distinct entry of `values` to the row indices that hold it.

        The indices for each value are ascending, so array[rows] gives the
        same rows in the same order as array[values == value] would. Built
        once with a stable sort, it replaces a full boolean scan of the array
        per tree in the loops below - on a large plot those arrays are the
        vegetation and stem point clouds, millions of rows, scanned once for
        every tree.
        """
        order = np.argsort(values, kind="stable")
        unique_values, starts = np.unique(values[order], return_index=True)
        ends = np.append(starts[1:], values.shape[0])
        return {value: order[start:end] for value, start, end in zip(unique_values, starts, ends)}

    @staticmethod
    def inside_conv_hull(point, hull, tolerance=1e-5):
        """Checks if a point is inside a convex hull."""
        return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

    @classmethod
    def circumferential_completeness_index(
        cls, fitted_circle_centre, estimated_radius, slice_points, fix_cci_sectors=False
    ):
        """
        Computes the Circumferential Completeness Index (CCI) of a fitted circle.

        Args:
            fitted_circle_centre: x, y coords of the circle centre
            estimated_radius: circle radius
            slice_points: the points the circle was fitted to

        Returns:
            CCI
        """
        sector_angle = 4.5  # degrees
        num_sections = int(np.ceil(360 / sector_angle))
        sectors = np.linspace(-180, 180, num=num_sections, endpoint=False)

        # `sectors` is in degrees, but the cos/sin below treat it as radians,
        # so the "evenly spaced sectors" are really 80 arbitrary directions.
        # They happen to land quasi-uniformly around the circle (a 4.5 rad
        # step is incommensurate with 2*pi), which is why CCI still roughly
        # tracks circumferential coverage - but the values are not what the
        # method describes, and CCI decides which cylinders survive.
        #
        # Fixing it moves every downstream number (stem count, DBH, height),
        # so it is opt-in via fix_cci_sectors in other_parameters.py.
        if fix_cci_sectors:
            sectors = np.radians(sectors)

        centre_vectors = slice_points[:, :2] - fitted_circle_centre
        norms = np.linalg.norm(centre_vectors, axis=1)

        centre_vectors = centre_vectors / np.atleast_2d(norms).T
        centre_vectors = centre_vectors[
            np.logical_and(norms >= 0.8 * estimated_radius, norms <= 1.2 * estimated_radius)
        ]

        sector_vectors = np.vstack((np.cos(sectors), np.sin(sectors))).T

        if centre_vectors.shape[0] == 0:
            return 0.0

        # One (points x sectors) matrix product instead of a Python loop doing
        # an einsum per sector. Comparing the cosine against cos(half sector)
        # avoids the arccos entirely; cos is monotonically decreasing on
        # [0, pi], so "angle < half" is exactly "cosine > cos(half)".
        cos_threshold = np.cos(np.radians(sector_angle / 2))
        dots = centre_vectors @ sector_vectors.T
        CCI = np.count_nonzero(np.any(dots > cos_threshold, axis=0)) / num_sections

        return CCI

    # One spawn Pool shared by every parallel stage of a measurement run.
    _pool = None
    _pool_processes = None

    @classmethod
    def _get_pool(cls, processes):
        """
        The measurement run's worker pool, created on first use.

        On Windows a spawned worker re-imports this module and everything under
        it - numpy, scipy, sklearn, hdbscan, networkx, pandas - which costs a
        few seconds per pool. The stage used to build a fresh pool for slice
        clustering, for cylinder fitting and again for cylinder cleaning, so
        that startup was paid three times over and on a small plot it was
        larger than the work itself.
        """
        if cls._pool is not None and cls._pool_processes == processes:
            return cls._pool
        cls.close_pool()

        # Pin each worker's BLAS to a single thread.
        #
        # OpenBLAS defaults to one thread per core, so N worker processes on an
        # N-core machine try to run N*N BLAS threads - 256 here. Each thread
        # reserves its own buffer from a fixed pool, which runs dry:
        #
        #     OpenBLAS error: Memory allocation still failed after 10 retries
        #
        # It is also pure waste: these workers fit circles to a hundred-odd
        # points, far too small for threaded BLAS to pay off, and heavily
        # oversubscribing the CPU makes the whole stage slower. The variables
        # must be set before the children start, because a spawned worker
        # reads them while importing numpy, long before any initializer runs.
        thread_vars = (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        saved = {name: os.environ.get(name) for name in thread_vars}
        for name in thread_vars:
            os.environ[name] = "1"
        try:
            cls._pool = get_context("spawn").Pool(processes=processes, initializer=_initialise_worker)
            cls._pool_processes = processes
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        return cls._pool

    @classmethod
    def close_pool(cls):
        """
        Shuts the shared worker pool down. Safe to call when there isn't one.

        terminate() rather than close(): this is only called once every stage
        has either collected all of its results or given up on them, so there
        is never outstanding work to wait for. What close() + join() would wait
        for is the pool's result cache to drain - and after a worker has died
        with a task in hand (see pool_map_batched), the entry for that task
        never drains, so join() blocked forever and the error the watchdog had
        raised never reached the user. terminate() is what Pool.__exit__ does.
        """
        if cls._pool is not None:
            cls._pool.terminate()
            cls._pool.join()
            cls._pool = None
            cls._pool_processes = None

    @classmethod
    def pool_map_batched(cls, func, items, processes, label="", batch_size=None, costs=None):
        """
        Run func over items in the shared spawn Pool with a bounded number of
        tasks in flight, returning the results in input order.

        multiprocessing.Pool.imap pickles the *entire* iterable into the task
        pipe as fast as its feeder thread can manage, without waiting for the
        workers to drain it. Each task here carries numpy point arrays, so a
        plot with thousands of stem clusters queued gigabytes of pending
        overlapped writes and Windows eventually refused:

            OSError: [WinError 1450] Insufficient system resources exist to
            complete the requested service

        The first fix submitted fixed-size batches and waited for each batch
        to finish before sending the next. That bounded the backlog but put a
        barrier after every batch: the whole pool sat idle until the slowest
        task in the batch - the main stem of a big tree, typically - was done.
        On the example plot cylinder fitting used 32 s of worker CPU across 16
        workers and still took 7.4 s of wall time.

        Now the window slides. A semaphore lets the feeder run at most
        batch_size tasks ahead of the results, and results are taken in
        completion order and slotted back by index, so a long task never
        holds up the submission of the ones behind it. `costs`, if given, is a
        per-item estimate of work; items are submitted largest first so the
        long tasks start at the front and the short ones fill in around them.
        Every task's result depends only on its own input (the random fits
        are seeded per cluster), so neither the submission order nor the
        completion order changes any output - only how long it takes.
        """
        items = list(items)
        total = len(items)
        if total == 0:
            return []

        if processes is None or processes < 2 or total == 1:
            # Not worth a pickle round trip per item.
            results = [func(item) for item in items]
            if label:
                print("\r", total, "/", total, end="")
                print("\nDone\n")
            return results

        if batch_size is None:
            # Enough to keep every worker fed several times over, small enough
            # that the queued payload stays bounded.
            batch_size = max(processes * 8, 64)

        if costs is not None:
            submission_order = np.argsort(-np.asarray(costs, dtype=np.float64), kind="stable")
        else:
            submission_order = np.arange(total)

        pool = cls._get_pool(processes)
        in_flight = threading.BoundedSemaphore(batch_size)
        abandoned = threading.Event()

        def gated_tasks():
            # Runs on the pool's feeder thread. Blocks once batch_size tasks
            # are outstanding; the timeout lets it notice a failure on the
            # consumer side and stop, otherwise a raised exception below would
            # leave this generator - and so the pool's task thread - blocked
            # forever and close_pool() would never return.
            for index in submission_order:
                while not in_flight.acquire(timeout=0.1):
                    if abandoned.is_set():
                        return
                if abandoned.is_set():
                    return
                yield int(index), func, items[index]

        results = [None] * total
        done = 0
        worker_pids = {process.pid for process in pool._pool}
        try:
            pending = pool.imap_unordered(_indexed_call, gated_tasks())
            while done < total:
                try:
                    index, result = pending.next(timeout=15)
                except StopIteration:
                    break
                except TimeoutError:
                    # No result for a while. That is normal for a big stem
                    # cluster, but it is also exactly what a lost task looks
                    # like: when a worker is killed - the OS reclaiming
                    # memory is the usual cause - Pool quietly starts a
                    # replacement and never re-runs the task it was holding,
                    # so imap waits for it forever. Then the GUI shows a run
                    # that never finishes and no error anywhere. A replaced
                    # worker shows up as a new pid in the pool, so fail with
                    # a message instead of hanging.
                    current_pids = {process.pid for process in pool._pool}
                    if current_pids != worker_pids:
                        lost = len(worker_pids - current_pids)
                        raise RuntimeError(
                            f"{lost} measurement worker process(es) died during {label or 'processing'} "
                            "and their work was lost. This usually means the machine ran out of "
                            "memory: close other applications, lower num_cpu_cores, or set "
                            "prewarm_worker_pool=False in other_parameters.py and run again."
                        )
                    continue
                in_flight.release()
                results[index] = result
                done += 1
                if label and done % 10 == 0:
                    print("\r", done, "/", total, end="")
        except BaseException:
            abandoned.set()
            raise

        if label:
            print("\r", total, "/", total, end="")
            print("\nDone\n")
        return results

    @classmethod
    def threaded_cyl_fitting(cls, args):
        """Helper function for multithreaded cylinder fitting."""
        skel_cluster, point_cluster, cluster_id, num_neighbours, cyl_dict, fit_options = args
        cyl_array = np.zeros((0, 14))
        if skel_cluster.shape[0] > num_neighbours:
            fit_options = dict(fit_options)
            # Seeded per cluster, so a rerun of the same plot fits the same
            # circles regardless of how the work is spread across workers.
            seed = fit_options.pop("seed", None)
            fit_options["rng"] = np.random.default_rng(None if seed is None else seed + int(cluster_id))
            cyl_array = cls.fit_cylinder(
                skel_cluster,
                point_cluster,
                num_neighbours=num_neighbours,
                fix_cci_sectors=fit_options.get("fix_cci_sectors", False),
                fit_options=fit_options,
            )
            cyl_array[:, cyl_dict["branch_id"]] = cluster_id
        return cyl_array

    @classmethod
    def slice_clustering(cls, new_slice, min_cluster_size):
        """Helper function for clustering stem slices and extracting the skeletons of these stems."""
        cluster_array_internal = np.zeros((0, 6))
        medians = np.zeros((0, 3))

        if new_slice.shape[0] > 1:
            new_slice = cluster_hdbscan(new_slice[:, :3], min_cluster_size)
            # Accumulated into lists and stacked once. The vstack-per-cluster
            # version reallocated and copied everything found so far on every
            # cluster, for every slice of the plot.
            cluster_parts = []
            median_parts = []
            for cluster_id in range(0, int(np.max(new_slice[:, -1])) + 1):
                cluster = new_slice[new_slice[:, -1] == cluster_id]
                median = np.median(cluster[:, :3], axis=0)
                median_parts.append(median)
                cluster_parts.append(np.hstack((cluster[:, :3], np.zeros((cluster.shape[0], 3)) + median)))
            if median_parts:
                medians = np.vstack(median_parts)
                cluster_array_internal = np.vstack(cluster_parts)
        return cluster_array_internal, medians

    @classmethod
    def threaded_slice_clustering(cls, args):
        """Helper function for multiprocessed slice clustering."""
        slice_points, min_cluster_size = args
        return cls.slice_clustering(slice_points, min_cluster_size)

    @classmethod
    def within_angle_tolerances(cls, normal1, normal2, angle_tolerance):
        """Checks if normal1 and normal2 are within "angle_tolerance"
        of each other."""
        norm1 = normal1 / np.atleast_2d(np.linalg.norm(normal1, axis=1)).T
        norm2 = normal2 / np.atleast_2d(np.linalg.norm(normal2, axis=1)).T
        dot = np.clip(np.einsum("ij, ij->i", norm1, norm2), a_min=-1, a_max=1)
        theta = np.degrees(np.arccos(dot))
        return abs((theta > 90) * 180 - theta) <= angle_tolerance

    @classmethod
    def within_search_cone(cls, normal1, vector1_2, search_angle):
        """Checks if the angle between vector1_2 and normal1 is less than search_angle."""
        norm1 = normal1 / np.linalg.norm(normal1)
        if not (vector1_2 == 0).all():
            norm2 = vector1_2 / np.linalg.norm(vector1_2)
            dot = np.clip(np.dot(norm1, norm2), -1, 1)
            theta = math.degrees(np.arccos(dot))
            return abs((theta > 90) * 180 - theta) <= search_angle
        else:
            return False

    def run_measurement_extraction(self):
        """Runs the measurement stage, tearing the shared worker pool down after."""
        try:
            return self._run_measurement_extraction()
        finally:
            MeasureTree.close_pool()

    def _run_measurement_extraction(self):
        import networkx as nx
        import pandas as pd
        from skspatial.objects import Plane

        skeleton_array = np.zeros((0, 3))
        cluster_array = np.zeros((0, 6))
        slice_heights = np.linspace(
            np.min(self.stem_points[:, 2]),
            np.max(self.stem_points[:, 2]),
            int(np.ceil((np.max(self.stem_points[:, 2]) - np.min(self.stem_points[:, 2])) / self.slice_increment)),
        )

        print("Making and clustering slices...")
        # Cutting the slices used to boolean-scan the whole stem cloud once per
        # slice - O(slices x points), and the slices are 5 cm apart over the
        # full height of the plot. One sort plus a pair of binary searches per
        # slice gives the same rows (indices are re-sorted so the slice keeps
        # the cloud's original ordering).
        z_order = np.argsort(self.stem_points[:, 2], kind="stable")
        z_sorted = self.stem_points[z_order, 2]
        slice_inputs = []
        for slice_height in slice_heights:
            lo = np.searchsorted(z_sorted, slice_height, side="left")
            hi = np.searchsorted(z_sorted, slice_height + self.slice_thickness, side="left")
            if hi > lo:
                # Only xyz is used downstream; sending the label and colour
                # columns too would triple what gets pickled to the workers.
                slice_inputs.append(
                    [self.stem_points[np.sort(z_order[lo:hi]), :3], self.parameters["min_cluster_size"]]
                )
        del z_order, z_sorted

        # HDBSCAN on each slice is independent, so the slices go over the same
        # process pool the cylinder fitting uses. Results come back in input
        # order, so the stacked arrays are identical to the serial version's.
        slice_results = MeasureTree.pool_map_batched(
            MeasureTree.threaded_slice_clustering,
            slice_inputs,
            self.num_cpu_cores,
            label="slice clustering",
            costs=[points.shape[0] for points, _ in slice_inputs],
        )
        del slice_inputs
        if slice_results:
            cluster_array = np.vstack([cluster_array] + [r[0] for r in slice_results])
            skeleton_array = np.vstack([skeleton_array] + [r[1] for r in slice_results])
        del slice_results

        print("Clustering skeleton...")
        try:
            skeleton_array = cluster_dbscan(skeleton_array[:, :3], eps=self.slice_increment * 1.5)
        except ValueError:
            raise DataQualityError("Failed to cluster tree skeletons.")
        # Just assigns random colours to the clusters to make it easier to see
        # different neighbouring groups.
        #
        # Was a loop over every cluster label that scanned the whole skeleton
        # array twice per label and re-stacked everything gathered so far -
        # cubic-ish in practice on a big plot, for a debugging colour. One
        # stable sort by label gives the same row order (labels ascending,
        # original order within each) in a single pass.
        skeleton_labels = skeleton_array[:, -1]
        unique_labels, label_index = np.unique(skeleton_labels, return_inverse=True)
        label_order = np.argsort(skeleton_labels, kind="stable")
        cluster_colours = np.random.randint(0, 10, size=unique_labels.shape[0]).astype(np.float64)
        skeleton_cluster_visualisation = np.hstack(
            (skeleton_array[label_order], cluster_colours[label_index[label_order]][:, np.newaxis])
        )
        del skeleton_labels, unique_labels, label_index, label_order, cluster_colours

        print("Saving skeleton and cluster array...")
        save_file(
            self.output_dir + "skeleton_cluster_visualisation.las",
            skeleton_cluster_visualisation,
            ["X", "Y", "Z", "cluster"],
        )

        print("Making kdtree...")
        # Assign unassigned skeleton points to the nearest group.
        #
        # As written this did nothing whatsoever, in two separate ways:
        #
        #   1. "skeleton_array[unassigned_bool, -1][mask] = ..." chains two
        #      fancy indexes. The first produces a copy, so the assignment
        #      landed in a temporary that was discarded on the next line.
        #   2. The tree was built from the unassigned points and queried with
        #      the unassigned points, so the neighbour it found was always
        #      another unassigned point, whose label is -1 by definition.
        #      Even had the write landed, it would have written -1 over -1.
        #
        # DBSCAN's noise points therefore stayed at -1, and the cluster loop
        # below starts at 0, so they were silently dropped from the plot.
        #
        # Doing what the comment describes - matching them against the
        # *assigned* points - recovers skeleton points that are currently
        # discarded, which changes which stems get cylinders fitted and so
        # moves tree count, DBH and height. That is a real change in results,
        # so it is opt-in, like fix_cci_sectors.
        if self.parameters.get("assign_unassigned_skeleton_points", False):
            unassigned_bool = skeleton_array[:, -1] == -1
            assigned_bool = ~unassigned_bool
            if np.any(unassigned_bool) and np.any(assigned_bool):
                assigned_points = skeleton_array[assigned_bool]
                kdtree = spatial.cKDTree(assigned_points[:, :3], leafsize=100000)
                distances, neighbours = kdtree.query(skeleton_array[unassigned_bool, :3], k=1)
                close_enough = distances < self.slice_increment * 3
                # One write, straight into the column, via the index array -
                # no intermediate copy to lose it in.
                unassigned_idx = np.flatnonzero(unassigned_bool)[close_enough]
                skeleton_array[unassigned_idx, -1] = assigned_points[neighbours[close_enough], -1]
                print("    Reassigned", unassigned_idx.shape[0], "unassigned skeleton points.")

        input_data = []
        i = 0
        max_i = int(np.max(skeleton_array[:, -1]) + 1)
        # leafsize 16, not 100000. A leaf of 100,000 points makes the tree a
        # single bucket, so the dual-tree query below degenerates into a
        # brute-force comparison of every skeleton point against every stem
        # point. Leaf size changes only how the same neighbour sets are
        # found, not which points are in them; this cut the query from 3.7 s
        # to 0.04 s on a 300k-point synthetic cloud.
        cl_kdtree = spatial.cKDTree(cluster_array[:, 3:], leafsize=16)
        cluster_ids = range(0, max_i)
        print("Making initial branch/stem section clusters...")

        # Carried through the args tuple rather than held on the class: the Pool
        # below uses "spawn", so the workers re-import this module fresh and
        # would not see any attribute set on the parent side.
        fit_options = {
            "fix_cci_sectors": self.parameters.get("fix_cci_sectors", False),
            "max_trials": self.parameters.get("circle_fit_trials", 1000),
            "max_points": self.parameters.get("circle_fit_max_points", 0),
            "seed": self.parameters.get("random_seed", 0),
        }

        # Group the skeleton points by cluster label once, and match every
        # skeleton point against the slice-cluster medians in one query.
        #
        # This used to scan the whole skeleton array with a boolean mask for
        # each cluster id, then build a fresh cKDTree over that cluster and
        # run query_ball_tree against the medians - O(clusters x skeleton
        # points) for the scans and one tree build per cluster on top. A
        # query_ball_tree result depends only on the individual source point
        # and the target tree, so one call over all skeleton points gives
        # exactly the per-point lists the per-cluster trees produced; grouping
        # those by label reproduces each cluster's index set, and a stable
        # sort keeps the skeleton rows in their original order within each.
        skeleton_labels = skeleton_array[:, -1].astype(np.intp)
        skeleton_kdtree = spatial.cKDTree(skeleton_array[:, :3], leafsize=16)
        median_matches = skeleton_kdtree.query_ball_tree(cl_kdtree, r=0.0001)
        label_order = np.argsort(skeleton_labels, kind="stable")
        # Start offset of each label 0..max_i-1 within label_order. Label -1
        # (unassigned) sorts first and is skipped, exactly as before.
        label_bounds = np.searchsorted(skeleton_labels[label_order], np.arange(max_i + 1))
        del skeleton_kdtree

        for cluster_id in cluster_ids:
            if i % 100 == 0:
                print("\r", i, "/", max_i, end="")
            i += 1
            skeleton_rows = label_order[label_bounds[cluster_id] : label_bounds[cluster_id + 1]]
            skel_cluster = skeleton_array[skeleton_rows, :3]
            # An integer index even when every list is empty; the float array
            # np.hstack returns for that case is not a valid index.
            results = np.unique(
                np.hstack([median_matches[row] for row in skeleton_rows] or [[]]).astype(np.intp)
            )
            cluster_array_clean = cluster_array[results, :3]
            input_data.append(
                [
                    skel_cluster[:, :3],
                    cluster_array_clean[:, :3],
                    cluster_id,
                    self.num_neighbours,
                    self.cyl_dict,
                    fit_options,
                ]
            )

        print("\r", max_i, "/", max_i, end="")
        print("\nDone\n")

        print("Starting multithreaded cylinder fitting... This can take a while.")
        outputlist = MeasureTree.pool_map_batched(
            MeasureTree.threaded_cyl_fitting,
            input_data,
            self.num_cpu_cores,
            label="cyl fitting",
            # One circle fit per skeleton point, each linear in the points
            # of its slice: skeleton size x point count tracks the cost.
            costs=[task[0].shape[0] * task[1].shape[0] for task in input_data],
        )
        # Free the task payloads before stacking - input_data holds a copy of
        # the stem points for every cluster.
        del input_data
        full_cyl_array = np.vstack(outputlist)
        del outputlist

        print("Deleting cyls with CCI less than:", self.parameters["minimum_CCI"])
        full_cyl_array = full_cyl_array[full_cyl_array[:, self.cyl_dict["CCI"]] >= self.parameters["minimum_CCI"]]

        # cyl_array = [x,y,z,nx,ny,nz,r,CCI,branch_id,tree_id,segment_volume,parent_branch_id]
        print("Saving cylinder array...")
        save_file(self.output_dir + "full_cyl_array.las", full_cyl_array, headers_of_interest=list(self.cyl_dict))
        # full_cyl_array, _ = load_file(self.output_dir + 'full_cyl_array.las',
        #                               headers_of_interest=list(self.cyl_dict))
        if 1:
            print("Making full_cyl visualisation...")
            initial_cyl_vis = self.make_cyl_visualisation_array(full_cyl_array)

            print("\nSaving cylinder visualisation...")
            save_file(self.output_dir + "initial_cyl_vis.las", initial_cyl_vis, headers_of_interest=self.cyl_vis_headers)
        print("Sorting Cylinders...")
        full_cyl_array = self.cylinder_sorting(
            full_cyl_array,
            angle_tolerance=self.parameters["sorting_angle_tolerance"],
            search_angle=self.parameters["sorting_search_angle"],
            distance_tolerance=self.parameters["sorting_search_radius"],
        )

        print("Correcting Cylinder assignments...")
        sorted_full_cyl_array = np.zeros((0, full_cyl_array.shape[1]))
        t_id = 1
        max_search_radius = self.parameters["max_search_radius"]
        min_points = 5
        max_search_angle = self.parameters["max_search_angle"]
        max_tree_id = np.unique(full_cyl_array[:, self.cyl_dict["tree_id"]]).shape[0]
        for tree_id in np.unique(full_cyl_array[:, self.cyl_dict["tree_id"]]):
            if int(tree_id) % 10 == 0:
                print("Tree ID", int(tree_id), "/", int(max_tree_id))
            tree = full_cyl_array[full_cyl_array[:, self.cyl_dict["tree_id"]] == int(tree_id)]
            tree_kdtree = spatial.cKDTree(sorted_full_cyl_array[:, :3], leafsize=1000)
            if tree.shape[0] >= min_points:
                lowest_point = tree[np.argmin(tree[:, 2])]
                highest_point = tree[np.argmax(tree[:, 2])]
                lowneighbours = sorted_full_cyl_array[
                    tree_kdtree.query_ball_point(lowest_point[:3], r=max_search_radius)
                ]
                highneighbours = sorted_full_cyl_array[
                    tree_kdtree.query_ball_point(highest_point[:3], r=max_search_radius)
                ]

                # Same interpolation as before, through the DTM triangulation
                # built once in __init__. This is one point per tree, so the
                # old griddata call spent all its time rebuilding the
                # triangulation and none of it interpolating.
                lowest_point_z = lowest_point[2] - self.dtm_interpolator(lowest_point[0:2])
                assigned = False
                if lowneighbours.shape[0] > 0:
                    angles = MeasureTree.compute_angle(lowest_point[3:6], lowest_point[:3] - lowneighbours[:, :3])
                    valid_angles = angles[angles <= max_search_angle]

                    if valid_angles.shape[0] > 0:
                        best_parent_point = lowneighbours[np.argmin(angles)]
                        tree = np.vstack(
                            (
                                tree,
                                self.interpolate_cyl(lowest_point, best_parent_point, resolution=self.slice_increment),
                            )
                        )
                        tree[:, self.cyl_dict["tree_id"]] = best_parent_point[self.cyl_dict["tree_id"]]
                        sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                        assigned = True
                    else:
                        assigned = False

                elif highneighbours.shape[0] > 0:
                    angles = MeasureTree.compute_angle(highest_point[3:6], highneighbours[:, :3] - highest_point[:3])
                    valid_angles = angles[angles <= max_search_angle]

                    if valid_angles.shape[0] > 0:
                        best_parent_point = highneighbours[np.argmin(angles)]
                        tree = np.vstack(
                            (
                                tree,
                                self.interpolate_cyl(best_parent_point, highest_point, resolution=self.slice_increment),
                            )
                        )
                        tree[:, self.cyl_dict["tree_id"]] = best_parent_point[self.cyl_dict["tree_id"]]
                        sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                        assigned = True
                    else:
                        assigned = False

                if assigned is False and lowest_point_z < self.parameters["tree_base_cutoff_height"]:
                    tree[:, self.cyl_dict["tree_id"]] = t_id
                    sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                    t_id += 1

        save_file(
            self.output_dir + "sorted_full_cyl_array.las",
            sorted_full_cyl_array,
            headers_of_interest=list(self.cyl_dict),
        )

        print("Cylinder interpolation...")

        tree_list = []
        # Appended to from three nesting levels below and never read
        # until the loop is done, so collect the pieces and stack once.
        # Re-stacking the whole array for every interpolated segment of
        # every branch of every tree was quadratic in the cylinder count.
        interpolated_parts = []
        max_tree_id = np.unique(sorted_full_cyl_array[:, self.cyl_dict["tree_id"]]).shape[0]
        for tree_id in np.unique(sorted_full_cyl_array[:, self.cyl_dict["tree_id"]]):
            if int(tree_id) % 10 == 0:
                print("Tree ID", int(tree_id), "/", int(max_tree_id))
            current_tree = sorted_full_cyl_array[sorted_full_cyl_array[:, self.cyl_dict["tree_id"]] == tree_id]
            if current_tree.shape[0] >= self.parameters["min_tree_cyls"]:
                # Copied because get_heights_above_DTM below writes into
                # current_tree in place; the old vstack took its copy here.
                interpolated_parts.append(current_tree.copy())
                _, individual_branches_indices = np.unique(
                    current_tree[:, self.cyl_dict["branch_id"]], return_index=True
                )
                tree_list.append(nx.Graph())
                for branch in current_tree[individual_branches_indices]:
                    branch_id = branch[self.cyl_dict["branch_id"]]
                    parent_branch_id = branch[self.cyl_dict["parent_branch_id"]]
                    tree_list[-1].add_edge(int(parent_branch_id), int(branch_id))
                    current_branch = current_tree[current_tree[:, self.cyl_dict["branch_id"]] == branch_id]
                    parent_branch = current_tree[current_tree[:, self.cyl_dict["branch_id"]] == parent_branch_id]

                    current_branch_copy = deepcopy(current_branch[np.argsort(current_branch[:, 2])])
                    while current_branch_copy.shape[0] > 1:
                        lowest_point = current_branch_copy[0]
                        current_branch_copy = current_branch_copy[1:]
                        # find nearest point. if nearest point > increment size, interpolate.
                        distances = np.abs(np.linalg.norm(current_branch_copy[:, :3] - lowest_point[:3], axis=1))
                        if distances[distances > 0].shape[0] > 0:
                            if np.min(distances[distances > 0]) > self.slice_increment:
                                interp_to_point = current_branch_copy[distances > 0]
                                if interp_to_point.shape[0] > 0:
                                    interp_to_point = interp_to_point[np.argmin(distances[distances > 0])]

                                # Interpolates a single branch.
                                if interp_to_point.shape[0] > 0:
                                    interpolated_cyls = self.interpolate_cyl(
                                        interp_to_point, lowest_point, resolution=self.slice_increment
                                    )
                                    current_branch = np.vstack((current_branch, interpolated_cyls))
                                    interpolated_parts.append(interpolated_cyls)

                    if parent_branch.shape[0] > 0:
                        # Both reductions were missing their axis. np.mean over
                        # the (n, 3) block returns one number averaged across x,
                        # y and z together rather than the branch centroid, and
                        # np.linalg.norm of the result is likewise a single
                        # number rather than one distance per cylinder - so
                        # argmin was always 0 and "the closest point of the
                        # current branch" was just its first row, whatever that
                        # happened to be.
                        parent_centre = np.mean(parent_branch[:, :3], axis=0)
                        closest_point_index = np.argmin(
                            np.linalg.norm(parent_centre - current_branch[:, :3], axis=1)
                        )
                        closest_point_of_current_branch = current_branch[closest_point_index]
                        kdtree = spatial.cKDTree(parent_branch[:, :3])
                        parent_points_in_range = parent_branch[
                            kdtree.query_ball_point(closest_point_of_current_branch[:3], r=max_search_radius)
                        ]
                        lowest_point_of_current_branch = current_branch[np.argmin(current_branch[:, 2])]
                        if parent_points_in_range.shape[0] > 0:
                            angles = MeasureTree.compute_angle(
                                lowest_point_of_current_branch[3:6],
                                lowest_point_of_current_branch[:3] - parent_points_in_range[:, :3],
                            )
                            # Keep the mask rather than overwriting `angles`
                            # with the filtered copy. The filtered array is
                            # shorter, so np.argmin(angles) indexed a different
                            # row of parent_points_in_range than the one that
                            # actually had the smallest angle - silently
                            # interpolating the branch to the wrong parent
                            # cylinder. The equivalent block in "Correcting
                            # Cylinder assignments" above keeps the unfiltered
                            # array and does this correctly.
                            within_search_angle = angles <= max_search_angle

                            if np.any(within_search_angle):
                                candidates = parent_points_in_range[within_search_angle]
                                best_parent_point = candidates[np.argmin(angles[within_search_angle])]
                                # Interpolates from lowest point of current branch to smallest angle parent point.
                                interpolated_parts.append(
                                    self.interpolate_cyl(
                                        lowest_point_of_current_branch,
                                        best_parent_point,
                                        resolution=self.slice_increment,
                                    )
                                )
                current_tree = get_heights_above_DTM(current_tree, self.DTM, self.dtm_interpolator)
                lowest_10_measured_tree_points = deepcopy(current_tree[np.argsort(current_tree[:, -1])][:10])
                lowest_measured_tree_point = np.median(lowest_10_measured_tree_points, axis=0)
                tree_base_point = deepcopy(current_tree[np.argmin(current_tree[:, self.cyl_dict["height_above_dtm"]])])
                tree_base_point[2] = tree_base_point[2] - tree_base_point[self.cyl_dict["height_above_dtm"]]

                interpolated_to_ground = self.interpolate_cyl(
                    lowest_measured_tree_point, tree_base_point, resolution=self.slice_increment
                )
                interpolated_parts.append(interpolated_to_ground)

        interpolated_full_cyl_array = np.vstack([np.zeros((0, 14))] + interpolated_parts)
        del interpolated_parts

        v1 = interpolated_full_cyl_array[:, 3:6]
        v2 = np.vstack(
            (
                interpolated_full_cyl_array[:, 3],
                interpolated_full_cyl_array[:, 4],
                np.zeros((interpolated_full_cyl_array.shape[0])),
            )
        ).T
        interpolated_full_cyl_array[:, self.cyl_dict["segment_angle_to_horiz"]] = self.compute_angle(v1, v2)
        interpolated_full_cyl_array = get_heights_above_DTM(
            interpolated_full_cyl_array, self.DTM, self.dtm_interpolator
        )

        save_file(
            self.output_dir + "interpolated_full_cyl_array.las",
            interpolated_full_cyl_array,
            headers_of_interest=list(self.cyl_dict),
        )
        # interpolated_full_cyl_array, _ = load_file(self.output_dir + 'interpolated_full_cyl_array.las', headers_of_interest=list(self.cyl_dict))
        print(interpolated_full_cyl_array.shape)
        if 0:
            print("Making interp_cyl visualisation...")
            interpolated_cyl_vis = self.make_cyl_visualisation_array(interpolated_full_cyl_array)

            print("\nSaving cylinder visualisation...")
            save_file(
                self.output_dir + "interpolated_cyl_vis.las",
                interpolated_cyl_vis,
                headers_of_interest=self.cyl_vis_headers,
            )

        tree_data = np.zeros((0, 16))
        radial_tree_aware_plot_cropping = False
        plot_centre = [[float(self.plot_summary["Plot Centre X"].iloc[0]), float(self.plot_summary["Plot Centre Y"].iloc[0])]]

        stem_points_sorted = np.zeros((0, len(list(self.stem_dict))))
        veg_points_sorted = np.zeros((0, len(list(self.veg_dict))))

        if self.parameters["plot_radius"] > 0 and self.parameters["plot_radius_buffer"] > 0:
            print("Using tree aware plot cropping mode...")
            radial_tree_aware_plot_cropping = True

        print("Cylinder Outlier Removal...")
        input_data = []
        i = 0
        tree_id_list = np.unique(interpolated_full_cyl_array[:, self.cyl_dict["tree_id"]])
        taper_meas_height_max = self.parameters["taper_measurement_height_max"]
        taper_meas_height_min = self.parameters["taper_measurement_height_min"]
        taper_meas_height_increment = self.parameters["taper_measurement_height_increment"]
        self.taper_measurement_heights = np.arange(
            np.floor(taper_meas_height_min * taper_meas_height_increment) / taper_meas_height_increment,
            (np.floor(taper_meas_height_max * taper_meas_height_increment) / taper_meas_height_increment)
            + taper_meas_height_increment,
            taper_meas_height_increment,
        )
        taper_array = np.zeros((0, self.taper_measurement_heights.shape[0] + 5))

        if tree_id_list.shape[0] > 0:
            max_tree_id = int(np.max(tree_id_list))
            interpolated_rows_by_tree = self._rows_by_value(
                interpolated_full_cyl_array[:, self.cyl_dict["tree_id"]]
            )
            for tree_id in tree_id_list:
                if tree_id % 10 == 0:
                    print("\r", tree_id, "/", max_tree_id, end="")
                i += 1
                single_tree = interpolated_full_cyl_array[interpolated_rows_by_tree[tree_id]]
                if single_tree.shape[0] > 0:
                    # single_tree = self.fix_outliers(single_tree)
                    input_data.append([single_tree, self.parameters["cleaned_measurement_radius"], self.cyl_dict])

            print("\r", max_tree_id, "/", max_tree_id, end="")
            print("\nDone\n")

            print("Starting multithreaded cylinder cleaning/smoothing...")
            cleaned_cyls_list = MeasureTree.pool_map_batched(
                MeasureTree.cylinder_cleaning_multithreaded,
                input_data,
                self.num_cpu_cores,
                label="cyl cleaning",
                costs=[task[0].shape[0] for task in input_data],
            )
            del input_data
            cleaned_cyls = np.vstack(cleaned_cyls_list)
            del cleaned_cyls_list

            save_file(self.output_dir + "cleaned_cyls.las", cleaned_cyls, headers_of_interest=list(self.cyl_dict))
            pd.DataFrame(cleaned_cyls, columns=list(self.cyl_dict)).to_csv(
                self.output_dir + "cleaned_cyls.csv", index=False
            )

            cleaned_cylinders = np.zeros((0, cleaned_cyls.shape[1]))
            # Per-tree pieces, stacked once after the loop (see below).
            tree_data_parts = []
            taper_parts = []
            stem_points_sorted_parts = []
            veg_points_sorted_parts = []
            cleaned_cylinders_parts = []
            text_point_cloud_parts = []

            print("Sorting vegetation...")
            # Simple nearest neighbours vegetation sorting.
            kdtree = spatial.cKDTree(cleaned_cyls[:, :2], leafsize=1000)
            # workers=-1: each query point is answered independently, so
            # spreading millions of them across the cores changes nothing but
            # the time (3.5 s single-threaded on a 5M point plot).
            results = kdtree.query(self.vegetation_points[:, :2], k=1, workers=-1)
            mask = results[0] <= self.parameters["veg_sorting_range"]
            self.vegetation_points = self.vegetation_points[mask]
            results = results[1][mask]
            self.vegetation_points[:, self.veg_dict["tree_id"]] = cleaned_cyls[results, self.cyl_dict["tree_id"]]

            # Stem points are sorted exactly like vegetation: to the nearest
            # cylinder horizontally, within veg_sorting_range. The original
            # FSCT does the same and never reads stem_sorting_range, although
            # it documents that parameter as a 3D limit. Doing what the
            # documentation says was tried and is worse: cylinders exist only
            # where a circle could be fitted, so much of the upper stem lies
            # more than 1 m from any cylinder in 3D while sitting directly
            # above one. On the example plot it left 120,231 of 164,553 stem
            # points assigned, and the rest would vanish from the
            # tree-segmented output cloud. Tree data was identical either way.
            kdtree = spatial.cKDTree(cleaned_cyls[:, :2], leafsize=1000)
            results = kdtree.query(self.stem_points[:, :2], k=1, workers=-1)
            mask = results[0] <= self.parameters["veg_sorting_range"]
            self.stem_points = self.stem_points[mask]
            results = results[1][mask]
            self.stem_points[:, self.stem_dict["tree_id"]] = cleaned_cyls[results, self.cyl_dict["tree_id"]]

            # Row groups per tree id, each built once (see _rows_by_value).
            cyl_rows_by_tree = self._rows_by_value(cleaned_cyls[:, self.cyl_dict["tree_id"]])
            veg_rows_by_tree = self._rows_by_value(self.vegetation_points[:, self.veg_dict["tree_id"]])
            stem_rows_by_tree = self._rows_by_value(self.stem_points[:, self.stem_dict["tree_id"]])
            no_rows = np.zeros(0, dtype=np.intp)

            for tree_id in np.unique(cleaned_cyls[:, self.cyl_dict["tree_id"]]):
                tree = cleaned_cyls[cyl_rows_by_tree[tree_id]]
                tree_vegetation = self.vegetation_points[veg_rows_by_tree.get(tree_id, no_rows)]
                tree_stem_points = self.stem_points[stem_rows_by_tree.get(tree_id, no_rows)]
                combined = np.vstack((tree[:, :3], tree_vegetation[:, :3]))
                combined = np.hstack((combined, np.zeros((combined.shape[0], 1))))
                combined = get_heights_above_DTM(combined, self.DTM, self.dtm_interpolator)

                # Get highest point of tree. Note, there is usually noise, so we use the 98th percentile.
                tree_max_point = combined[
                    abs(
                        combined[:, 2]
                        # `interpolation=` was renamed to `method=` in numpy
                        # 1.22 and removed in 2.0, which made this line raise
                        # TypeError on the pinned numpy and killed the run just
                        # before tree heights were computed.
                        - np.percentile(combined[:, 2], self.parameters["height_percentile"], method="nearest")
                    ).argmin()
                ]

                tree_base_point = deepcopy(combined[np.argmin(combined[:, -1])])
                z_tree_base = tree_base_point[2] - tree_base_point[-1]

                tree_mean_position = np.mean(combined[:, :2], axis=0)
                tree_height = tree_max_point[-1]
                del combined

                if self.parameters["sort_stems"] or self.parameters["generate_output_point_cloud"]:
                    tree_points = tree_stem_points

                DBH_cyls_slice = tree[
                    np.logical_and(
                        tree[:, self.cyl_dict["height_above_dtm"]] >= 1.0,
                        tree[:, self.cyl_dict["height_above_dtm"]] <= 1.6,
                    )
                ]

                # Breast-height slice of *this tree's* stem points. The height
                # filter used to be applied to self.stem_points, the whole
                # plot, so every tree's CCI at breast height was measured
                # against a slice containing every other tree's stem returns
                # too. Any neighbouring stem that happened to fall in the
                # 0.8-1.2 r annulus around this tree's centre was counted as
                # coverage of this tree's circumference, inflating the number.
                # CCI_at_BH is reported in tree_data.csv; nothing downstream
                # filters on it, so this corrects that column alone.
                DBH_points_slice = tree_stem_points[
                    np.logical_and(
                        tree_stem_points[:, self.stem_dict["height_above_dtm"]] >= 1.2,
                        tree_stem_points[:, self.stem_dict["height_above_dtm"]] <= 1.4,
                    )
                ]

                DBH = 0
                CCI_at_BH = 0
                DBH_X = 0
                DBH_Y = 0
                DBH_Z = 0

                if DBH_cyls_slice.shape[0] > 0:
                    DBH = np.around(np.mean(DBH_cyls_slice[:, self.cyl_dict["radius"]]) * 2, 3)
                    DBH_X, DBH_Y = np.mean(DBH_cyls_slice[:, :2], axis=0)
                    DBH_Z = z_tree_base + 1.3
                    CCI_at_BH = self.circumferential_completeness_index(
                        [DBH_X, DBH_Y], DBH / 2, DBH_points_slice[:, :2]
                    )

                volume_1 = tree[0, self.cyl_dict["tree_volume"]]
                volume_2 = (
                    np.pi * ((DBH / 2) ** 2) * (((tree_height - 1.3) / 3) + 1.3)
                )  # Volume of a simple vertical cone from DBH to treetop + volume of cylinder from DBH to ground.

                x_tree_base = tree[np.argmin(tree[:, 2]), 0]
                y_tree_base = tree[np.argmin(tree[:, 2]), 1]
                mean_vegetation_density_in_5m_radius = 0
                mean_understory_height_in_5m_radius = 0
                nearby_understory_points = self.ground_veg[self.ground_veg_kdtree.query_ball_point([DBH_X, DBH_Y], r=5)]

                if nearby_understory_points.shape[0] > 0:
                    mean_understory_height_in_5m_radius = np.around(
                        np.nanmean(nearby_understory_points[:, self.veg_dict["height_above_dtm"]]), 2
                    )
                if tree.shape[0] > 0:
                    description = "Tree " + str(int(tree_id))
                    description = description + "\nDBH: " + str(DBH) + " m"
                    description = description + "\nVolume: " + str(np.around(volume_1, 3)) + " m^3"
                    description = description + "\nVolume 2: " + str(np.around(volume_2, 3)) + " m^3"
                    description = description + "\nHeight: " + str(np.around(tree_height, 3)) + " m"
                    description = (
                        description
                        + "\nMean Veg Density (5 m radius): "
                        + str(mean_vegetation_density_in_5m_radius)
                        + " units"
                    )
                    description = (
                        description
                        + "\nMean Understory Height (5 m radius): "
                        + str(mean_understory_height_in_5m_radius)
                        + " m"
                    )

                    print(description)
                    this_trees_data = np.zeros((1, 16), dtype="object")
                    this_trees_data[:, self.tree_data_dict["PlotId"]] = self.plot_summary["PlotId"]
                    this_trees_data[:, self.tree_data_dict["TreeId"]] = int(tree_id)
                    this_trees_data[:, self.tree_data_dict["x_tree_base"]] = x_tree_base
                    this_trees_data[:, self.tree_data_dict["y_tree_base"]] = y_tree_base
                    this_trees_data[:, self.tree_data_dict["z_tree_base"]] = z_tree_base
                    this_trees_data[:, self.tree_data_dict["DBH"]] = DBH
                    this_trees_data[:, self.tree_data_dict["CCI_at_BH"]] = CCI_at_BH
                    this_trees_data[:, self.tree_data_dict["Height"]] = tree_height
                    this_trees_data[:, self.tree_data_dict["Volume_1"]] = volume_1
                    this_trees_data[:, self.tree_data_dict["Volume_2"]] = volume_2
                    this_trees_data[:, self.tree_data_dict["Crown_mean_x"]] = tree_mean_position[0]
                    this_trees_data[:, self.tree_data_dict["Crown_mean_y"]] = tree_mean_position[1]
                    this_trees_data[:, self.tree_data_dict["Crown_top_x"]] = tree_max_point[0]
                    this_trees_data[:, self.tree_data_dict["Crown_top_y"]] = tree_max_point[1]
                    this_trees_data[:, self.tree_data_dict["Crown_top_z"]] = tree_max_point[2]
                    this_trees_data[
                        :, self.tree_data_dict["mean_understory_height_in_5m_radius"]
                    ] = mean_understory_height_in_5m_radius

                    text_size = 0.00256
                    line_height = 0.025
                    if DBH_X != 0 and DBH_Y != 0 and DBH_Z != 0 and x_tree_base != 0 and y_tree_base != 0:
                        tree_id = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y + 2 * line_height,
                            DBH_Z + 2 * line_height,
                            DBH * 0.5,
                            "         TREE ID: " + str(int(tree_id)),
                        )
                        line0 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y + line_height,
                            DBH_Z + line_height,
                            DBH * 0.5,
                            "            DIAM: " + str(np.around(DBH, 2)) + "m",
                        )
                        line1 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y,
                            DBH_Z,
                            DBH * 0.5,
                            "       CCI AT BH: " + str(np.around(CCI_at_BH, 2)),
                        )
                        line2 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y - 2 * line_height,
                            DBH_Z - 2 * line_height,
                            DBH * 0.5,
                            "          HEIGHT: " + str(np.around(tree_height, 2)) + "m",
                        )
                        line3 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y - 3 * line_height,
                            DBH_Z - 3 * line_height,
                            DBH * 0.5,
                            "          VOLUME 1: " + str(np.around(volume_1, 2)) + "m3",
                        )
                        line4 = self.point_cloud_annotations(
                            text_size,
                            DBH_X,
                            DBH_Y - 4 * line_height,
                            DBH_Z - 4 * line_height,
                            DBH * 0.5,
                            "          VOLUME 2: " + str(np.around(volume_2, 2)) + "m3",
                        )

                        height_measurement_line = self.points_along_line(
                            x_tree_base,
                            y_tree_base,
                            z_tree_base,
                            x_tree_base,
                            y_tree_base,
                            z_tree_base + tree_height,
                            resolution=0.025,
                        )

                        dbh_circle_points = self.create_3d_circles_as_points_flat(
                            DBH_X, DBH_Y, DBH_Z, DBH / 2, circle_points=100
                        )

                        taper = get_taper(
                            tree,
                            self.taper_measurement_heights,
                            z_tree_base,
                            self.parameters["taper_slice_thickness"],
                            self.plot_summary["PlotId"].item(),
                        )

                        # Both branches below did the same work; only the
                        # keep/drop test differed, so it is now written once.
                        #
                        # Every one of these used to be
                        # "arr = np.vstack((arr, new_rows))" per tree, which
                        # reallocates the whole accumulated array and copies it
                        # again for each tree in the plot - quadratic in the
                        # tree count, and stem_points_sorted and
                        # veg_points_sorted accumulate entire point clouds, not
                        # just a row. Collecting the pieces and stacking once
                        # after the loop produces byte-identical arrays.
                        keep_tree = True
                        if radial_tree_aware_plot_cropping:
                            keep_tree = (
                                np.linalg.norm(np.array([x_tree_base, y_tree_base]) - np.array(plot_centre))
                                < self.parameters["plot_radius"]
                            )

                        if keep_tree:
                            tree_data_parts.append(this_trees_data)
                            taper_parts.append(taper)
                            if self.parameters["sort_stems"] or self.parameters["generate_output_point_cloud"]:
                                stem_points_sorted_parts.append(tree_points)
                            veg_points_sorted_parts.append(tree_vegetation)
                            cleaned_cylinders_parts.append(tree)
                            text_point_cloud_parts.extend(
                                (
                                    tree_id,
                                    line0,
                                    line1,
                                    line2,
                                    line3,
                                    # line4 (the "VOLUME 2" line) is built for
                                    # every tree and was stacked in the
                                    # tree-aware-cropping branch but was missing
                                    # from the uncropped one, so in the default
                                    # mode the annotation cloud silently lost
                                    # that line for every tree.
                                    line4,
                                    height_measurement_line,
                                    dbh_circle_points,
                                )
                            )

            tree_data = np.vstack([tree_data] + tree_data_parts)
            taper_array = np.vstack([taper_array] + taper_parts)
            stem_points_sorted = np.vstack([stem_points_sorted] + stem_points_sorted_parts)
            veg_points_sorted = np.vstack([veg_points_sorted] + veg_points_sorted_parts)
            cleaned_cylinders = np.vstack([cleaned_cylinders] + cleaned_cylinders_parts)
            self.text_point_cloud = np.vstack([self.text_point_cloud] + text_point_cloud_parts)

            save_file(self.output_dir + "text_point_cloud.las", self.text_point_cloud)
            if self.parameters["sort_stems"] or self.parameters["generate_output_point_cloud"]:
                if not self.parameters["minimise_output_size_mode"]:
                    save_file(
                        self.output_dir + "stem_points_sorted.las",
                        stem_points_sorted,
                        headers_of_interest=list(self.stem_dict),
                    )

            if not self.parameters["minimise_output_size_mode"]:
                save_file(
                    self.output_dir + "veg_points_sorted.las",
                    veg_points_sorted,
                    headers_of_interest=list(self.veg_dict),
                )

            if 1:
                print("Making cleaned cylinder visualisation...")
                cleaned_cyl_vis = self.make_cyl_visualisation_array(cleaned_cylinders)

                print("\nSaving cylinder visualisation...")
                save_file(
                    self.output_dir + "cleaned_cyl_vis.las", cleaned_cyl_vis, headers_of_interest=self.cyl_vis_headers
                )

        if radial_tree_aware_plot_cropping and self.parameters["generate_output_point_cloud"]:
            self.terrain_points = self.terrain_points[
                np.linalg.norm(self.terrain_points[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]
            ]
            self.cwd_points = self.cwd_points[
                np.linalg.norm(self.cwd_points[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]
            ]
            self.ground_veg = self.ground_veg[
                np.linalg.norm(self.ground_veg[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]
            ]

            self.DTM = self.DTM[np.linalg.norm(self.DTM[:, :2] - plot_centre, axis=1) < self.parameters["plot_radius"]]
            save_file(self.output_dir + "cropped_DTM.las", self.DTM)
            tree_aware_cropped_point_cloud = np.vstack(
                (self.terrain_points, self.cwd_points, self.ground_veg, stem_points_sorted, veg_points_sorted)
            )  # , stem_points_sorted, veg_points_sorted))

            save_file(
                self.output_dir + "tree_aware_cropped_point_cloud.las",
                tree_aware_cropped_point_cloud,
                headers_of_interest=list(self.stem_dict),
            )

        elif self.parameters["generate_output_point_cloud"]:
            tree_aware_cropped_point_cloud = np.vstack(
                (self.terrain_points, self.cwd_points, self.ground_veg, stem_points_sorted, veg_points_sorted)
            )
            save_file(
                self.output_dir + "tree_aware_cropped_point_cloud.las",
                tree_aware_cropped_point_cloud,
                headers_of_interest=list(self.stem_dict),
            )

        taper_data = pd.DataFrame(
            taper_array,
            columns=["PlotId", "TreeId", "x_base", "y_base", "z_base"]
            + [str(i) for i in self.taper_measurement_heights],
        )
        taper_data.to_csv(self.output_dir + "taper_data.csv", index=None, sep=",")

        tree_data = pd.DataFrame(tree_data, columns=list(self.tree_data_dict))
        tree_data.to_csv(self.output_dir + "tree_data.csv", index=None, sep=",")

        plane = Plane.best_fit(self.DTM)
        avg_gradient = self.compute_angle(plane.normal, [0, 0, 1])
        avg_gradient_x = self.compute_angle(plane.normal[[0, 2]], [0, 1])
        avg_gradient_y = self.compute_angle(plane.normal[[1, 2]], [0, 1])
        self.measure_time_end = time.time()
        self.measure_total_time = self.measure_time_end - self.measure_time_start

        self.plot_summary["Measurement Time (s)"] = self.measure_total_time
        self.plot_summary["Total Run Time (s)"] = (
            self.plot_summary["Preprocessing Time (s)"]
            + self.plot_summary["Semantic Segmentation Time (s)"]
            + self.plot_summary["Post processing time (s)"]
            + self.plot_summary["Measurement Time (s)"]
        )

        self.plot_summary["Num Trees in Plot"] = tree_data.shape[0]
        # A plot whose DTM convex hull has zero area gives plot_area == 0;
        # dividing by it raised ZeroDivisionError right at the end of the run.
        self.plot_summary["Stems/ha"] = (
            np.around(tree_data.shape[0] / self.plot_area, 1) if self.plot_area else 0.0
        )

        if tree_data.shape[0] > 0:
            self.plot_summary["Mean DBH"] = np.mean(tree_data["DBH"])
            self.plot_summary["Median DBH"] = np.median(tree_data["DBH"])
            self.plot_summary["Min DBH"] = np.min(tree_data["DBH"])
            self.plot_summary["Max DBH"] = np.max(tree_data["DBH"])

            self.plot_summary["Mean Height"] = np.mean(tree_data["Height"])
            self.plot_summary["Median Height"] = np.median(tree_data["Height"])
            self.plot_summary["Min Height"] = np.min(tree_data["Height"])
            self.plot_summary["Max Height"] = np.max(tree_data["Height"])

            self.plot_summary["Mean Volume 1"] = np.mean(tree_data["Volume_1"])
            self.plot_summary["Median Volume 1"] = np.median(tree_data["Volume_1"])
            self.plot_summary["Min Volume 1"] = np.min(tree_data["Volume_1"])
            self.plot_summary["Max Volume 1"] = np.max(tree_data["Volume_1"])
            self.plot_summary["Total Volume 1"] = np.sum(tree_data["Volume_1"])

            self.plot_summary["Mean Volume 2"] = np.mean(tree_data["Volume_2"])
            self.plot_summary["Median Volume 2"] = np.median(tree_data["Volume_2"])
            self.plot_summary["Min Volume 2"] = np.min(tree_data["Volume_2"])
            self.plot_summary["Max Volume 2"] = np.max(tree_data["Volume_2"])
            self.plot_summary["Total Volume 2"] = np.sum(tree_data["Volume_2"])

        else:
            # Zero the per-tree statistics under their real names. This used to
            # write "Mean Volume", "Median Volume", "Min Volume" and "Max
            # Volume", columns that exist nowhere else, so a plot without trees
            # had four extra columns and misaligned when summaries were
            # combined.
            for statistic in ("Mean", "Median", "Min", "Max"):
                for quantity in ("DBH", "Height", "Volume 1", "Volume 2"):
                    self.plot_summary[f"{statistic} {quantity}"] = 0
            self.plot_summary["Total Volume 1"] = 0
            self.plot_summary["Total Volume 2"] = 0

        # Canopy cover is measured from the vegetation points, not the trees,
        # so it is reported whether or not any trees were found. It used to be
        # set to 0 for a plot without trees, unlike the understorey and CWD
        # fractions below.
        self.plot_summary["Canopy Cover Fraction"] = self.canopy_cover_fraction

        self.plot_summary["Avg Gradient"] = avg_gradient
        self.plot_summary["Avg Gradient X"] = avg_gradient_x
        self.plot_summary["Avg Gradient Y"] = avg_gradient_y

        self.plot_summary["Understory Veg Coverage Fraction"] = self.understory_veg_fraction
        self.plot_summary["CWD Coverage Fraction"] = self.cwd_fraction

        self.plot_summary.to_csv(self.output_dir + "plot_summary.csv", index=False)
        print("Measuring plot took", self.measure_total_time, "s")
        print("Measuring plot done.")
