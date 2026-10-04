import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import fps, knn, radius, global_max_pool

# PointConv was renamed PointNetConv in PyG 2.0. The deprecated alias was kept
# for a while and removed by 2.6. Same constructor (the MLP is local_nn) and
# same forward(x, pos, edge_index) signature, so this is a pure rename.
try:
    from torch_geometric.nn import PointNetConv
except ImportError:  # PyG < 2.0
    from torch_geometric.nn import PointConv as PointNetConv


# The three neighbourhood operations below are written so that a sample's
# result depends only on the sample itself: not on the device, not on the batch
# size, and not on which other samples share its batch. torch_cluster's own
# kernels give none of those guarantees.


def seeded_fps(pos, batch, ratio, u=None):
    """
    Farthest-point sampling from a start chosen by `u`, one value per sample.

    torch_cluster picks the random start itself, and differently per device:
    its CUDA kernel draws from torch's CUDA generator, its CPU kernel from C's
    rand(), which torch.manual_seed never touches. So the same seed gave
    different samples on CPU and GPU; on CPU it had no effect at all, and a
    second run in one process drew different starts (212,008 against 220,012
    stem points on the example plot). The draws also came from one stream
    shared across the batch, so changing the batch size changed every sample.

    Here sample i starts at point floor(u[i] * n_i) of its n_i points. The
    inference dataset derives u from (random_seed, box id), so the start is
    fixed per box; without u (training) it is drawn from torch's generator.
    The start is swapped into the sample's first slot, which is where
    fps(random_start=False) begins on both devices. `batch` must be sorted, as
    PyG collates it.
    """
    counts = torch.bincount(batch)
    firsts = torch.cumsum(counts, 0) - counts
    if u is None:
        u = torch.rand(counts.numel(), dtype=torch.float64)
    offsets = torch.floor(u.to(pos.device, torch.float64) * counts.to(torch.float64)).long()
    starts = firsts + torch.minimum(offsets, counts - 1)
    perm = torch.arange(pos.size(0), device=pos.device)
    perm[firsts], perm[starts] = starts, firsts
    return perm[fps(pos[perm], batch, ratio=ratio, random_start=False)]


def first_neighbours(x, y, r, batch_x, batch_y, max_num_neighbors):
    """
    For each point of y, its first `max_num_neighbors` points of x within r,
    in index order: the neighbourhoods the network was trained on.

    torch_cluster's CUDA kernel scans x in index order and stops at the
    limit. Its CPU kernel walks a k-d tree and stops at the limit in tree
    order, so wherever a point has more than 64 neighbours - most of a dense
    stem - the CPU fed the network a different neighbourhood from the one it
    was trained on. That is why CPU segmentation was so much worse (2 trees
    instead of 4 on the example plot). On CPU, take every neighbour and keep
    the first 64 by index, which is exactly the CUDA selection.
    """
    if x.is_cuda:
        row, col = radius(x, y, r, batch_x, batch_y, max_num_neighbors=max_num_neighbors)
        return row, col
    row, col = radius(x, y, r, batch_x, batch_y, max_num_neighbors=x.size(0))
    order = torch.argsort(row * x.size(0) + col)
    row, col = row[order], col[order]
    counts = torch.bincount(row, minlength=y.size(0))
    firsts = torch.cumsum(counts, 0) - counts
    keep = torch.arange(row.numel()) - firsts[row] < max_num_neighbors
    return row[keep], col[keep]


def ordered_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k):
    """
    PyG's knn_interpolate, summed in a fixed order.

    PyG sums the k weighted neighbours with a scatter, which on the GPU adds
    in whatever order the atomics land, so a rerun can differ in the last bit.
    Here each target's neighbours are put in (distance, index) order and added
    one at a time, the same way on every device and every run.
    """
    with torch.no_grad():
        y_idx, x_idx = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y)
    if y_idx.numel() != pos_y.size(0) * k:  # a sample with fewer than k points
        return knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=k)
    with torch.no_grad():
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1)
        # Three stable sorts, least significant key first: index, distance, target.
        order = torch.argsort(x_idx, stable=True)
        order = order[torch.argsort(squared_distance[order], stable=True)]
        order = order[torch.argsort(y_idx[order], stable=True)]
        x_idx = x_idx[order].view(-1, k)
        weights = (1.0 / torch.clamp(squared_distance[order], min=1e-16)).view(-1, k, 1)
    numerator = x[x_idx[:, 0]] * weights[:, 0]
    denominator = weights[:, 0]
    for j in range(1, k):
        numerator = numerator + x[x_idx[:, j]] * weights[:, j]
        denominator = denominator + weights[:, j]
    return numerator / denominator


def per_sample_self_loops(src, dst, batch_src, batch_dst):
    """
    PointNetConv's self-loop handling, applied one sample at a time.

    With add_self_loops (its default, and what the model was trained with),
    PointNetConv drops every edge whose source index equals its destination
    index and then links source point i to centroid i, for each i. Source and
    centroids are different point sets here, so those are arbitrary pairings,
    and PyG makes them by global index across the whole batch: centroid i of the
    second box was linked to a point of the first. A box's features leaked into
    its batch-mate's, and which leak depended on the batch size - batch 4
    against batch 2 moved 21,593 labels on the example plot.

    This makes the same pairings by index *within* each sample, which is what
    PyG does for a box processed on its own. The result is batch-size 1
    behaviour at any batch size.
    """
    n_src = torch.bincount(batch_src)
    n_dst = torch.bincount(batch_dst, minlength=n_src.numel())
    first_src = torch.cumsum(n_src, 0) - n_src
    first_dst = torch.cumsum(n_dst, 0) - n_dst
    sample = batch_dst[dst]
    keep = (src - first_src[sample]) != (dst - first_dst[sample])
    src, dst = src[keep], dst[keep]
    loop_dst = torch.arange(batch_dst.numel(), device=dst.device)
    local = loop_dst - first_dst[batch_dst]
    within = local < n_src[batch_dst]  # PyG pairs only the first min(n_src, n_dst)
    loop_dst = loop_dst[within]
    loop_src = first_src[batch_dst[within]] + local[within]
    return torch.stack([torch.cat([src, loop_src]), torch.cat([dst, loop_dst])], dim=0)


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, NN):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        # Self-loops are added per sample by per_sample_self_loops instead.
        self.conv = PointNetConv(NN, add_self_loops=False)

    def forward(self, x, pos, batch, u=None):
        idx = seeded_fps(pos, batch, ratio=self.ratio, u=u)
        row, col = first_neighbours(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = per_sample_self_loops(col, row, batch, batch[idx])
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, NN):
        super(GlobalSAModule, self).__init__()
        self.NN = NN

    def forward(self, x, pos, batch):
        x = self.NN(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i])) for i in range(1, len(channels))])


class FPModule(torch.nn.Module):
    def __init__(self, k, NN):
        super(FPModule, self).__init__()
        self.k = k
        self.NN = NN

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = ordered_knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.NN(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.sa1_module = SAModule(0.1, 0.2, MLP([3, 128, 256, 512]))
        self.sa2_module = SAModule(0.05, 0.4, MLP([512 + 3, 512, 1024, 1024]))
        self.sa3_module = GlobalSAModule(MLP([1024 + 3, 1024, 2048, 2048]))

        self.fp3_module = FPModule(1, MLP([3072, 1024, 1024]))
        self.fp2_module = FPModule(3, MLP([1536, 1024, 1024]))
        self.fp1_module = FPModule(3, MLP([1024, 1024, 1024]))

        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, num_classes, 1)
        self.drop1 = torch.nn.Dropout(0.5)
        self.bn1 = torch.nn.BatchNorm1d(1024)

    def forward(self, data):
        # One sampling start per sample and layer, shape (batch, 2), set by the
        # inference dataset from (random_seed, box id). Training leaves it out
        # and the starts are drawn at random.
        u = getattr(data, "fps_u", None)
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out, u=None if u is None else u[:, 0])
        sa2_out = self.sa2_module(*sa1_out, u=None if u is None else u[:, 1])
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = x.unsqueeze(dim=0)
        x = x.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x
