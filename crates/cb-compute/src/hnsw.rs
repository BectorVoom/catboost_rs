//! Online-HNSW approximate nearest-neighbor index — a **bit-for-bit** Rust port of
//! upstream CatBoost 1.2.10's `library/cpp/online_hnsw/base/` (plus the `NHnsw`
//! routines it calls), the incremental graph the KNN embedding calcer
//! (`catboost/private/libs/embedding_features/knn.{h,cpp}`) uses as its neighbor
//! source (FEAT-07 / Phase 16).
//!
//! # Why this exists (the FEAT-07 blocker)
//!
//! Upstream's `TKNNCalcer` stores inserted embeddings in an ONLINE HNSW index
//! (`TOnlineHnswDenseVectorIndex<float, TL2SqrDistance<float>>`) and votes over the
//! `CloseNum` **approximate** nearest neighbors. The prior Rust calcer used a
//! brute-force-EXACT scan ([`crate::KnnCloud`]); it matches upstream only where the
//! graph degenerates to exact (small / well-separated prefixes). On the XOR corpus
//! the approximate graph returns a genuinely different neighbor set (upstream's
//! `{1,3,4}` vs the exact `{0,2,4}` for the documented cloud-B doc6 case), so the
//! estimated columns — and every downstream stage — diverge. Closing the ≤1e-5 gate
//! requires reproducing upstream's HNSW neighbor SET index-for-index, which an
//! off-the-shelf ANN crate can never do (different graph construction / RNG / heap
//! tie-break). This module transcribes the upstream C++ verbatim.
//!
//! # Parity crux — construction order + heap tie-break (D-01)
//!
//! Upstream has NO RNG in the online-HNSW path (level growth is deterministic by
//! fullness via `AddNewLevelIfLastIsFull`, not a random level draw). Bit-for-bit
//! fidelity therefore rests on three transcribed invariants:
//!   1. **Insertion order** — the graph is built by `GetNearestNeighborsAndAddItem`
//!      in the exact order the online seam feeds it (the learn permutation).
//!   2. **`std::priority_queue` heap order** — ties in distance are broken by the
//!      libc++ binary-heap internal order, NOT by a stable secondary key. The heap
//!      operations ([`Heap`]) are a line-for-line port of libc++'s
//!      `push_heap`/`pop_heap`/`make_heap` (`__sift_up` / `__floyd_sift_down` /
//!      `__sift_down`) so identical-distance neighbors resolve exactly as upstream.
//!   3. **f32 L2-squared accumulation** — [`l2_sqr_f32`] reproduces upstream's
//!      SSE lane layout (`library/cpp/l2_distance/l2_distance.cpp:148-168`): four
//!      lane accumulators reduced `((l0+l1)+l2)+l3`, scalar tail folded into lane 0.
//!
//! # No `cubecl` / backend symbols (MODEL-02 discipline)
//!
//! Pure-CPU compute; imports no backend/cubecl symbol, exactly like the sibling
//! embedding calcers.

use std::collections::HashSet;
use std::collections::VecDeque;

use cb_core::{CbError, CbResult};

/// One graph edge / search result: a neighbor id and its distance to the query
/// (`NHnsw::TDistanceTraits::TNeighbor`, `build_routines.h:71-76`). `dist` is the
/// `TL2SqrDistance<float>` result (`f32`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    /// Distance to the query (`TL2SqrDistance<float>`, squared L2, `f32`).
    pub dist: f32,
    /// Item id (insertion order id in the item storage).
    pub id: usize,
}

/// A harmless placeholder used only as the `unwrap_or` fallback for checked slice
/// access on indices that are provably in-bounds (the workspace denies raw
/// indexing). It never influences a result: every read site addresses a valid heap
/// / graph slot, so the fallback branch is dead.
const NIL: Neighbor = Neighbor { dist: 0.0, id: 0 };

/// Squared-L2 distance between two equal-length `f32` vectors, reproducing
/// upstream's SSE accumulation order (`L2SqrDistance(const float*, const float*,
/// int)`, `l2_distance.cpp:148-168`).
///
/// The SSE kernel keeps four lane accumulators — `lane[c] += (a-b)²` for each
/// stride-4 block — then reduces `((lane0 + lane1) + lane2) + lane3`; a scalar tail
/// (`length % 4`) is folded into `lane0` BEFORE the reduction. All arithmetic is
/// single-precision, matching the graph's comparator exactly. Mismatched lengths
/// (never produced in-tree) fold only the shared prefix.
#[must_use]
pub fn l2_sqr_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut lane = [0.0_f32; 4];
    let mut i = 0;
    while i + 4 <= len {
        for l in 0..4 {
            let av = *a.get(i + l).unwrap_or(&0.0);
            let bv = *b.get(i + l).unwrap_or(&0.0);
            let d = av - bv;
            if let Some(slot) = lane.get_mut(l) {
                *slot += d * d;
            }
        }
        i += 4;
    }
    // Scalar tail folded into lane 0 (upstream `res[0] += Sqr(...)`).
    while i < len {
        let av = *a.get(i).unwrap_or(&0.0);
        let bv = *b.get(i).unwrap_or(&0.0);
        let d = av - bv;
        if let Some(slot) = lane.get_mut(0) {
            *slot += d * d;
        }
        i += 1;
    }
    let l0 = *lane.first().unwrap_or(&0.0);
    let l1 = *lane.get(1).unwrap_or(&0.0);
    let l2 = *lane.get(2).unwrap_or(&0.0);
    let l3 = *lane.get(3).unwrap_or(&0.0);
    ((l0 + l1) + l2) + l3
}

// ===========================================================================
// libc++ binary heap (std::priority_queue) — bit-for-bit port.
// ===========================================================================
//
// `std::priority_queue<TNeighbor, TVector, Compare>` push/pop are libc++
// `std::push_heap` / `std::pop_heap`; the RANGE constructor is `std::make_heap`.
// Distance ties are resolved purely by these operations' element movement, so an
// exact transcription is required for neighbor-set parity. `Compare` here only ever
// inspects `.dist` (`TNeighborLess` / `TNeighborGreater`), captured by
// [`Heap::greater`]: a max-queue uses `less` (`a<b`), a min-queue uses `greater`
// (`b<a`).

/// A `std::priority_queue<TNeighbor, …>` analog over [`Neighbor`], comparing on
/// `.dist` only. `greater == false` → max-queue (`top` = largest dist, the
/// `TNeighborMaxQueue`); `greater == true` → min-queue (`top` = smallest dist, the
/// `TNeighborMinQueue`).
struct Heap {
    data: Vec<Neighbor>,
    greater: bool,
}

impl Heap {
    fn new(greater: bool) -> Self {
        Heap {
            data: Vec::new(),
            greater,
        }
    }

    /// The heap comparator (`Compare`): inspects `.dist` only.
    #[inline]
    fn comp(&self, a: &Neighbor, b: &Neighbor) -> bool {
        if self.greater {
            b.dist < a.dist
        } else {
            a.dist < b.dist
        }
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `top()` — the container front (`c.front()`).
    fn top(&self) -> Neighbor {
        *self.data.first().unwrap_or(&NIL)
    }

    /// `push(x)` — `c.push_back(x); std::push_heap(...)`.
    fn push(&mut self, x: Neighbor) {
        self.data.push(x);
        let len = self.data.len();
        self.sift_up(len);
    }

    /// `pop()` — `std::pop_heap(...); c.pop_back()`.
    fn pop(&mut self) {
        let len = self.data.len();
        self.pop_heap(len);
        self.data.pop();
    }

    #[inline]
    fn get(&self, i: usize) -> Neighbor {
        *self.data.get(i).unwrap_or(&NIL)
    }

    #[inline]
    fn set(&mut self, i: usize, v: Neighbor) {
        if let Some(slot) = self.data.get_mut(i) {
            *slot = v;
        }
    }

    /// libc++ `__sift_up(first, first+len, comp, len)` — sift the element at
    /// `len - 1` toward the root (`push_heap.h`).
    fn sift_up(&mut self, len: usize) {
        if len <= 1 {
            return;
        }
        let mut parent = (len - 2) / 2;
        let mut last = len - 1;
        if !self.comp(&self.get(parent), &self.get(last)) {
            return;
        }
        let t = self.get(last);
        loop {
            let pv = self.get(parent);
            self.set(last, pv);
            last = parent;
            if parent == 0 {
                break;
            }
            parent = (parent - 1) / 2;
            if !self.comp(&self.get(parent), &t) {
                break;
            }
        }
        self.set(last, t);
    }

    /// libc++ `__floyd_sift_down(first, comp, len)` — sift the root hole down to a
    /// leaf, returning the final hole index (`sift_down.h`).
    fn floyd_sift_down(&mut self, len: usize) -> usize {
        let mut hole = 0usize;
        let mut child_i = 0usize;
        let mut child = 0usize;
        loop {
            child_i += child + 1;
            child = 2 * child + 1;
            if child + 1 < len && self.comp(&self.get(child_i), &self.get(child_i + 1)) {
                child_i += 1;
                child += 1;
            }
            let cv = self.get(child_i);
            self.set(hole, cv);
            hole = child_i;
            if child > (len - 2) / 2 {
                return hole;
            }
        }
    }

    /// libc++ `__pop_heap(first, first+len, comp, len)` — move the max/min to the
    /// back, restore the heap over `[0, len-1)` (`pop_heap.h`).
    fn pop_heap(&mut self, len: usize) {
        if len <= 1 {
            return;
        }
        let top = self.get(0);
        let hole = self.floyd_sift_down(len);
        let last = len - 1;
        if hole == last {
            self.set(hole, top);
        } else {
            let lastv = self.get(last);
            self.set(hole, lastv);
            let new_len = hole + 1;
            self.set(last, top);
            self.sift_up(new_len);
        }
    }

}

// ===========================================================================
// TDynamicDenseGraph — one HNSW level (dynamic_dense_graph.h).
// ===========================================================================

/// One HNSW level's dense adjacency (`NOnlineHnsw::TDynamicDenseGraph`,
/// `dynamic_dense_graph.h`). Edges are stored row-major padded to
/// `max_neighbor_count` per vertex; `neighbor_count` is the currently-live width
/// (grows as `size - 1` until it saturates at `max_neighbor_count`).
#[derive(Debug, Clone)]
struct DynamicDenseGraph {
    max_neighbor_count: usize,
    size: usize,
    neighbor_count: usize,
    distances: Vec<f32>,
    ids: Vec<usize>,
}

impl DynamicDenseGraph {
    fn new(max_neighbor_count: usize, max_size: usize) -> Self {
        DynamicDenseGraph {
            max_neighbor_count,
            size: 0,
            neighbor_count: 0,
            distances: Vec::with_capacity(max_size.saturating_mul(max_neighbor_count)),
            ids: Vec::with_capacity(max_size.saturating_mul(max_neighbor_count)),
        }
    }

    /// Copy-construct a larger level from `other` (`TDynamicDenseGraph(mnc, ms,
    /// other)`, `dynamic_dense_graph.h:30-50`). When the neighbor width is
    /// unchanged the dense buffers are copied verbatim; otherwise each vertex's
    /// live edges are re-padded to the new width.
    fn from_other(max_neighbor_count: usize, max_size: usize, other: &DynamicDenseGraph) -> Self {
        let mut g = DynamicDenseGraph::new(max_neighbor_count, max_size);
        g.size = other.size;
        g.neighbor_count = other.neighbor_count;
        if max_neighbor_count == other.max_neighbor_count {
            g.distances.extend_from_slice(&other.distances);
            g.ids.extend_from_slice(&other.ids);
            return g;
        }
        // `max_neighbor_count >= other.max_neighbor_count >= other.neighbor_count`
        // (upstream `Y_ASSERT(other.MaxNeighborCount <= MaxNeighborCount)`), so the
        // pad width is non-negative; `saturating_sub` matches the rest of the file
        // and cannot underflow-panic if that invariant is ever violated.
        let pad = max_neighbor_count.saturating_sub(g.neighbor_count);
        for vertex_id in 0..other.size {
            let d = other.neighbor_distances(vertex_id);
            let i = other.neighbor_ids(vertex_id);
            // ExtendWithPadding: append `neighbor_count` live entries, pad to mnc.
            g.distances.extend_from_slice(&d);
            g.distances.resize(g.distances.len() + pad, 0.0);
            g.ids.extend_from_slice(&i);
            g.ids.resize(g.ids.len() + pad, 0);
        }
        g
    }

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_neighbor_count(&self) -> usize {
        self.neighbor_count
    }

    /// The live neighbor ids of `index` as a BORROWED slice (length
    /// `neighbor_count`) — the zero-alloc form used on the read-only search hot
    /// path. The mutation paths that read-then-mutate the graph use the owned
    /// [`Self::neighbor_ids`] instead to sidestep the borrow conflict.
    fn neighbor_ids_slice(&self, index: usize) -> &[usize] {
        let start = index.saturating_mul(self.max_neighbor_count);
        let end = start + self.neighbor_count;
        self.ids.get(start..end).unwrap_or(&[])
    }

    /// The live neighbor ids of `index` (length `neighbor_count`), OWNED — for
    /// callers that must retain the ids across a subsequent graph mutation.
    fn neighbor_ids(&self, index: usize) -> Vec<usize> {
        self.neighbor_ids_slice(index).to_vec()
    }

    /// The live neighbor distances of `index` (length `neighbor_count`).
    fn neighbor_distances(&self, index: usize) -> Vec<f32> {
        let start = index.saturating_mul(self.max_neighbor_count);
        let end = start + self.neighbor_count;
        self.distances.get(start..end).map(<[f32]>::to_vec).unwrap_or_default()
    }

    /// Append a new vertex with `neighbors` edges, padding to `max_neighbor_count`
    /// (`Append`, `dynamic_dense_graph.h:92-109`). Bumps `neighbor_count` to
    /// `size - 1` until it saturates at `max_neighbor_count`.
    fn append(&mut self, neighbors: &[Neighbor]) {
        for n in neighbors {
            self.distances.push(n.dist);
            self.ids.push(n.id);
        }
        let pad = self.max_neighbor_count.saturating_sub(neighbors.len());
        self.distances.resize(self.distances.len() + pad, 0.0);
        self.ids.resize(self.ids.len() + pad, 0);
        self.size += 1;
        if self.neighbor_count < self.max_neighbor_count {
            self.neighbor_count = self.size - 1;
        }
    }

    /// Overwrite `index`'s first `neighbors.len()` edge slots in place
    /// (`ReplaceNeighbors`, `dynamic_dense_graph.h:111-119`). Does NOT change
    /// `neighbor_count`.
    fn replace_neighbors(&mut self, index: usize, neighbors: &[Neighbor]) {
        let offset = index.saturating_mul(self.max_neighbor_count);
        for (pos, n) in neighbors.iter().enumerate() {
            if let Some(slot) = self.distances.get_mut(offset + pos) {
                *slot = n.dist;
            }
            if let Some(slot) = self.ids.get_mut(offset + pos) {
                *slot = n.id;
            }
        }
    }
}

/// `NHnsw::GetLevelSizes` (`build_routines.cpp`): the per-level vertex counts for a
/// static build. Not used by the online insertion path (which grows levels by
/// fullness), but reproduced for the `NumVertices`-known construction branch and
/// offline parity.
#[must_use]
fn get_level_sizes(num_vectors: usize, level_size_decay: usize) -> Vec<usize> {
    let mut level_sizes = Vec::new();
    if num_vectors == 1 {
        level_sizes.push(num_vectors);
    } else {
        let mut n = num_vectors;
        while n > 1 {
            level_sizes.push(n);
            if level_size_decay == 0 {
                break;
            }
            n /= level_size_decay;
        }
    }
    level_sizes
}

// ===========================================================================
// TOnlineHnswBuildOptions (build_options.h).
// ===========================================================================

/// Online-HNSW build options (`NOnlineHnsw::TOnlineHnswBuildOptions`,
/// `build_options.h`). The KNN calcer constructs `{MaxNeighbors=CloseNum,
/// SearchNeighborhoodSize=300}` with `LevelSizeDecay`/`NumVertices` AUTO.
#[derive(Debug, Clone)]
struct BuildOptions {
    max_neighbors: usize,
    search_neighborhood_size: usize,
    level_size_decay: usize,
    num_vertices: usize,
}

const AUTO_SELECT: usize = 0;

// ===========================================================================
// TOnlineHnswIndexBase + TDenseVectorExtendableItemStorage (index_base.h /
// item_storage.h) — combined, mirroring TOnlineHnswDenseVectorIndex.
// ===========================================================================

/// The online HNSW index over dense `f32` vectors — a port of
/// `TOnlineHnswDenseVectorIndex<float, TL2SqrDistance<float>>`, which fuses
/// `TOnlineHnswIndexBase` (the graph) with `TDenseVectorExtendableItemStorage` (the
/// point storage). Items are enumerated 0..n in insertion order; that id is the
/// neighbor id the KNN vote indexes.
#[derive(Debug, Clone)]
pub struct OnlineHnswIndex {
    dimension: usize,
    /// Flat row-major point storage (`TDenseVectorExtendableItemStorage::Data`).
    points: Vec<f32>,
    /// Number of inserted items (`GetNumItems`).
    size: usize,
    opts: BuildOptions,
    /// `Levels`: front (`levels[0]`) is the base level holding every item; higher
    /// indices are the coarser frozen upper levels used for greedy descent.
    levels: VecDeque<DynamicDenseGraph>,
    /// `LevelSizes`: front = largest capacity, back = smallest (`LevelSizeDecay`).
    level_sizes: VecDeque<usize>,
    /// Per-vertex count of "diverse" neighbors at the base level
    /// (`DiverseNeighborsNums`).
    diverse_neighbors_nums: Vec<usize>,
}

impl OnlineHnswIndex {
    /// Construct an empty index over `dimension`-dim vectors with the KNN calcer's
    /// options (`TOnlineHnswBuildOptions({close_num, 300})`,
    /// `TOnlineHnswIndexBase(opts)` constructor, `index_base.h:30-50`).
    ///
    /// `LevelSizeDecay` and `NumVertices` are AUTO: `LevelSizeDecay = max(2,
    /// close_num / 2)`, `NumVertices = 0` → `LevelSizes = {LevelSizeDecay}` and the
    /// first (empty) base level is created immediately.
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] if the options are invalid (`1 <= max_neighbors <=
    /// search_neighborhood_size`).
    pub fn new(dimension: usize, close_num: usize, search_neighborhood_size: usize) -> CbResult<Self> {
        let mut opts = BuildOptions {
            max_neighbors: close_num,
            search_neighborhood_size,
            level_size_decay: AUTO_SELECT,
            num_vertices: AUTO_SELECT,
        };
        if opts.max_neighbors == 0 || opts.max_neighbors > opts.search_neighborhood_size {
            return Err(CbError::OutOfRange(format!(
                "OnlineHnswIndex::new: require 1 <= max_neighbors ({}) <= search_neighborhood_size ({})",
                opts.max_neighbors, opts.search_neighborhood_size
            )));
        }
        // LevelSizeDecay AUTO → max(2, MaxNeighbors / 2).
        if opts.level_size_decay == AUTO_SELECT {
            opts.level_size_decay = (opts.max_neighbors / 2).max(2);
        }

        let mut level_sizes: VecDeque<usize> = VecDeque::new();
        if opts.num_vertices == 0 || opts.num_vertices == AUTO_SELECT {
            level_sizes.push_back(opts.level_size_decay);
        } else {
            for s in get_level_sizes(opts.num_vertices, opts.level_size_decay) {
                level_sizes.push_back(s);
            }
        }

        let mut levels: VecDeque<DynamicDenseGraph> = VecDeque::new();
        let back = *level_sizes.back().unwrap_or(&opts.level_size_decay);
        levels.push_front(DynamicDenseGraph::new(
            back.saturating_sub(1).min(opts.max_neighbors),
            back,
        ));

        Ok(OnlineHnswIndex {
            dimension,
            points: Vec::new(),
            size: 0,
            opts,
            levels,
            level_sizes,
            diverse_neighbors_nums: Vec::new(),
        })
    }

    /// Number of inserted items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether no items have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Stored item `id` as a slice (`GetItem`).
    fn item(&self, id: usize) -> &[f32] {
        let start = id.saturating_mul(self.dimension);
        self.points
            .get(start..start + self.dimension)
            .unwrap_or(&[])
    }

    /// Distance from an external `query` to stored item `id`.
    fn dist_to(&self, query: &[f32], id: usize) -> f32 {
        l2_sqr_f32(query, self.item(id))
    }

    /// Distance from `query` to stored item `id`, reading `points`/`dimension` as
    /// FIELDS (not via `&self` methods) so the search hot path can compute it while
    /// holding an immutable borrow of the disjoint `levels` field — enabling the
    /// zero-alloc [`DynamicDenseGraph::neighbor_ids_slice`] on the traversal loops.
    fn dist_to_fielded(points: &[f32], dim: usize, query: &[f32], id: usize) -> f32 {
        let start = id.saturating_mul(dim);
        l2_sqr_f32(query, points.get(start..start + dim).unwrap_or(&[]))
    }

    /// `GetNearestNeighbors(item, topSize)` — the read path (`index_base.h:178-187`).
    /// Naive-exact when `max_neighbors + 1 >= num_items`, else the approximate HNSW
    /// search. Returns neighbors ASCENDING by distance (nearest first).
    #[must_use]
    pub fn get_nearest_neighbors(&self, query: &[f32], top_size: usize) -> Vec<Neighbor> {
        if self.opts.max_neighbors + 1 >= self.size {
            return self.get_nearest_neighbors_naive(query, top_size);
        }
        let mut result = self.find_approximate_neighbors(query, top_size);
        result.reverse();
        result
    }

    /// `GetNearestNeighborsNaive` (`index_base.h:211-233`): exact top-`topSize` via
    /// a bounded max-queue. Result is ASCENDING (nearest first).
    fn get_nearest_neighbors_naive(&self, query: &[f32], top_size: usize) -> Vec<Neighbor> {
        let mut nearest = Heap::new(false); // TNeighborMaxQueue
        for neighbor_id in 0..self.size {
            let d = self.dist_to(query, neighbor_id);
            let is_full = nearest.size() == top_size;
            if !is_full || d < nearest.top().dist {
                nearest.push(Neighbor {
                    dist: d,
                    id: neighbor_id,
                });
                if is_full {
                    nearest.pop();
                }
            }
        }
        let sz = nearest.size();
        let mut result = vec![NIL; sz];
        let mut pos = sz;
        while pos > 0 {
            pos -= 1;
            if let Some(slot) = result.get_mut(pos) {
                *slot = nearest.top();
            }
            nearest.pop();
        }
        result
    }

    /// `NHnsw::NRoutines::FindApproximateNeighbors` (`build_routines.h:212-282`):
    /// greedy descent through the coarse levels to an entry point, then an
    /// ef-bounded search at the base level. Returns neighbors DESCENDING by distance
    /// (the raw max-queue drain order; the caller reverses).
    fn find_approximate_neighbors(&self, query: &[f32], top_size: usize) -> Vec<Neighbor> {
        let num_levels = self.levels.len();
        let dim = self.dimension;
        let mut entry_id = 0usize;
        let mut entry_dist = self.dist_to(query, entry_id);

        // Greedy descent: for level = size; level-- > 1;
        // `points` is borrowed as a field alongside the `levels` slice (disjoint),
        // so `neighbor_ids_slice` stays zero-alloc on this hot loop.
        let points = &self.points;
        let mut level = num_levels;
        while level > 1 {
            level -= 1;
            loop {
                let mut entry_changed = false;
                if let Some(graph) = self.levels.get(level) {
                    for &neighbor_id in graph.neighbor_ids_slice(entry_id) {
                        let d = Self::dist_to_fielded(points, dim, query, neighbor_id);
                        if d < entry_dist {
                            entry_dist = d;
                            entry_id = neighbor_id;
                            entry_changed = true;
                        }
                    }
                }
                if !entry_changed {
                    break;
                }
            }
        }

        let ef = self.opts.search_neighborhood_size;
        let mut nearest = Heap::new(false); // max-queue
        let mut candidates = Heap::new(true); // min-queue
        let mut visited: HashSet<usize> = HashSet::new();
        nearest.push(Neighbor {
            dist: entry_dist,
            id: entry_id,
        });
        candidates.push(Neighbor {
            dist: entry_dist,
            id: entry_id,
        });
        visited.insert(entry_id);

        while !candidates.is_empty() {
            let cur = candidates.top();
            candidates.pop();
            if nearest.top().dist < cur.dist {
                break;
            }
            // Borrow the base level's neighbor slice for `cur.id` (zero-alloc) while
            // reading `points` as the disjoint field — no per-candidate Vec.
            if let Some(base) = self.levels.front() {
                for &neighbor_id in base.neighbor_ids_slice(cur.id) {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    let d = Self::dist_to_fielded(points, dim, query, neighbor_id);
                    if nearest.size() < ef || d < nearest.top().dist {
                        nearest.push(Neighbor {
                            dist: d,
                            id: neighbor_id,
                        });
                        candidates.push(Neighbor {
                            dist: d,
                            id: neighbor_id,
                        });
                        visited.insert(neighbor_id);
                        if nearest.size() > ef {
                            nearest.pop();
                        }
                    }
                }
            }
        }

        while nearest.size() > top_size {
            nearest.pop();
        }
        let mut result = Vec::with_capacity(nearest.size());
        while !nearest.is_empty() {
            result.push(nearest.top());
            nearest.pop();
        }
        result
    }

    /// `GetNearestNeighborsAndAddItem(item)` — the insert path
    /// (`index_base.h:198-207`): search the current graph, append the item to
    /// storage, grow a level if the base is full, then wire the new vertex's edges.
    /// Returns the pre-insertion nearest ids (ASCENDING), matching upstream's return
    /// (the KNN calcer discards it and re-reads via [`Self::get_nearest_neighbors`]).
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] if `item.len()` != the index dimension.
    pub fn get_nearest_neighbors_and_add_item(&mut self, item: &[f32]) -> CbResult<Vec<usize>> {
        if item.len() != self.dimension {
            return Err(CbError::OutOfRange(format!(
                "OnlineHnswIndex: item dim {} != index dim {}",
                item.len(),
                self.dimension
            )));
        }
        let nearest = self.get_nearest_neighbors(item, usize::MAX);

        // itemStorage.AddItem(item)
        self.points.extend_from_slice(item);
        self.size += 1;

        self.add_new_level_if_last_is_full();
        self.extend_last_level(&nearest);

        Ok(nearest.iter().map(|n| n.id).collect())
    }

    /// `AddNewLevelIfLastIsFull` (`index_base.h:235-244`): when the base level fills
    /// to its cap, push a new larger base (copying the old base) and register the
    /// next `LevelSizes` step.
    fn add_new_level_if_last_is_full(&mut self) {
        let ls_len = self.level_sizes.len();
        let lv_len = self.levels.len();
        // lastLevelMaxSize = *(LevelSizes.rbegin() + (Levels.size() - 1))
        let last_level_max_size = *self
            .level_sizes
            .get(ls_len.wrapping_sub(1).wrapping_sub(lv_len - 1))
            .unwrap_or(&0);
        let front_size = self.levels.front().map_or(0, DynamicDenseGraph::get_size);
        if front_size == last_level_max_size {
            if self.level_sizes.len() == self.levels.len() {
                let front = *self.level_sizes.front().unwrap_or(&0);
                self.level_sizes
                    .push_front(front.saturating_mul(self.opts.level_size_decay));
            }
            // newLevelSize = *(LevelSizes.rbegin() + Levels.size())
            let ls_len2 = self.level_sizes.len();
            let new_level_size = *self
                .level_sizes
                .get(ls_len2.wrapping_sub(1).wrapping_sub(self.levels.len()))
                .unwrap_or(&0);
            let cap = self.opts.max_neighbors.min(new_level_size.saturating_sub(1));
            let new_level = self
                .levels
                .front()
                .map(|old| DynamicDenseGraph::from_other(cap, new_level_size, old))
                .unwrap_or_else(|| DynamicDenseGraph::new(cap, new_level_size));
            self.levels.push_front(new_level);
        }
    }

    /// `ExtendLastLevel` (`index_base.h:246-260`): trim the found neighbors to the
    /// new vertex's diverse edge set, add the reverse edges into existing vertices,
    /// then append the new vertex to the base level.
    fn extend_last_level(&mut self, neighbors: &[Neighbor]) {
        self.diverse_neighbors_nums.push(0);
        let mut trimmed: Vec<Neighbor> = Vec::new();
        let mut num_diverse = 0usize;
        self.trim_sorted_neighbors(neighbors, &mut trimmed, &mut num_diverse);
        if let Some(slot) = self.diverse_neighbors_nums.last_mut() {
            *slot = num_diverse;
        }

        let new_item_id = self.levels.front().map_or(0, DynamicDenseGraph::get_size);
        for edge in &trimmed {
            self.try_add_inverse_edge(*edge, new_item_id);
        }
        if let Some(front) = self.levels.front_mut() {
            front.append(&trimmed);
        }
    }

    /// `TrimSortedNeighbors` (`index_base.h:388-426`): the HNSW diversity heuristic —
    /// keep a neighbor only if no already-kept neighbor is closer to it than the
    /// query is; the rest fill remaining slots as "clustering" edges. Writes the
    /// diverse count into `num_diverse`.
    fn trim_sorted_neighbors(
        &self,
        neighbors: &[Neighbor],
        result: &mut Vec<Neighbor>,
        num_diverse: &mut usize,
    ) {
        if neighbors.is_empty() {
            *num_diverse = 0;
            return;
        }
        let max_neighbors = self.opts.max_neighbors.min(neighbors.len());
        if let Some(first) = neighbors.first() {
            result.push(*first);
        }
        let mut clustering: Vec<Neighbor> = Vec::new();
        let mut pos = 1usize;
        while pos < neighbors.len() && result.len() < max_neighbors {
            let current = *neighbors.get(pos).unwrap_or(&NIL);
            let current_item = self.item(current.id);
            let mut take = true;
            for taken in result.iter() {
                let neighbor_item = self.item(taken.id);
                let dist = l2_sqr_f32(current_item, neighbor_item);
                if dist < current.dist {
                    take = false;
                    break;
                }
            }
            if take {
                result.push(current);
            } else if result.len() + clustering.len() < max_neighbors {
                clustering.push(current);
            }
            pos += 1;
        }
        *num_diverse = result.len();
        let mut cpos = 0usize;
        while result.len() < max_neighbors {
            if let Some(c) = clustering.get(cpos) {
                result.push(*c);
            } else {
                break;
            }
            cpos += 1;
        }
    }

    /// `TryAddInverseEdge` (`index_base.h:262-309`): consider adding the reverse edge
    /// `edge.id -> source_id` at the base level, respecting the diversity ordering;
    /// may escalate to [`Self::retrim_and_add_inverse_edge`].
    fn try_add_inverse_edge(&mut self, edge: Neighbor, source_id: usize) {
        let new_distance = edge.dist;
        let destination_id = edge.id;
        let num_diverse = *self
            .diverse_neighbors_nums
            .get(destination_id)
            .unwrap_or(&0);
        let neighbor_count = self.levels.front().map_or(0, DynamicDenseGraph::get_neighbor_count);
        let new_num_neighbors = (neighbor_count + 1).min(self.opts.max_neighbors);

        let ids = self
            .levels
            .front()
            .map(|g| g.neighbor_ids(destination_id))
            .unwrap_or_default();
        let dists = self
            .levels
            .front()
            .map(|g| g.neighbor_distances(destination_id))
            .unwrap_or_default();

        let mut is_diverse = true;
        let mut neighbor_pos = 0usize;
        while neighbor_pos < num_diverse {
            let nd = *dists.get(neighbor_pos).unwrap_or(&0.0);
            if new_distance < nd {
                break;
            }
            let diverse_neighbor_id = *ids.get(neighbor_pos).unwrap_or(&0);
            let dist_to_diverse =
                l2_sqr_f32(self.item(diverse_neighbor_id), self.item(source_id));
            if dist_to_diverse < new_distance {
                is_diverse = false;
                break;
            }
            neighbor_pos += 1;
        }

        let need_retrim = neighbor_count > 0 && is_diverse && neighbor_pos < num_diverse;
        if need_retrim {
            self.retrim_and_add_inverse_edge(edge, source_id);
            return;
        }

        let mut new_neighbor_pos = num_diverse;
        if !is_diverse {
            while new_neighbor_pos < neighbor_count
                && *dists.get(new_neighbor_pos).unwrap_or(&0.0) < new_distance
            {
                new_neighbor_pos += 1;
            }
        }
        if new_neighbor_pos >= new_num_neighbors {
            return;
        }
        if let Some(slot) = self.diverse_neighbors_nums.get_mut(destination_id) {
            *slot += usize::from(is_diverse);
        }
        self.add_edge_on_position(
            new_neighbor_pos,
            new_num_neighbors,
            destination_id,
            source_id,
            new_distance,
        );
    }

    /// `AddEdgeOnPosition` (`index_base.h:311-332`): splice a new edge into
    /// `edge_start_id`'s neighbor list at `new_neighbor_pos`, truncating to
    /// `new_num_neighbors`.
    fn add_edge_on_position(
        &mut self,
        new_neighbor_pos: usize,
        new_num_neighbors: usize,
        edge_start_id: usize,
        edge_end_id: usize,
        edge_distance: f32,
    ) {
        let ids = self
            .levels
            .front()
            .map(|g| g.neighbor_ids(edge_start_id))
            .unwrap_or_default();
        let dists = self
            .levels
            .front()
            .map(|g| g.neighbor_distances(edge_start_id))
            .unwrap_or_default();

        let mut neighbors: Vec<Neighbor> = Vec::with_capacity(new_num_neighbors);
        for old_pos in 0..new_neighbor_pos {
            neighbors.push(Neighbor {
                dist: *dists.get(old_pos).unwrap_or(&0.0),
                id: *ids.get(old_pos).unwrap_or(&0),
            });
        }
        neighbors.push(Neighbor {
            dist: edge_distance,
            id: edge_end_id,
        });
        let mut old_pos = new_neighbor_pos;
        while neighbors.len() < new_num_neighbors {
            neighbors.push(Neighbor {
                dist: *dists.get(old_pos).unwrap_or(&0.0),
                id: *ids.get(old_pos).unwrap_or(&0),
            });
            old_pos += 1;
        }
        if let Some(front) = self.levels.front_mut() {
            front.replace_neighbors(edge_start_id, &neighbors);
        }
    }

    /// `RetrimAndAddInverseEdge` (`index_base.h:334-386`): merge the new edge into
    /// `destination_id`'s existing (diverse ++ clustering) neighbor stream in
    /// distance order, then re-run the diversity trim.
    fn retrim_and_add_inverse_edge(&mut self, edge: Neighbor, source_id: usize) {
        let new_distance = edge.dist;
        let destination_id = edge.id;
        let num_diverse = *self
            .diverse_neighbors_nums
            .get(destination_id)
            .unwrap_or(&0);
        let neighbor_count = self.levels.front().map_or(0, DynamicDenseGraph::get_neighbor_count);

        let ids = self
            .levels
            .front()
            .map(|g| g.neighbor_ids(destination_id))
            .unwrap_or_default();
        let dists = self
            .levels
            .front()
            .map(|g| g.neighbor_distances(destination_id))
            .unwrap_or_default();

        let mut neighbors: Vec<Neighbor> = Vec::with_capacity(neighbor_count + 1);
        let mut new_edge_available = true;
        let mut diverse_pos = 0usize;
        let mut clustering_pos = num_diverse;
        while neighbors.len() < neighbor_count + 1 {
            let diverse_available = diverse_pos != num_diverse;
            let clustering_available = clustering_pos != neighbor_count;
            let mut next_neighbor = NIL;

            let clustering_dist = *dists.get(clustering_pos).unwrap_or(&0.0);
            let diverse_dist = *dists.get(diverse_pos).unwrap_or(&0.0);
            if clustering_available && (!diverse_available || clustering_dist < diverse_dist) {
                next_neighbor = Neighbor {
                    dist: clustering_dist,
                    id: *ids.get(clustering_pos).unwrap_or(&0),
                };
                clustering_pos += 1;
            } else if diverse_available {
                next_neighbor = Neighbor {
                    dist: diverse_dist,
                    id: *ids.get(diverse_pos).unwrap_or(&0),
                };
                diverse_pos += 1;
            }

            let add_old_edge = diverse_available || clustering_available;
            if new_edge_available && (!add_old_edge || new_distance < next_neighbor.dist) {
                neighbors.push(Neighbor {
                    dist: new_distance,
                    id: source_id,
                });
                new_edge_available = false;
            }
            if add_old_edge && neighbors.len() < neighbor_count + 1 {
                neighbors.push(next_neighbor);
            }
        }

        let mut trimmed: Vec<Neighbor> = Vec::new();
        let mut nd = 0usize;
        self.trim_sorted_neighbors(&neighbors, &mut trimmed, &mut nd);
        if let Some(slot) = self.diverse_neighbors_nums.get_mut(destination_id) {
            *slot = nd;
        }
        if let Some(front) = self.levels.front_mut() {
            front.replace_neighbors(destination_id, &trimmed);
        }
    }
}

// ===========================================================================
// HnswKnnCloud — a KnnCloud-shaped wrapper so the KNN calcer can swap backends.
// ===========================================================================

/// The online-HNSW neighbor cloud, exposing the same `add_vector` / insertion-order
/// / `nearest_neighbors` surface as the brute-force-exact [`crate::KnnCloud`] so the
/// KNN calcer can select it as the DEFAULT (parity) backend (D-03).
///
/// `add_vector` mirrors upstream's `TKNNUpdatableCloud::AddItem` (which calls
/// `GetNearestNeighborsAndAddItem`, building the graph); `nearest_neighbors` mirrors
/// `Compute`'s read (`GetNearestNeighbors(embed, k)`).
#[derive(Debug, Clone)]
pub struct HnswKnnCloud {
    dimension: usize,
    index: OnlineHnswIndex,
}

/// The search-neighborhood size (`ef`) the KNN calcer pins
/// (`TOnlineHnswBuildOptions({CloseNum, 300})`, `knn.h:109`).
pub const KNN_SEARCH_NEIGHBORHOOD_SIZE: usize = 300;

impl HnswKnnCloud {
    /// A new empty online-HNSW cloud over `dimension`-dim vectors with query `k`
    /// (`close_num`). `MaxNeighbors = close_num`, `SearchNeighborhoodSize = 300`.
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] if the derived options are invalid.
    pub fn new(dimension: usize, close_num: usize) -> CbResult<Self> {
        Ok(HnswKnnCloud {
            dimension,
            index: OnlineHnswIndex::new(dimension, close_num, KNN_SEARCH_NEIGHBORHOOD_SIZE)?,
        })
    }

    /// Number of inserted vectors so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether no vectors have been inserted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Insert `embed` (`AddItem` → `GetNearestNeighborsAndAddItem`, building the
    /// incremental graph). The insertion id equals [`Self::len`] before the call.
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] if `embed.len()` != the cloud dimension.
    pub fn add_vector(&mut self, embed: &[f32]) -> CbResult<()> {
        if embed.len() != self.dimension {
            return Err(CbError::OutOfRange(format!(
                "HnswKnnCloud::add_vector: embedding length {} != cloud dimension {}",
                embed.len(),
                self.dimension
            )));
        }
        self.index.get_nearest_neighbors_and_add_item(embed)?;
        Ok(())
    }

    /// The `min(k, len)` approximate nearest insertion ids to `query`, nearest
    /// first (`GetNearestNeighbors(embed, k)` → `TKNNUpdatableCloud::GetNearestNeighbors`).
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] if `query.len()` != the cloud dimension.
    pub fn nearest_neighbors(&self, query: &[f32], k: usize) -> CbResult<Vec<usize>> {
        if query.len() != self.dimension {
            return Err(CbError::OutOfRange(format!(
                "HnswKnnCloud::nearest_neighbors: query length {} != cloud dimension {}",
                query.len(),
                self.dimension
            )));
        }
        Ok(self
            .index
            .get_nearest_neighbors(query, k)
            .into_iter()
            .map(|n| n.id)
            .collect())
    }
}
