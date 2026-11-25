//! The CPU wave engine on a padded f32 grid, rows in parallel (docs/spec/30-engine.md, deviations in
//! docs/spec/05-deviations.md; docs/physics.md for the scheme).
//!
//! Buffers are padded by [`PAD`] rings: `stride = grid width + 2 PAD`, grid cell (gx, gy) at index
//! `(gy + PAD) * stride + gx + PAD`, canvas pixel (x, y) at grid cell (x + rim, y + rim). The rings stay 0 (the
//! wall) unless the edges are open without a rim, where Mur's first-order condition sets them in `u` (`w` stays 0,
//! docs/physics.md, Open edges). A mirror cell has c2 = 0 and stays 0; taps across a mirror are dropped (pml.rs,
//! `cuts`).
//!
//! Stencils are weighted sums of differences from the centre, `K sum (u_k - u)`: each difference is exact in f32
//! for a smooth field (Sterbenz), so round-off scales with the result and the weights sum to 0 exactly. Summing
//! the taps first biases the 9-point's energy (docs/physics.md, Energy).
//!
//! The 9-point sub-step is one leapfrog pass: `next = u + ((u - prev) d + c2 L u)`. The 4th-order (DRP) stencil
//! applies the operator twice, `w = c2 L u`, then `next = u + ((u - prev) d + (w + c2 L w / 12))` (stencil.rs), in
//! one pass over bands of rows that keeps each band's `w` in a small buffer (`modified_pass`); `next` is written
//! over `prev`. With a rim, the CPML corrects the rim after each application (pml.rs).
//!
//! A capture planned for a step (`WaveEngine::plan_capture`) is converted by the step's last pass, each block row of
//! the planes as soon as its band has written the field and intensity rows of it (`Capturing`); the bands start on
//! canvas rows that are multiples of 4, so no block row straddles two. The 9-point's forcing, which changes a few
//! cells after its pass, has their block rows converted again after the step.
//!
//! Threads: one driver thread; each pass fans out over rows or bands on the rayon pool, reading only its input
//! field.

use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use super::capture;
use super::forcing::{IMAGE_LEVELS, SourceGroups, live};
use super::mur::{MurChain, mur_chains};
use super::pml::{self, CpuPml, RimCells, cut, taps};
use super::shared::{Shared, blend};
use super::stencil::{INV12, K9_AXIS, K9_DIAGONAL, PAD, StencilCoefficients, TimeScheme, centred};
use super::substeps::sub_steps;
use super::types::{
	Emitter, EngineOptions, FrameShape, PendingCapture, PendingProbes, ProbeCells, ProbeQueue, ProbeReadings,
	SceneData, Snapshot,
};
use super::{EngineError, WaveEngine};
use crate::model::Layer;

/// The reference engine. See the module docs for the grid layout and the scheme.
pub struct CpuEngine {
	shared: Shared,
	stride: usize,
	/// The axis weights a0..a3 of the stencil.
	axis: [f32; 4],
	/// c2 / n^2 per padded cell.
	c2: Vec<f32>,
	/// d^(1/n) per padded cell.
	d: Vec<f32>,
	/// The grid rows' stretches of equal medium, which the row loops take as constants.
	runs: MediumRuns,
	/// Cut bytes per padded cell (pml.rs, `cuts`); all 0 without mirrors.
	cuts: Vec<u8>,
	mirrors: bool,
	u: Vec<f32>,
	prev: Vec<f32>,
	/// The modified equation's `c2 L u` on the cells the CPML's second application reads: each sub-step writes only
	/// the cells within `rim + 3` of the grid's edge ([`rim_w`]), the rest stay 0, so it is not a full plane of `w`.
	/// Empty without a rim.
	w: Vec<f32>,
	/// The CPML's corrections of `w`; empty without a rim.
	corrections: Option<RimCells>,
	/// The intensity's EMA stages; empty unless the intensity is tracked.
	s1: Vec<f32>,
	s2: Vec<f32>,
	mur: Vec<MurChain>,
	pml: Option<CpuPml>,
	groups: SourceGroups,
	/// `groups` by padded row.
	group_rows: GroupRows,
	/// The source table of the current sub-step (forcing.rs).
	values: Vec<f32>,
	alpha: f32,
	/// Per probe its layer and padded cells.
	probes: Vec<(Layer, Vec<usize>)>,
	probe_queue: ProbeQueue,
	probe_reading: ProbeReadings,
	/// The shape of the capture planned for the next step ([`WaveEngine::plan_capture`]).
	planned: Option<FrameShape>,
	/// The planes the last step captured while writing the canvas, for a `begin_capture` of their shape.
	captured: Option<Snapshot>,
	/// Planes to capture into next, the allocations of a capture handed back.
	planes: Option<Snapshot>,
}

/// The bytes [`CpuEngine`] allocates by the grid of `scene` with `options`: the field, medium and stage planes, the
/// mirror cuts and the CPML's auxiliary fields.
pub fn grid_bytes(scene: &SceneData, options: &EngineOptions) -> u64 {
	let cells = scene.padded_cells() as u64;
	let strips = pml::Strips::of(scene);
	let modified = TimeScheme::of(options.stencil) == TimeScheme::ModifiedEquation && !strips.is_empty();
	let planes = 4 + u64::from(modified) + 2 * u64::from(options.track_intensity);
	let (psi, xi) = (strips.len() as u64, (strips.len() + strips.y_len()) as u64);
	let corrections = if modified { strips.rim_cells() as u64 } else { 0 };
	cells * (4 * planes + 1) + 2 * 4 * (psi + xi) + 4 * corrections
}

impl CpuEngine {
	/// An engine at rest on `scene` with the sub-steps its CFL bound requires.
	pub fn new(scene: impl Into<Arc<SceneData>>, options: EngineOptions) -> Self {
		let scene = scene.into();
		let n = sub_steps(options.stencil, scene.max_c2());
		Self::with_sub_steps(scene, options, n)
	}

	/// An engine taking `n` sub-steps per step whatever the scene; below the CFL bound it diverges.
	/// For tests of the bound.
	pub fn with_sub_steps(scene: impl Into<Arc<SceneData>>, options: EngineOptions, n: u32) -> Self {
		assert!(n >= 1, "at least one sub-step");
		let scene = scene.into();
		let stride = scene.stride();
		let cells = scene.padded_cells();
		let stages = if options.track_intensity { cells } else { 0 };
		let mut engine = Self {
			stride,
			axis: StencilCoefficients::of(options.stencil).axis.map(|a| a as f32),
			c2: vec![0.0; cells],
			d: vec![1.0; cells],
			runs: MediumRuns::default(),
			cuts: Vec::new(),
			mirrors: false,
			u: vec![0.0; cells],
			prev: vec![0.0; cells],
			w: Vec::new(),
			corrections: None,
			s1: vec![0.0; stages],
			s2: vec![0.0; stages],
			mur: Vec::new(),
			pml: None,
			groups: SourceGroups::default(),
			group_rows: GroupRows::default(),
			values: vec![0.0; IMAGE_LEVELS],
			alpha: 0.0,
			probes: Vec::new(),
			probe_queue: ProbeQueue::default(),
			probe_reading: ProbeReadings::default(),
			planned: None,
			captured: None,
			planes: None,
			shared: Shared::new(scene, options, n),
		};
		engine.load_scene();
		engine.update_alpha();
		engine
	}

	/// Sets the field now and one sub-step before, each grid-sized (no padding), row-major.
	/// For analytic tests that start from a given state.
	pub fn set_field(&mut self, now: &[f32], before: &[f32]) {
		let scene = &self.shared.scene;
		let gw = scene.grid_width() as usize;
		assert_eq!(now.len(), gw * scene.grid_height() as usize);
		assert_eq!(before.len(), now.len());
		for (gy, (a, b)) in now.chunks_exact(gw).zip(before.chunks_exact(gw)).enumerate() {
			let at = scene.padded_index(0, gy);
			self.u[at..at + gw].copy_from_slice(a);
			self.prev[at..at + gw].copy_from_slice(b);
		}
		self.captured = None;
	}

	/// The padded field buffer (see [`SceneData::stride`]).
	pub fn field_padded(&self) -> &[f32] {
		&self.u
	}

	// The inherent calls below take `&mut self` like the trait's, so they win method resolution on a
	// `CpuEngine`: callers on the concrete type get no `Result`.

	/// [`WaveEngine::advance_one_step`], which cannot fail here. With a capture planned, its last sub-step converts
	/// the canvas rows into the planes as it writes them.
	pub fn advance_one_step(&mut self) {
		let mut planes = self.planned_planes();
		let n = self.shared.sub_steps;
		for k in 0..n {
			self.shared.source_values(&mut self.values, k);
			self.sub_step(self.shared.accumulates(k), if k + 1 == n { planes.as_mut() } else { None });
		}
		self.shared.time += 1;
		if let Some(mut out) = planes {
			if self.shared.scheme == TimeScheme::Leapfrog {
				self.recapture_forced(&mut out);
			}
			out.step = self.shared.time - 1;
			self.captured = Some(out);
		}
	}

	/// Planes for the capture planned for this step, shaped and sized, when it can be made in the pass: a shape of this
	/// canvas whose divisor divides 4 ([`band_lengths`]). An unclaimed capture's planes are kept for reuse.
	fn planned_planes(&mut self) -> Option<Snapshot> {
		if let Some(stale) = self.captured.take() {
			self.planes = Some(stale);
		}
		let scene = &self.shared.scene;
		let shape = self.planned.take().filter(|p| {
			(p.canvas_width, p.canvas_height) == (scene.canvas_width, scene.canvas_height)
				&& p.divisor > 0
				&& 4 % p.divisor == 0
		})?;
		let mut out = self.planes.take().unwrap_or_else(|| Snapshot::new(shape));
		out.shape = shape;
		out.field.resize(shape.plane_bytes(), 0);
		if self.shared.options.track_intensity {
			out.intensity.resize(shape.plane_bytes(), 0);
		} else {
			out.intensity.clear();
		}
		Some(out)
	}

	/// Converts again the block rows of `out` that hold a cell the leapfrog's forcing drove after the pass wrote them
	/// ([`force_next`](Self::force_next)), from the field and the stages after the step.
	fn recapture_forced(&self, out: &mut Snapshot) {
		let scene = &self.shared.scene;
		let (s, rim, w, h) =
			(self.stride, scene.rim as usize, scene.canvas_width as usize, scene.canvas_height as usize);
		let shape = out.shape;
		let div = shape.divisor as usize;
		let mut rows: Vec<usize> = (self.groups.cells.iter())
			.filter_map(|&c| {
				let (x, y) = ((c % s).checked_sub(PAD + rim)?, (c / s).checked_sub(PAD + rim)?);
				(x < w && y < h).then_some(y / div)
			})
			.collect();
		rows.sort_unstable();
		rows.dedup();
		let capturing = self.capturing(&shape);
		let mut scratch = CaptureScratch::default();
		let bytes = capture::row_bytes(&shape);
		for by in rows {
			let (y0, at) = (by * div, scene.canvas_index(0, by * div));
			let stages = (!self.s1.is_empty()).then(|| (&self.s1[at..], &self.s2[at..]));
			let intensity = (!out.intensity.is_empty()).then(|| &mut out.intensity[by * bytes..][..bytes]);
			let field = &mut out.field[by * bytes..][..bytes];
			capturing.block(&self.u[at..], stages, div.min(h - y0), &mut scratch, field, intensity);
		}
	}

	/// The capture of `shape` a pass makes.
	fn capturing<'a>(&self, shape: &'a FrameShape) -> Capturing<'a> {
		let scene = &self.shared.scene;
		Capturing { shape, scale: self.shared.options.intensity_scale as f32, rim: scene.rim as usize, s: self.stride }
	}

	/// [`WaveEngine::sample_probes`], which cannot fail here.
	pub fn sample_probes(&mut self, out: &mut ProbeReadings) {
		out.clear();
		for (layer, cells) in &self.probes {
			match layer {
				Layer::Field => out.push(cells.iter().map(|&c| self.u[c])),
				Layer::Intensity => out.push(cells.iter().map(|&c| self.blended(c))),
			}
		}
	}

	/// Converts the canvas after the last step into `out`'s planes in `shape`, their allocations reused.
	pub fn capture_into(&self, shape: &FrameShape, out: &mut Snapshot) {
		let (scene, options) = (&self.shared.scene, &self.shared.options);
		assert_eq!((shape.canvas_width, shape.canvas_height), (scene.canvas_width, scene.canvas_height));
		capture::plane_rows(shape, Layer::Field, &mut out.field, |y, row| {
			let at = scene.canvas_index(0, y);
			row.copy_from_slice(&self.u[at..at + row.len()]);
		});
		if options.track_intensity {
			let scale = options.intensity_scale as f32;
			capture::plane_rows(shape, Layer::Intensity, &mut out.intensity, |y, row| {
				let at = scene.canvas_index(0, y);
				let n = row.len();
				for ((r, &a), &b) in row.iter_mut().zip(&self.s1[at..at + n]).zip(&self.s2[at..at + n]) {
					*r = blend(a, b, scale);
				}
			});
		} else {
			out.intensity.clear();
		}
		out.shape = *shape;
		out.step = self.shared.time.saturating_sub(1);
	}

	/// [`WaveEngine::capture_frame`], which cannot fail here.
	pub fn capture_frame(&mut self, shape: &FrameShape) -> Snapshot {
		let mut out = Snapshot::new(*shape);
		self.capture_into(shape, &mut out);
		out
	}

	/// [`WaveEngine::prepare_frame`], which cannot fail here.
	pub fn prepare_frame(&mut self) -> (Vec<f32>, Vec<f32>) {
		let stages = (!self.s1.is_empty()).then_some((&self.s1[..], &self.s2[..]));
		self.shared.canvas_frame(&self.u, stages)
	}

	/// Copies the scene's maps into the padded buffers at the sub-step scale and builds the medium's runs, the
	/// cuts, the Mur chains, the PML and the source groups.
	fn load_scene(&mut self) {
		let (scene, n) = (&self.shared.scene, self.shared.sub_steps);
		scene.fill_sub_step_medium(n, &mut self.c2, &mut self.d);
		let (gw, gh) = (scene.grid_width() as usize, scene.grid_height() as usize);
		self.runs = MediumRuns::new(&self.c2, &self.d, self.stride, gw, gh);
		self.cuts = pml::cuts(scene);
		self.mirrors = self.cuts.iter().any(|&c| c != 0);
		self.mur = mur_chains(scene, &self.c2, &self.d);
		let cross = StencilCoefficients::of(self.shared.options.stencil).cross as f32;
		self.pml = CpuPml::new(scene, n, self.axis, cross);
		let rim = self.pml.is_some() && self.shared.scheme == TimeScheme::ModifiedEquation;
		self.w = if rim { vec![0.0; self.u.len()] } else { Vec::new() };
		self.corrections = rim.then(|| RimCells::new(pml::Strips::of(scene)));
		self.group_sources();
	}

	fn update_alpha(&mut self) {
		self.alpha = self.shared.alpha();
	}

	fn group_sources(&mut self) {
		self.groups = self.shared.groups();
		self.group_rows = GroupRows::new(&self.groups.cells, self.stride, self.u.len() / self.stride);
		self.values.resize(self.shared.sources.table_len(), 0.0);
	}

	/// Adds `h F` to the next field at the driven cells, where leapfrog takes `h A u`, and moves the
	/// intensity stages of the last sub-step to the forced value.
	fn force_next(&mut self, intensity: bool) {
		let rest = 1.0 - self.alpha;
		for (cell, run) in self.groups.iter() {
			let (before, h) = (self.prev[cell], centred(self.d[cell]));
			let mut v = before;
			let mut driven = false;
			for &e in run {
				let f = self.values[e as usize];
				if live(f) {
					v += h * f;
					driven = true;
				}
			}
			if !driven {
				continue;
			}
			self.prev[cell] = v;
			if intensity {
				let s1 = self.s1[cell] + (v * v - before * before) * rest;
				self.s2[cell] += (s1 - self.s1[cell]) * rest;
				self.s1[cell] = s1;
			}
		}
	}

	/// The intensity stages and their weight when `intensity` (and tracked).
	fn stages<'a>(s1: &'a mut [f32], s2: &'a mut [f32], alpha: f32, intensity: bool) -> Option<Stages<'a>> {
		(intensity && !s1.is_empty()).then_some(Stages { s1, s2, alpha })
	}

	/// The leapfrog pass of the 9-point stencil: writes the next field over `prev`, the intensity stages when
	/// `intensity`, row by row ([`grid_rows`]), or with `planes` given in bands of [`BAND`] rows that also write them.
	fn leapfrog_pass(&mut self, intensity: bool, planes: Option<&mut Snapshot>) {
		let (s, gw) = (self.stride, self.shared.scene.grid_width() as usize);
		let shape = planes.as_ref().map(|p| p.shape);
		let capturing = shape.as_ref().map(|shape| self.capturing(shape));
		let (u, c2, d, runs, cuts, mirrors) = (&self.u, &self.c2, &self.d, &self.runs, &self.cuts, self.mirrors);
		let scene = &self.shared.scene;
		let (gh, rim) = (scene.grid_height() as usize, scene.rim as usize);
		let stages = Self::stages(&mut self.s1, &mut self.s2, self.alpha, intensity);
		let row = |r: usize, next: &mut [f32]| {
			let at = r * s + PAD;
			runs.of_row(r, row_cuts(cuts, mirrors, at, gw), |cells, medium, cuts| {
				let (at, len, next) = (at + cells.start, cells.len(), &mut next[cells.clone()]);
				match medium {
					Some(m) => nine_row(u, at, s, len, cuts, |x, v, lap| {
						next[x] = v + ((v - next[x]) * m.d + lap * m.c2 * m.centred);
					}),
					None => {
						let (c2, d) = (&c2[at..][..len], &d[at..][..len]);
						nine_row(u, at, s, len, cuts, |x, v, lap| {
							next[x] = v + ((v - next[x]) * d[x] + lap * c2[x] * centred(d[x]));
						});
					}
				}
			});
		};
		let Some(planes) = planes else { return grid_rows(&mut self.prev, stages, s, gw, row) };
		let alpha = stages.as_ref().map_or(0.0, |st| st.alpha);
		let bands = bands(&mut self.prev, stages, None, Some(planes), s, gh, rim);
		bands.into_par_iter().for_each_init(CaptureScratch::default, |scratch, band| {
			let Band { r0, next, mut stages, mut planes, .. } = band;
			for k in 0..next.len() / s {
				let line = &mut next[k * s + PAD..][..gw];
				row(r0 + k, line);
				if let Some((s1, s2)) = &mut stages {
					accumulate(&mut s1[k * s + PAD..][..gw], &mut s2[k * s + PAD..][..gw], line, alpha);
				}
				if let (Some(c), Some(p)) = (&capturing, &mut planes) {
					c.after_row(
						r0 + k - PAD,
						r0 - PAD,
						next,
						stages.as_ref().map(|(a, b)| (&a[..], &b[..])),
						p,
						scratch,
					);
				}
			}
		});
	}

	/// The modified equation's pass: writes the next field over `prev`, the intensity stages when `intensity`, and
	/// `planes` when given. Each band of [`BAND`] grid rows computes its `w` with 3 rows each side into a buffer of
	/// its job ([`Modified::w_row`]), each `w` row just before the first next row that reads it, so `w` stays in
	/// cache; the rows on either side are computed by both bands, from the same inputs. The cells the CPML's second
	/// application reads go to `w` ([`rim_w`]).
	fn modified_pass(&mut self, intensity: bool, planes: Option<&mut Snapshot>) {
		let scene = &self.shared.scene;
		let shape = planes.as_ref().map(|p| p.shape);
		let capturing = shape.as_ref().map(|shape| self.capturing(shape));
		let pass = Modified {
			u: &self.u,
			c2: &self.c2,
			d: &self.d,
			runs: &self.runs,
			cuts: &self.cuts,
			mirrors: self.mirrors,
			axis: &self.axis,
			corrections: self.corrections.as_ref(),
			groups: &self.groups,
			group_rows: &self.group_rows,
			values: &self.values,
			s: self.stride,
			gw: scene.grid_width() as usize,
			gh: scene.grid_height() as usize,
			rim: scene.rim as usize,
			capturing: capturing.as_ref(),
		};
		let stages = Self::stages(&mut self.s1, &mut self.s2, self.alpha, intensity);
		let alpha = stages.as_ref().map_or(0.0, |st| st.alpha);
		let w = (!self.w.is_empty()).then_some(&mut self.w[..]);
		let bands = bands(&mut self.prev, stages, w, planes, pass.s, pass.gh, pass.rim);
		bands.into_par_iter().for_each_init(<(Vec<f32>, CaptureScratch)>::default, |(buffer, scratch), band| {
			pass.band(band, buffer, scratch, alpha);
		});
	}

	/// One sub-step; the next field ends in `u`. Given `planes`, the pass converts the canvas into them as it writes it.
	fn sub_step(&mut self, intensity: bool, planes: Option<&mut Snapshot>) {
		match self.shared.scheme {
			TimeScheme::Leapfrog => {
				self.leapfrog_pass(intensity, planes);
				if let Some(p) = &mut self.pml {
					p.apply(0, &self.u, &self.cuts, &self.c2, &mut self.prev, 1.0);
				}
				self.force_next(intensity);
			}
			TimeScheme::ModifiedEquation => {
				if let (Some(p), Some(out)) = (&mut self.pml, &mut self.corrections) {
					p.apply_into(0, &self.u, &self.cuts, &self.c2, out, 1.0);
				}
				self.modified_pass(intensity, planes);
				if let Some(p) = &mut self.pml {
					p.apply(1, &self.w, &self.cuts, &self.c2, &mut self.prev, INV12);
				}
			}
		}
		self.apply_open_edges();
		std::mem::swap(&mut self.u, &mut self.prev);
	}

	/// Mur's condition on the first ring (mur.rs, `mur_weights`); the second and third rings, which only the DRP stencil
	/// reads, continue the field linearly from the edge cell through the first ring. Chaining Mur ring by ring grows a
	/// mode where two open edges meet at Courant numbers above about 0.75 (docs/physics.md, Open edges).
	fn apply_open_edges(&mut self) {
		for m in &self.mur {
			let [a, b, c] = m.weights;
			let [r1, r2, r3] = m.rings;
			let (edge, first) = (self.prev[m.inner], a * self.u[m.inner] + b * self.prev[m.inner] + c * self.u[r1]);
			self.prev[r1] = first;
			self.prev[r2] = 2.0 * first - edge;
			self.prev[r3] = 3.0 * first - 2.0 * edge;
		}
	}

	/// The intensity at padded cell `cell`, 0 when it is not tracked.
	fn blended(&self, cell: usize) -> f32 {
		if self.s1.is_empty() {
			return 0.0;
		}
		blend(self.s1[cell], self.s2[cell], self.shared.options.intensity_scale as f32)
	}
}

/// The intensity stages a pass advances, and the EMA weight.
struct Stages<'a> {
	s1: &'a mut [f32],
	s2: &'a mut [f32],
	alpha: f32,
}

/// Runs `row(r, cells)` on the `gw` grid cells of every grid row `r` of `out` (padded rows of `s`) in
/// parallel, then advances the intensity stages of those cells by the values written, if given.
fn grid_rows(out: &mut [f32], stages: Option<Stages<'_>>, s: usize, gw: usize, row: impl Fn(usize, &mut [f32]) + Sync) {
	let rows = out.len() / s;
	let grid = |r: usize| (PAD..rows - PAD).contains(&r);
	match stages {
		Some(Stages { s1, s2, alpha }) => out
			.par_chunks_mut(s)
			.zip(s1.par_chunks_mut(s))
			.zip(s2.par_chunks_mut(s))
			.enumerate()
			.filter(|(r, _)| grid(*r))
			.for_each(|(r, ((line, s1), s2))| {
				let cells = &mut line[PAD..][..gw];
				row(r, cells);
				accumulate(&mut s1[PAD..][..gw], &mut s2[PAD..][..gw], cells, alpha);
			}),
		None => out
			.par_chunks_mut(s)
			.enumerate()
			.filter(|(r, _)| grid(*r))
			.for_each(|(r, line)| row(r, &mut line[PAD..][..gw])),
	}
}

/// Grid rows of a band of the passes: the modified equation's computes 6 rows of `w` twice a band, and its `w`
/// buffer of `BAND + 6` rows should stay in the core's cache.
const BAND: usize = 32;

/// The lengths of the bands of `band` grid rows (a multiple of 4) over `gh` grid rows with `rim` rows of rim, the
/// first longer by `rim % 4`: every later band starts on a canvas row that is a multiple of 4, so each block row of a
/// capture (divisor 1, 2 or 4) lies in one band.
fn band_lengths(gh: usize, rim: usize, band: usize) -> impl Iterator<Item = usize> {
	let first = (band + rim % 4).min(gh);
	std::iter::once(first).chain((first..gh).step_by(band).map(move |g| band.min(gh - g)))
}

/// `v` split into consecutive pieces of `lens[k] * unit` elements.
fn split_lens<'a, T>(mut v: &'a mut [T], lens: &[usize], unit: usize) -> Vec<&'a mut [T]> {
	lens.iter()
		.map(|&n| {
			let (piece, rest) = std::mem::take(&mut v).split_at_mut(n * unit);
			v = rest;
			piece
		})
		.collect()
}

/// A band of a pass: its first padded row, its rows of `prev`, of the intensity stages when they advance, of `w`
/// when the pass keeps one, and of a capture's planes when the pass makes one.
struct Band<'a> {
	r0: usize,
	next: &'a mut [f32],
	stages: Option<(&'a mut [f32], &'a mut [f32])>,
	w: Option<&'a mut [f32]>,
	planes: Option<BandPlanes<'a>>,
}

/// The grid rows of the padded `prev`, `stages` and `w` (rows of `s`, `gh` grid rows with `rim` rows of rim) and the
/// block rows of `planes` in bands of [`BAND`] rows ([`band_lengths`]).
fn bands<'a>(
	prev: &'a mut [f32],
	stages: Option<Stages<'a>>,
	w: Option<&'a mut [f32]>,
	planes: Option<&'a mut Snapshot>,
	s: usize,
	gh: usize,
	rim: usize,
) -> Vec<Band<'a>> {
	let lens: Vec<usize> = band_lengths(gh, rim, BAND).collect();
	let grid = PAD * s..(PAD + gh) * s;
	let (s1, s2) = match stages {
		Some(Stages { s1, s2, .. }) => {
			(split_lens(&mut s1[grid.clone()], &lens, s), split_lens(&mut s2[grid.clone()], &lens, s))
		}
		None => (Vec::new(), Vec::new()),
	};
	let w = w.map_or_else(Vec::new, |w| split_lens(&mut w[grid.clone()], &lens, s));
	let planes = planes.map_or_else(Vec::new, |out| {
		let (h, div) = (out.shape.canvas_height as usize, out.shape.divisor as usize);
		let block = |g: usize| g.saturating_sub(rim).min(h).div_ceil(div);
		let starts: Vec<usize> = lens.iter().scan(0, |g, &n| Some(std::mem::replace(g, *g + n))).collect();
		let counts: Vec<usize> = starts.iter().zip(&lens).map(|(&g, &n)| block(g + n) - block(g)).collect();
		let bytes = capture::row_bytes(&out.shape);
		let field = split_lens(&mut out.field, &counts, bytes);
		let tracked = !out.intensity.is_empty();
		let mut intensity = tracked.then(|| split_lens(&mut out.intensity, &counts, bytes).into_iter());
		field
			.into_iter()
			.zip(&starts)
			.map(|(field, &g)| BandPlanes {
				by0: block(g),
				field,
				intensity: intensity.as_mut().and_then(Iterator::next),
			})
			.collect()
	});
	let (mut s1, mut s2, mut w, mut planes) = (s1.into_iter(), s2.into_iter(), w.into_iter(), planes.into_iter());
	let mut r0 = PAD;
	split_lens(&mut prev[grid], &lens, s)
		.into_iter()
		.zip(&lens)
		.map(|(next, &n)| {
			let band = Band { r0, next, stages: s1.next().zip(s2.next()), w: w.next(), planes: planes.next() };
			r0 += n;
			band
		})
		.collect()
}

/// A band's share of a capture: its first block row and its output rows of the field plane and of the intensity
/// plane when the intensity is tracked.
struct BandPlanes<'a> {
	by0: usize,
	field: &'a mut [u8],
	intensity: Option<&'a mut [u8]>,
}

/// Scratch of a capture made in a pass: the blended intensity of a block's rows and the block means.
#[derive(Default)]
struct CaptureScratch {
	rows: Vec<f32>,
	means: Vec<f32>,
}

/// A capture a pass makes in the last sub-step of a step ([`WaveEngine::plan_capture`]): the shape, the intensity's
/// scale, the rim and the padded row length.
struct Capturing<'a> {
	shape: &'a FrameShape,
	scale: f32,
	rim: usize,
	s: usize,
}

impl Capturing<'_> {
	/// Converts the block row that grid row `gy` completes, if it completes one, into `planes`: `next` and `stages`
	/// are the band's padded rows from grid row `g0`, written up to `gy`.
	fn after_row(
		&self,
		gy: usize,
		g0: usize,
		next: &[f32],
		stages: Option<(&[f32], &[f32])>,
		planes: &mut BandPlanes<'_>,
		scratch: &mut CaptureScratch,
	) {
		let (h, div) = (self.shape.canvas_height as usize, self.shape.divisor as usize);
		let Some(y) = gy.checked_sub(self.rim).filter(|&y| y < h) else { return };
		if (y + 1) % div != 0 && y + 1 != h {
			return;
		}
		let (by, y0) = (y / div, y / div * div);
		let at = (y0 + self.rim - g0) * self.s + PAD + self.rim;
		let bytes = capture::row_bytes(self.shape);
		let k = (by - planes.by0) * bytes;
		let intensity = planes.intensity.as_deref_mut().map(|p| &mut p[k..][..bytes]);
		let stages = stages.map(|(s1, s2)| (&s1[at..], &s2[at..]));
		self.block(&next[at..], stages, y + 1 - y0, scratch, &mut planes.field[k..][..bytes], intensity);
	}

	/// Converts the `n` canvas rows of a block row into `field` and, given the stages, `intensity`: `u` and `stages`
	/// hold the rows from their first canvas cell, rows of `s`.
	fn block(
		&self,
		u: &[f32],
		stages: Option<(&[f32], &[f32])>,
		n: usize,
		scratch: &mut CaptureScratch,
		field: &mut [u8],
		intensity: Option<&mut [u8]>,
	) {
		let (shape, s) = (self.shape, self.s);
		let w = shape.canvas_width as usize;
		scratch.means.resize(shape.width() as usize, 0.0);
		capture::block_row(shape, Layer::Field, u, s, n, &mut scratch.means, field);
		if let (Some((s1, s2)), Some(intensity)) = (stages, intensity) {
			scratch.rows.resize(w * n, 0.0);
			for (k, row) in scratch.rows.chunks_exact_mut(w).take(n).enumerate() {
				for ((r, &a), &b) in row.iter_mut().zip(&s1[k * s..][..w]).zip(&s2[k * s..][..w]) {
					*r = blend(a, b, self.scale);
				}
			}
			capture::block_row(shape, Layer::Intensity, &scratch.rows, w, n, &mut scratch.means, intensity);
		}
	}
}

/// The inputs of [`CpuEngine::modified_pass`].
struct Modified<'a> {
	u: &'a [f32],
	c2: &'a [f32],
	d: &'a [f32],
	runs: &'a MediumRuns,
	cuts: &'a [u8],
	mirrors: bool,
	axis: &'a [f32; 4],
	corrections: Option<&'a RimCells>,
	groups: &'a SourceGroups,
	group_rows: &'a GroupRows,
	values: &'a [f32],
	s: usize,
	gw: usize,
	gh: usize,
	rim: usize,
	/// The capture the pass makes, if any.
	capturing: Option<&'a Capturing<'a>>,
}

impl Modified<'_> {
	/// Runs `band` with `buffer` for its `w` rows and `scratch` for its capture; `alpha` the intensity's EMA weight.
	fn band(&self, band: Band<'_>, buffer: &mut Vec<f32>, scratch: &mut CaptureScratch, alpha: f32) {
		let Band { r0, next, mut stages, mut w, mut planes } = band;
		let (s, gw) = (self.s, self.gw);
		let h = next.len() / s;
		if buffer.len() < (h + 6) * s {
			buffer.resize((h + 6) * s, 0.0);
		}
		// buffer row k holds w of padded row r0 - 3 + k
		let buffer = &mut buffer[..(h + 6) * s];
		for k in 0..6 {
			self.w_row(r0 - 3 + k, &mut buffer[k * s..][..s]);
		}
		for k in 0..h {
			let r = r0 + k;
			self.w_row(r + 3, &mut buffer[(k + 6) * s..][..s]);
			if let Some(w) = &mut w {
				rim_w(&buffer[(k + 3) * s..][..s], &mut w[k * s..][..s], r - PAD, gw, self.gh, self.rim);
			}
			let line = &mut next[k * s + PAD..][..gw];
			self.next_row(r, buffer, (k + 3) * s + PAD, line);
			if let Some((s1, s2)) = &mut stages {
				accumulate(&mut s1[k * s + PAD..][..gw], &mut s2[k * s + PAD..][..gw], line, alpha);
			}
			if let (Some(c), Some(p)) = (self.capturing, &mut planes) {
				c.after_row(r - PAD, r0 - PAD, next, stages.as_ref().map(|(a, b)| (&a[..], &b[..])), p, scratch);
			}
		}
	}

	/// `w = c2 L u` of padded row `q` into `out` (`s` values, 0 on the padding and off the grid), then the CPML's
	/// corrections and the sub-step's forcing `F` at the driven cells, where the modified equation takes `A u`
	/// (forcing.rs).
	fn w_row(&self, q: usize, out: &mut [f32]) {
		let (u, c2, s, gw, axis) = (self.u, self.c2, self.s, self.gw, self.axis);
		if !(PAD..PAD + self.gh).contains(&q) {
			out.fill(0.0);
			return;
		}
		out[..PAD].fill(0.0);
		out[PAD + gw..].fill(0.0);
		let line = &mut out[PAD..][..gw];
		let at = q * s + PAD;
		self.runs.of_row(q, row_cuts(self.cuts, self.mirrors, at, gw), |cells, medium, cuts| {
			let (at, len, w) = (at + cells.start, cells.len(), &mut line[cells.clone()]);
			match medium {
				Some(m) => drp_row(u, at, s, len, axis, cuts, |x, lap| w[x] = m.c2 * lap),
				None => {
					let c2 = &c2[at..][..len];
					drp_row(u, at, s, len, axis, cuts, |x, lap| w[x] = c2[x] * lap);
				}
			}
		});
		if let Some(corrections) = self.corrections {
			let [left, mid, right] = corrections.row(q - PAD);
			for (from, c) in [(0, left), (left.len(), mid), (gw - right.len(), right)] {
				for (w, &c) in line[from..][..c.len()].iter_mut().zip(c) {
					*w += c;
				}
			}
		}
		for &g in self.group_rows.of(q) {
			let g = g as usize;
			let (start, len) = self.groups.runs[g];
			let x = self.groups.cells[g] - at;
			let mut v = line[x];
			for &e in &self.groups.entries[start as usize..][..len as usize] {
				let f = self.values[e as usize];
				if live(f) {
					v += f;
				}
			}
			line[x] = v;
		}
	}

	/// The next field of padded row `r` over `next` (its grid cells of `prev`) from `w`, whose row's grid cells
	/// start at `w_at`.
	fn next_row(&self, r: usize, w: &[f32], w_at: usize, next: &mut [f32]) {
		let (u, c2, d, s, gw, axis) = (self.u, self.c2, self.d, self.s, self.gw, self.axis);
		let at = r * s + PAD;
		self.runs.of_row(r, row_cuts(self.cuts, self.mirrors, at, gw), |cells, medium, cuts| {
			let (at, w_at, len, next) = (at + cells.start, w_at + cells.start, cells.len(), &mut next[cells.clone()]);
			let (uc, wc) = (&u[at..][..len], &w[w_at..][..len]);
			match medium {
				Some(m) => drp_row(w, w_at, s, len, axis, cuts, |x, lw| {
					let c = uc[x];
					next[x] = c + ((c - next[x]) * m.d + (wc[x] + m.c2 * lw * INV12) * m.centred);
				}),
				None => {
					let (c2, d) = (&c2[at..][..len], &d[at..][..len]);
					drp_row(w, w_at, s, len, axis, cuts, |x, lw| {
						let c = uc[x];
						next[x] = c + ((c - next[x]) * d[x] + (wc[x] + c2[x] * lw * INV12) * centred(d[x]));
					});
				}
			}
		});
	}
}

/// Copies the cells of padded row `from` (of grid row `gy`) that the CPML's second application reads to `to`:
/// those within `rim + 3` cells of the grid's edge (the x strips and the y strips with the stencil's reach).
fn rim_w(from: &[f32], to: &mut [f32], gy: usize, gw: usize, gh: usize, rim: usize) {
	let e = rim + 3;
	if gy < e || gy + e >= gh || 2 * e >= gw {
		to.copy_from_slice(from);
		return;
	}
	to[PAD..][..e].copy_from_slice(&from[PAD..][..e]);
	to[PAD + gw - e..][..e].copy_from_slice(&from[PAD + gw - e..][..e]);
}

/// The source groups by padded row: per row the indices of its groups ([`SourceGroups`]).
#[derive(Default)]
struct GroupRows {
	groups: Vec<u32>,
	/// The first entry of each padded row in `groups`, then their number.
	rows: Vec<usize>,
}

impl GroupRows {
	/// The rows of groups at padded `cells` on `rows` rows of `s`.
	fn new(cells: &[usize], s: usize, rows: usize) -> Self {
		let mut groups: Vec<u32> = (0..cells.len() as u32).collect();
		groups.sort_by_key(|&g| cells[g as usize]);
		let mut starts = Vec::with_capacity(rows + 1);
		let mut k = 0;
		for r in 0..=rows {
			while k < groups.len() && cells[groups[k] as usize] < r * s {
				k += 1;
			}
			starts.push(k);
		}
		Self { groups, rows: starts }
	}

	/// The groups on padded row `r`.
	fn of(&self, r: usize) -> &[u32] {
		&self.groups[self.rows[r]..self.rows[r + 1]]
	}
}

/// The shortest stretch of a row with one medium that the row loops take as constants ([`MediumRuns`]).
const MIN_RUN: usize = 16;

/// The medium a stretch of cells shares: its c2, d and `centred(d)`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Medium {
	c2: f32,
	d: f32,
	centred: f32,
}

/// A stretch of a grid row's cells: `medium` when they share one, else None and each cell reads its own.
#[derive(Clone, Debug, PartialEq)]
struct Run {
	cells: Range<usize>,
	medium: Option<Medium>,
}

/// Per grid row the stretches of [`MIN_RUN`] cells or more with the same c2 and d bits, and the cells between,
/// in order and covering the row. Most of a scene's cells lie in such stretches; there the row loops read neither
/// map.
#[derive(Default)]
struct MediumRuns {
	runs: Vec<Run>,
	/// The first run of each grid row, then the number of runs.
	rows: Vec<usize>,
}

impl MediumRuns {
	/// The runs of the `gh` grid rows of `gw` cells of the padded maps `c2` and `d` (rows of `s`).
	fn new(c2: &[f32], d: &[f32], s: usize, gw: usize, gh: usize) -> Self {
		let mut out = Self { runs: Vec::new(), rows: Vec::with_capacity(gh + 1) };
		for gy in 0..gh {
			out.rows.push(out.runs.len());
			let at = (gy + PAD) * s + PAD;
			medium_runs(&c2[at..][..gw], &d[at..][..gw], MIN_RUN, &mut out.runs);
		}
		out.rows.push(out.runs.len());
		out
	}

	/// Hands `f` the runs of padded row `r` in order, each with the cut bytes of its cells when one of them is cut
	/// (`cuts` the row's, [`row_cuts`]).
	#[inline(always)]
	fn of_row(&self, r: usize, cuts: Option<&[u8]>, mut f: impl FnMut(Range<usize>, Option<Medium>, Option<&[u8]>)) {
		let gy = r - PAD;
		for run in &self.runs[self.rows[gy]..self.rows[gy + 1]] {
			let cuts = cuts.map(|c| &c[run.cells.clone()]).filter(|c| c.iter().any(|&c| c != 0));
			f(run.cells.clone(), run.medium, cuts);
		}
	}
}

/// Appends the runs of one row of `c2` and `d` to `out`: each maximal stretch of at least `min` cells with the same
/// bits of both with its medium, the cells between such stretches in runs without one.
fn medium_runs(c2: &[f32], d: &[f32], min: usize, out: &mut Vec<Run>) {
	let n = c2.len();
	let bits = |x: usize| (c2[x].to_bits(), d[x].to_bits());
	let mut mixed: Option<usize> = None;
	let mut start = 0;
	while start < n {
		let b = bits(start);
		let end = (start..n).find(|&x| bits(x) != b).unwrap_or(n);
		if end - start >= min {
			if let Some(m) = mixed.take() {
				out.push(Run { cells: m..start, medium: None });
			}
			let (c2, d) = (c2[start], d[start]);
			out.push(Run { cells: start..end, medium: Some(Medium { c2, d, centred: centred(d) }) });
		} else {
			mixed.get_or_insert(start);
		}
		start = end;
	}
	if let Some(m) = mixed {
		out.push(Run { cells: m..n, medium: None });
	}
}

/// The 9-point Laplacian at padded cell `i` (row length `s`) with the cut byte `cut`: a diagonal tap
/// is dropped when either axis neighbour it passes is a mirror, so no wave crosses a mirror line
/// that is only 8-connected; the rule is symmetric between the two cells of a tap. Same order of
/// operations as the uncut rows.
#[inline]
fn nine_point(u: &[f32], i: usize, s: usize, cut: u8) -> f32 {
	let v = u[i];
	let [nw, ne, sw, se] = pml::diagonal_cuts(cut);
	let tap = |k: usize, cut: bool| if cut { 0.0 } else { u[k] };
	let diagonals =
		((tap(i - s - 1, nw) - v) + (tap(i - s + 1, ne) - v)) + ((tap(i + s - 1, sw) - v) + (tap(i + s + 1, se) - v));
	let axes = ((u[i - s] - v) + (u[i + s] - v)) + ((u[i - 1] - v) + (u[i + 1] - v));
	K9_DIAGONAL * diagonals + K9_AXIS * axes
}

/// The 9-point Laplacian of the `len` cells from padded cell `at`, handed to `f` with the cell's offset from `at` and
/// its value. Cells without a cut run on row slices ([`nine_span`]), the cut ones through [`nine_point`].
#[inline(always)]
fn nine_row(u: &[f32], at: usize, s: usize, len: usize, cuts: Option<&[u8]>, mut f: impl FnMut(usize, f32, f32)) {
	let Some(cuts) = cuts else { return nine_span(u, at, s, 0..len, &mut f) };
	cut_spans(cuts, |span, cut| {
		if !cut {
			return nine_span(u, at, s, span, &mut f);
		}
		for x in span {
			f(x, u[at + x], nine_point(u, at + x, s, cuts[x]));
		}
	});
}

/// The 9-point Laplacian of the cells `span` from padded cell `at`, none of them cut, on row slices, as
/// [`nine_point`] in the same order. Every slice has the span's length, so the loop has no bounds checks.
#[inline(always)]
fn nine_span(u: &[f32], at: usize, s: usize, span: Range<usize>, f: &mut impl FnMut(usize, f32, f32)) {
	let x0 = span.start;
	let tap = |k: isize| &u[(at + x0).wrapping_add_signed(k)..][..span.len()];
	let si = s as isize;
	let (c, w, e, n, so) = (tap(0), tap(-1), tap(1), tap(-si), tap(si));
	let (nw, ne, sw, se) = (tap(-si - 1), tap(-si + 1), tap(si - 1), tap(si + 1));
	for x in 0..span.len() {
		let v = c[x];
		let diagonals = ((nw[x] - v) + (ne[x] - v)) + ((sw[x] - v) + (se[x] - v));
		let axes = ((n[x] - v) + (so[x] - v)) + ((w[x] - v) + (e[x] - v));
		f(x0 + x, v, K9_DIAGONAL * diagonals + K9_AXIS * axes);
	}
}

/// The DRP Laplacian at padded cell `i` (row length `s`), the taps beyond a mirror dropped (`cut`):
/// `sum over m of a_m sum over ring m of (u_k - u)`, the stencil since `a0 = -2 (a1 + a2 + a3)`.
#[inline]
fn drp(u: &[f32], i: usize, s: usize, a: &[f32; 4], cut: u8) -> f32 {
	let c = u[i];
	let (x, y) = (taps(u, i, 1, cut >> cut::X_SHIFT), taps(u, i, s, cut >> cut::Y_SHIFT));
	let ring = |m: usize| ((x[3 - m] - c) + (x[3 + m] - c)) + ((y[3 - m] - c) + (y[3 + m] - c));
	a[1] * ring(1) + a[2] * ring(2) + a[3] * ring(3)
}

/// The cut bytes of the `gw` grid cells from padded cell `at`, or None when the row has no mirror.
#[inline]
fn row_cuts(cuts: &[u8], mirrors: bool, at: usize, gw: usize) -> Option<&[u8]> {
	if !mirrors {
		return None;
	}
	let row = &cuts[at..][..gw];
	row.iter().any(|&c| c != 0).then_some(row)
}

/// Hands `f` the maximal runs of `cuts` in order, each with whether its cells drop a tap (a non-zero cut byte).
/// A row crossed by a mirror has only a few cut cells near it; the runs between take the row-slice loops.
#[inline(always)]
fn cut_spans(cuts: &[u8], mut f: impl FnMut(Range<usize>, bool)) {
	let mut start = 0;
	while start < cuts.len() {
		let cut = cuts[start] != 0;
		let len = cuts[start..].iter().position(|&c| (c != 0) != cut).unwrap_or(cuts.len() - start);
		f(start..start + len, cut);
		start += len;
	}
}

/// The DRP Laplacian of the `len` cells from padded cell `at`, handed to `f` with the cell's offset from `at`.
/// Cells without a cut run on row slices ([`drp_span`]), the cut ones through [`drp`].
#[inline(always)]
fn drp_row(
	u: &[f32],
	at: usize,
	s: usize,
	len: usize,
	a: &[f32; 4],
	cuts: Option<&[u8]>,
	mut f: impl FnMut(usize, f32),
) {
	let Some(cuts) = cuts else { return drp_span(u, at, s, 0..len, a, &mut f) };
	cut_spans(cuts, |span, cut| {
		if !cut {
			return drp_span(u, at, s, span, a, &mut f);
		}
		for x in span {
			f(x, drp(u, at + x, s, a, cuts[x]));
		}
	});
}

/// The DRP Laplacian of the cells `span` of the row from padded cell `at`, none of them cut, on row slices, as
/// [`drp`] in the same order.
#[inline(always)]
fn drp_span(u: &[f32], at: usize, s: usize, span: Range<usize>, a: &[f32; 4], f: &mut impl FnMut(usize, f32)) {
	let x0 = span.start;
	let tap = |k: isize| &u[(at + x0).wrapping_add_signed(k)..][..span.len()];
	let si = s as isize;
	let c = tap(0);
	let (w1, e1, n1, s1) = (tap(-1), tap(1), tap(-si), tap(si));
	let (w2, e2, n2, s2) = (tap(-2), tap(2), tap(-2 * si), tap(2 * si));
	let (w3, e3, n3, s3) = (tap(-3), tap(3), tap(-3 * si), tap(3 * si));
	for x in 0..span.len() {
		let v = c[x];
		let near = ((w1[x] - v) + (e1[x] - v)) + ((n1[x] - v) + (s1[x] - v));
		let mid = ((w2[x] - v) + (e2[x] - v)) + ((n2[x] - v) + (s2[x] - v));
		let far = ((w3[x] - v) + (e3[x] - v)) + ((n3[x] - v) + (s3[x] - v));
		f(x0 + x, a[1] * near + a[2] * mid + a[3] * far);
	}
}

/// Advances the intensity's two EMA stages of one row by the field `v`.
#[inline]
fn accumulate(s1: &mut [f32], s2: &mut [f32], v: &[f32], alpha: f32) {
	let rest = 1.0 - alpha;
	for ((a, b), &v) in s1.iter_mut().zip(s2).zip(v) {
		*a = *a * alpha + v * v * rest;
		*b = *b * alpha + *a * rest;
	}
}

impl WaveEngine for CpuEngine {
	fn name(&self) -> &'static str {
		"CPU"
	}

	fn scene(&self) -> &SceneData {
		&self.shared.scene
	}

	fn options(&self) -> &EngineOptions {
		&self.shared.options
	}

	fn time(&self) -> u32 {
		self.shared.time
	}

	fn sub_steps(&self) -> u32 {
		self.shared.sub_steps
	}

	fn reset(&mut self) {
		for b in [&mut self.u, &mut self.prev, &mut self.w, &mut self.s1, &mut self.s2] {
			b.fill(0.0);
		}
		if let Some(p) = &mut self.pml {
			p.reset();
		}
		self.shared.reset();
		self.probe_queue.clear();
		self.captured = None;
	}

	fn set_track_intensity(&mut self, on: bool) {
		if self.shared.options.track_intensity != on {
			self.shared.options.track_intensity = on;
			let stages = if on { self.u.len() } else { 0 };
			self.s1 = vec![0.0; stages];
			self.s2 = vec![0.0; stages];
			self.captured = None;
		}
	}

	fn set_emitters(&mut self, emitters: &[Emitter]) {
		self.shared.set_emitters(emitters);
		self.group_sources();
		self.update_alpha();
	}

	fn set_probes(&mut self, probes: &[ProbeCells]) {
		let scene = &self.shared.scene;
		self.probes = probes
			.iter()
			.map(|p| (p.layer, p.cells.iter().map(|&(x, y)| scene.canvas_index(x as usize, y as usize)).collect()))
			.collect();
		self.probe_queue.set_lengths(probes.iter().map(|p| p.cells.len()).collect());
	}

	fn try_replace_scene(&mut self, scene: &Arc<SceneData>) -> bool {
		let Some(n) = self.shared.replacement(scene) else { return false };
		self.shared.replace_scene(scene.clone(), n);
		self.load_scene();
		self.captured = None;
		self.update_alpha();
		true
	}

	fn advance_one_step(&mut self) -> Result<(), EngineError> {
		CpuEngine::advance_one_step(self);
		Ok(())
	}

	fn sample_probes(&mut self, out: &mut ProbeReadings) -> Result<(), EngineError> {
		CpuEngine::sample_probes(self, out);
		Ok(())
	}

	fn queue_probes(&mut self) -> Result<(), EngineError> {
		let mut reading = std::mem::take(&mut self.probe_reading);
		CpuEngine::sample_probes(self, &mut reading);
		self.probe_queue.push(reading.values());
		self.probe_reading = reading;
		Ok(())
	}

	fn pending_probes(&mut self) -> Result<PendingProbes, EngineError> {
		Ok(PendingProbes::done(self.probe_queue.end()))
	}

	fn drain_probes(&mut self, pending: PendingProbes, f: &mut dyn FnMut(&ProbeReadings)) -> Result<(), EngineError> {
		self.probe_queue.drain(pending.end(), f);
		Ok(())
	}

	fn plan_capture(&mut self, shape: &FrameShape) {
		self.planned = Some(*shape);
	}

	/// The planes the last step captured when they have `shape` (`into` kept for the next), else a conversion now.
	fn begin_capture(&mut self, shape: &FrameShape, mut into: Snapshot) -> Result<PendingCapture, EngineError> {
		match self.captured.take() {
			Some(captured) if captured.shape == *shape => {
				self.planes = Some(into);
				Ok(PendingCapture::done(captured))
			}
			other => {
				if other.is_some() {
					self.planes = other;
				}
				self.capture_into(shape, &mut into);
				Ok(PendingCapture::done(into))
			}
		}
	}

	fn finish_capture(&mut self, pending: PendingCapture) -> Result<Snapshot, EngineError> {
		Ok(pending.snapshot)
	}

	fn prepare_frame(&mut self) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
		Ok(CpuEngine::prepare_frame(self))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::model::{Edges, PulseShape, Scene, Stencil};

	fn options(stencil: Stencil) -> EngineOptions {
		EngineOptions {
			stencil,
			source_frequency_scale: 2.0,
			image_source_scale: 1.0,
			time_decay: 10.0,
			intensity_scale: 1.0,
			track_intensity: true,
		}
	}

	fn emitter(x: i32, y: i32) -> Emitter {
		Emitter {
			x: f64::from(x),
			y: f64::from(y),
			level: 100.0,
			amplitude: 1.0,
			phase_radians: 0.0,
			pulsed: false,
			shape: PulseShape::Gaussian,
			on_width_cycles: 10,
			off_width_cycles: 10,
			off_width_base: 10,
			randomize_off_width: 0,
			detach_after_step: None,
			rng_seed: std::num::NonZeroU32::MIN,
		}
	}

	fn engine(edges: Edges, rim: i32, stencil: Stencil) -> CpuEngine {
		let scene = SceneData::blank(40, 30, &Scene { damping_edge: rim, edges, ..Scene::default() });
		let mut e = CpuEngine::new(scene, options(stencil));
		e.set_emitters(&[emitter(20, 15), emitter(-1, 3), emitter(40, 0)]);
		e
	}

	#[test]
	fn a_cpu_capture_is_ready_at_once() {
		use std::future::Future;
		use std::task::{Context, Poll, Waker};
		let mut e = engine(Edges::Open, 0, Stencil::NinePoint);
		e.advance_one_step();
		let shape = FrameShape {
			canvas_width: 40,
			canvas_height: 30,
			divisor: 1,
			format: crate::engine::SampleFormat::Half,
			field_brightness: 1.0,
			intensity_brightness: 1.0,
			intensity_root: false,
		};
		let pending = e.begin_capture(&shape, Snapshot::new(shape)).unwrap();
		let mut ready = std::pin::pin!(pending.ready());
		assert_eq!(ready.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(()));
		assert_eq!(e.finish_capture(pending).unwrap().shape, shape);
	}

	#[test]
	fn outside_emitters_are_dropped() {
		assert_eq!(engine(Edges::Open, 0, Stencil::NinePoint).shared.sources.emitters.len(), 1);
	}

	#[test]
	fn reflective_ring_stays_zero_and_intensity_accumulates() {
		let mut e = engine(Edges::Reflective, 0, Stencil::FourthOrder);
		for _ in 0..80 {
			e.advance_one_step();
		}
		let (s, rows) = (e.stride, e.u.len() / e.stride);
		let padding = (0..rows * s).filter(|&i| {
			let (x, y) = (i % s, i / s);
			x < PAD || y < PAD || x >= s - PAD || y >= rows - PAD
		});
		assert!(padding.into_iter().all(|i| e.u[i] == 0.0 && e.prev[i] == 0.0));
		let (field, intensity) = e.prepare_frame();
		assert!(field.iter().any(|&v| v != 0.0));
		assert!(intensity.iter().all(|&v| v >= 0.0) && intensity.iter().any(|&v| v > 0.0));
		assert_eq!(e.time(), 80);
	}

	#[test]
	fn reset_and_replay_are_identical() {
		let mut e = engine(Edges::Open, -8, Stencil::NinePoint);
		let run = |e: &mut CpuEngine| {
			for _ in 0..50 {
				e.advance_one_step();
			}
			e.prepare_frame()
		};
		let a = run(&mut e);
		e.reset();
		assert_eq!(a, run(&mut e));
	}

	#[test]
	fn probes_read_the_field_and_the_intensity() {
		let mut e = engine(Edges::Open, -4, Stencil::FourthOrder);
		e.set_probes(&[
			ProbeCells { layer: Layer::Field, cells: vec![(22, 15)] },
			ProbeCells { layer: Layer::Intensity, cells: vec![(22, 15), (22, 15)] },
		]);
		for _ in 0..30 {
			e.advance_one_step();
		}
		let mut r = ProbeReadings::default();
		e.sample_probes(&mut r);
		let (field, intensity) = e.prepare_frame();
		assert_eq!(r.probe(0), [field[15 * 40 + 22]]);
		assert_eq!(r.probe(1), [intensity[15 * 40 + 22]; 2]);
	}

	#[test]
	fn pending_probes_are_ready_at_once_and_drain_in_step_order() {
		use std::future::Future;
		use std::task::{Context, Poll, Waker};
		let mut e = engine(Edges::Open, -4, Stencil::FourthOrder);
		e.set_probes(&[
			ProbeCells { layer: Layer::Field, cells: vec![(22, 15)] },
			ProbeCells { layer: Layer::Intensity, cells: vec![(10, 5), (11, 5), (12, 5)] },
		]);
		let (mut sampled, mut drained, mut r) = (Vec::new(), Vec::new(), ProbeReadings::default());
		let mut held = None;
		for step in 1..=40 {
			e.advance_one_step();
			e.sample_probes(&mut r);
			sampled.push(r.clone());
			e.queue_probes().unwrap();
			if step % 15 == 0 || step == 40 {
				if let Some((pending, covered)) = held.take() {
					e.drain_probes(pending, &mut |r| drained.push(r.clone())).unwrap();
					assert_eq!(drained.len(), covered, "the readings queued after the token stay");
				}
				let pending = e.pending_probes().unwrap();
				let mut ready = std::pin::pin!(pending.ready());
				assert_eq!(ready.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(()));
				held = Some((pending, step));
			}
		}
		let (pending, _) = held.unwrap();
		e.drain_probes(pending, &mut |r| drained.push(r.clone())).unwrap();
		assert_eq!(drained, sampled);
	}

	#[test]
	fn a_token_taken_before_a_reset_drains_nothing_after_it() {
		let mut e = engine(Edges::Open, -4, Stencil::FourthOrder);
		e.set_probes(&[ProbeCells { layer: Layer::Field, cells: vec![(22, 15)] }]);
		for _ in 0..10 {
			e.advance_one_step();
			e.queue_probes().unwrap();
		}
		let stale = e.pending_probes().unwrap();
		e.reset();
		for _ in 0..5 {
			e.advance_one_step();
			e.queue_probes().unwrap();
		}
		e.drain_probes(stale, &mut |_| panic!("a reading from before the reset")).unwrap();
		let (pending, mut drained) = (e.pending_probes().unwrap(), 0);
		e.drain_probes(pending, &mut |_| drained += 1).unwrap();
		assert_eq!(drained, 5);
	}

	#[test]
	fn capture_holds_the_canvas() {
		let mut e = engine(Edges::Open, -4, Stencil::FourthOrder);
		for _ in 0..30 {
			e.advance_one_step();
		}
		let shape = FrameShape {
			canvas_width: 40,
			canvas_height: 30,
			divisor: 1,
			format: crate::engine::SampleFormat::Half,
			field_brightness: 1.0,
			intensity_brightness: 1.0,
			intensity_root: true,
		};
		let snap = e.capture_frame(&shape);
		let (field, intensity) = e.prepare_frame();
		assert_eq!(snap.step, 29);
		let h = snap.halves(Layer::Field);
		assert!(h.iter().zip(&field).all(|(a, &b)| *a == half::f16::from_f32(b)));
		let i = snap.halves(Layer::Intensity);
		assert!(i.iter().zip(&intensity).all(|(a, &b)| *a == half::f16::from_f32(b.sqrt())));
	}

	#[test]
	fn replace_refuses_another_grid() {
		let mut e = engine(Edges::Open, -4, Stencil::FourthOrder);
		let other = SceneData::blank(41, 30, &Scene { damping_edge: -4, ..Scene::default() });
		assert!(!e.try_replace_scene(&Arc::new(other)));
		let reflective =
			SceneData::blank(40, 30, &Scene { damping_edge: -4, edges: Edges::Reflective, ..Scene::default() });
		assert!(!e.try_replace_scene(&Arc::new(reflective)));
		let slower = SceneData::blank(40, 30, &Scene { damping_edge: -4, refractive_index: 2.0, ..Scene::default() });
		assert!(e.try_replace_scene(&Arc::new(slower)));
		assert_eq!(e.sub_steps(), 1);
	}

	#[test]
	fn alpha_follows_the_lowest_level() {
		let mut e = engine(Edges::Open, 0, Stencil::NinePoint);
		assert_eq!(e.alpha, (-1.0f64 / 10.0).exp() as f32);
		let mut low = emitter(1, 1);
		low.level = 50.0;
		e.set_emitters(&[low]);
		assert_eq!(e.alpha, (-1.0f64 / 20.0).exp() as f32);
		e.set_emitters(&[]);
		assert_eq!(e.alpha, (-1.0f64 / 10.0).exp() as f32);
		e.shared.sources.level_used[0] = true;
		e.update_alpha();
		assert_eq!(e.alpha, (-1.0f64 / 10.0).exp() as f32, "level 0 is not a source level");
	}

	#[test]
	fn cut_spans_cover_the_row_in_maximal_runs() {
		let mut runs = Vec::new();
		cut_spans(&[0, 0, 3, 1, 0, 0, 0, 8], |span, cut| runs.push((span, cut)));
		assert_eq!(runs, [(0..2, false), (2..4, true), (4..7, false), (7..8, true)]);
		runs.clear();
		cut_spans(&[0; 5], |span, cut| runs.push((span, cut)));
		assert_eq!(runs, [(0..5, false)]);
	}

	#[test]
	fn a_row_with_cuts_matches_drp_per_cell() {
		let (s, gw) = (20, 14);
		let u: Vec<f32> = (0..s * 10).map(|i| ((i * 7919 % 101) as f32 - 50.0) * 0.013).collect();
		let at = 4 * s + PAD;
		let mut cuts = vec![0u8; gw];
		(cuts[3], cuts[4], cuts[10]) = (0b0001_0010, 0b1000_0000, 0b0000_0101);
		let a = [0.0, 1.3, -0.17, 0.021];
		let mut got = vec![f32::NAN; gw];
		drp_row(&u, at, s, gw, &a, Some(&cuts), |x, v| got[x] = v);
		let want: Vec<f32> = (0..gw).map(|x| drp(&u, at + x, s, &a, cuts[x])).collect();
		assert_eq!(
			got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
			want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
		);
	}

	#[test]
	fn medium_runs_cover_the_row_and_keep_the_long_stretches() {
		let c2 = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 5.0];
		let d = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.0];
		let medium = |c2: f32, d: f32| Some(Medium { c2, d, centred: centred(d) });
		let mut runs = Vec::new();
		medium_runs(&c2, &d, 3, &mut runs);
		let want = [
			Run { cells: 0..3, medium: medium(1.0, 0.5) },
			Run { cells: 3..5, medium: None },
			Run { cells: 5..9, medium: medium(3.0, 0.5) },
			Run { cells: 9..13, medium: None },
		];
		assert_eq!(runs, want);
		runs.clear();
		medium_runs(&[0.0; 4], &[1.0, 1.0, 1.0, 1.0], 8, &mut runs);
		assert_eq!(runs, [Run { cells: 0..4, medium: None }]);
		runs.clear();
		medium_runs(&[0.0; 4], &[1.0; 4], 4, &mut runs);
		assert_eq!(runs, [Run { cells: 0..4, medium: medium(0.0, 1.0) }]);
	}

	#[test]
	fn a_row_with_cuts_matches_nine_point_per_cell() {
		let (s, gw) = (20, 14);
		let u: Vec<f32> = (0..s * 10).map(|i| ((i * 7919 % 101) as f32 - 50.0) * 0.013).collect();
		let at = 4 * s + PAD;
		let mut cuts = vec![0u8; gw];
		(cuts[3], cuts[4], cuts[10]) = (0b0001_0010, 0b1000_0000, 0b0000_0101);
		let mut got = vec![(f32::NAN, f32::NAN); gw];
		nine_row(&u, at, s, gw, Some(&cuts), |x, v, lap| got[x] = (v, lap));
		let bits = |(v, lap): (f32, f32)| (v.to_bits(), lap.to_bits());
		let want: Vec<_> = (0..gw).map(|x| bits((u[at + x], nine_point(&u, at + x, s, cuts[x])))).collect();
		assert_eq!(got.into_iter().map(bits).collect::<Vec<_>>(), want);
	}

	#[test]
	fn group_rows_list_each_row_s_groups() {
		let (s, rows) = (10, 6);
		let cells = [52, 13, 57, 11, 30];
		let g = GroupRows::new(&cells, s, rows);
		let expect: [&[u32]; 6] = [&[], &[3, 1], &[], &[4], &[], &[0, 2]];
		for (r, e) in expect.iter().enumerate() {
			assert_eq!(g.of(r), *e, "row {r}");
		}
	}

	/// Captures planned before their steps, made by the passes as they write the canvas, equal those converted after
	/// the steps byte for byte, and the state stays the same: odd canvas sizes, rims that put block rows across the
	/// bands, both stencils, every divisor and format, a mirror, emitters and an image source on the canvas's edges.
	#[test]
	fn a_planned_capture_equals_a_capture_after_the_step() {
		use crate::engine::{ImageSource, SampleFormat};
		let shape = |divisor, format, intensity_root| FrameShape {
			canvas_width: 37,
			canvas_height: 75,
			divisor,
			format,
			field_brightness: 40.0,
			intensity_brightness: 300.0,
			intensity_root,
		};
		let shapes = [
			shape(1, SampleFormat::Half, true),
			shape(2, SampleFormat::Index, true),
			shape(4, SampleFormat::Half, false),
			shape(4, SampleFormat::Index, false),
			shape(2, SampleFormat::Half, true),
			shape(1, SampleFormat::Index, true),
		];
		let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
		for stencil in [Stencil::NinePoint, Stencil::FourthOrder] {
			for (rim, tracked) in [(0, true), (1, true), (2, false), (3, true), (5, true), (6, true)] {
				let mut scene = SceneData::blank(37, 75, &Scene { damping_edge: -rim, ..Scene::default() });
				let gw = scene.grid_width() as usize;
				let r = rim as usize;
				for x in 5..20 {
					scene.c2[(r + 40) * gw + r + x] = 0.0;
				}
				scene.image_sources.push(ImageSource { gx: r as u32 + 36, gy: r as u32 + 33, level: 120 });
				let build = || {
					let mut e = CpuEngine::new(scene.clone(), options(stencil));
					e.set_emitters(&[emitter(20, 15), emitter(3, 31), emitter(36, 74), emitter(0, 0), emitter(8, 63)]);
					e.set_track_intensity(tracked);
					e
				};
				let (mut fused, mut after) = (build(), build());
				for step in 0..14 {
					let shape = shapes[step % shapes.len()];
					if step != 9 {
						fused.plan_capture(&shape);
					}
					fused.advance_one_step();
					after.advance_one_step();
					assert_eq!(fused.captured.is_some(), step != 9);
					if step == 11 {
						continue;
					}
					let ask = if step == 7 { shapes[0] } else { shape };
					let pending = fused.begin_capture(&ask, Snapshot::new(ask)).unwrap();
					let got = fused.finish_capture(pending).unwrap();
					let want = after.capture_frame(&ask);
					let at = format!("{stencil:?} rim {rim} step {step}");
					assert_eq!((got.shape, got.step), (want.shape, want.step), "{at}");
					assert!(got.field == want.field, "field, {at}");
					assert!(got.intensity == want.intensity, "intensity, {at}");
					assert_eq!(got.intensity.is_empty(), !tracked);
					for (a, b) in [
						(&fused.u, &after.u),
						(&fused.prev, &after.prev),
						(&fused.s1, &after.s1),
						(&fused.s2, &after.s2),
					] {
						assert!(bits(a) == bits(b), "state, {at}");
					}
				}
			}
		}
	}

	#[test]
	fn band_lengths_cover_the_grid_and_start_on_canvas_rows_of_four() {
		for (gh, rim) in [(75, 0), (77, 1), (81, 3), (139, 32), (5, 2), (36, 0), (130, 33)] {
			for band in [4, 32] {
				let lens: Vec<usize> = band_lengths(gh, rim, band).collect();
				assert_eq!(lens.iter().sum::<usize>(), gh);
				assert!(lens.iter().all(|&n| n > 0));
				let mut g = 0;
				for &n in &lens[..lens.len() - 1] {
					g += n;
					assert!(g < rim || (g - rim) % 4 == 0, "gh {gh} rim {rim} band {band} at {g}");
				}
			}
		}
	}

	#[test]
	fn rim_w_copies_the_cells_within_reach_of_the_rim() {
		let (gw, gh, rim) = (20, 16, 2);
		let s = gw + 2 * PAD;
		let from: Vec<f32> = (1..=s).map(|v| v as f32).collect();
		for gy in 0..gh {
			let mut to = vec![0.0; s];
			rim_w(&from, &mut to, gy, gw, gh, rim);
			for (i, (&t, &f)) in to.iter().zip(&from).enumerate() {
				let whole = gy < rim + 3 || gy >= gh - rim - 3;
				let read = whole || (PAD..PAD + gw).contains(&i) && (i - PAD < rim + 3 || i - PAD >= gw - rim - 3);
				assert!(if read { t == f } else { t == 0.0 }, "row {gy} cell {i}");
			}
		}
	}
}
