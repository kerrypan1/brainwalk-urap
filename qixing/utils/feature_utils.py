# Feature names here are the original ones. docs/feature_renames.csv maps them to the current names.
from typing import Dict, Tuple, Optional
import numpy as np
from scipy import signal
from scipy.stats import entropy

"""
Pose-only utilities and metrics. Audio-related helpers were removed to align
with the visual-only scope requested by the advisor.
"""


# Video filename parsing utilities

def parse_video_type(filename: str, max_label_len: int = 5) -> Tuple[Optional[str], Optional[int]]:
	"""Parse underscore-separated filename and return (label, index).

	Prefer purely alphabetic tokens (length <= max_label_len). If none found, fall back
	to alphanumeric tokens excluding long numeric tokens (likely timestamps).
	"""
	import os
	import re
	if not filename:
		return None, None
	base = os.path.basename(filename)
	base_noext = os.path.splitext(base)[0]
	parts = base_noext.split('_')

	# First prefer alphabetic tokens
	for i, p in enumerate(parts, start=1):
		p_str = p.strip()
		if 0 < len(p_str) <= int(max_label_len) and p_str.isalpha():
			return p_str.upper(), i

	# Fallback: short alphanumeric tokens but skip long digit-only tokens
	pattern = re.compile(r'^[A-Za-z0-9]{1,' + str(int(max_label_len)) + r'}$')
	for i, p in enumerate(parts, start=1):
		p_str = p.strip()
		if pattern.match(p_str):
			if p_str.isdigit() and len(p_str) > 2:
				continue
			return p_str.upper(), i
	return None, None


def _video_type_map_path() -> str:
	"""Return default path for the persisted video type mapping file inside src/."""
	import os
	return os.path.join(os.path.dirname(__file__), 'video_type_map.json')


def load_video_type_map(path: Optional[str] = None) -> Dict[str, int]:
	"""Load label->int mapping from JSON file. If missing returns empty dict."""
	import json, os
	p = path or _video_type_map_path()
	if not os.path.exists(p):
		return {}
	try:
		with open(p, 'r', encoding='utf-8') as f:
			data = json.load(f)
		# ensure keys are strings, values ints
		return {str(k): int(v) for k, v in data.items()}
	except Exception:
		return {}


def save_video_type_map(mapping: Dict[str, int], path: Optional[str] = None) -> None:
	"""Atomically save mapping to JSON file."""
	import json, os, tempfile
	p = path or _video_type_map_path()
	d = os.path.dirname(p)
	if d and not os.path.exists(d):
		os.makedirs(d, exist_ok=True)
	# atomic write
	fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_vtype_', text=True)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as f:
			json.dump(mapping, f, ensure_ascii=False, indent=2)
		os.replace(tmp, p)
	finally:
		if os.path.exists(tmp):
			try:
				os.remove(tmp)
			except Exception:
				pass


def encode_video_type_from_filename(filename: str, mapping_path: Optional[str] = None, method: str = 'both') -> Dict[str, object]:
	"""Parse filename to label and return encoded features; persist mapping if new label seen.

	Returns a dict with keys: 'video_type', 'video_type_code' and optionally one-hot keys like 'video_type_FW'.
	Unknown labels map to code 0. Persistent mapping saved at mapping_path or default inside src/.
	"""
	label, idx = parse_video_type(filename)
	mapping = load_video_type_map(mapping_path)
	if label is None:
		code = 0
	else:
		# canonicalize
		lab = label.upper()
		if lab in mapping:
			code = int(mapping[lab])
		else:
			# assign next integer code (start at 1)
			next_code = max(mapping.values()) + 1 if mapping else 1
			mapping[lab] = int(next_code)
			try:
				save_video_type_map(mapping, mapping_path)
			except Exception:
				# ignore persistence errors but keep in-memory mapping
				pass
			code = int(mapping[lab])

	out: Dict[str, object] = {}
	out['video_type'] = label
	out['video_type_code'] = int(code)
	if method in ('onehot', 'both') and label is not None:
		out[f'video_type_{label}'] = 1.0
	return out



#Pose
#A Symmetry / Side-to-Side Asymmetry features computations


def stride_time_cv(stride_times: np.ndarray) -> float:
	"""Coefficient of variation of stride durations: std / mean.

	Returns 0.0 when mean is zero or input is empty.
	"""
	arr = np.asarray(stride_times)
	if arr.size == 0:
		return 0.0
	mean = float(np.mean(arr))
	if mean == 0:
		return 0.0
	return float(np.std(arr) / mean)


def stride_width_mean(left_x: np.ndarray, right_x: np.ndarray) -> float:
	"""Mean lateral distance between left and right ankle x-coordinates.

	left_x and right_x should be same-length sequences sampled over time.
	"""
	L = np.asarray(left_x)
	R = np.asarray(right_x)
	if L.size == 0 or R.size == 0:
		return 0.0
	# broadcast to same shape if needed by truncation
	n = min(L.size, R.size)
	return float(np.mean(np.abs(L[:n] - R[:n])))


def shoulder_height_asymmetry(left_shoulder_y: np.ndarray, right_shoulder_y: np.ndarray) -> float:
	"""Relative shoulder height asymmetry: |meanL - meanR| / ((meanL + meanR)/2).

	Returns 0.0 if denominator is zero or inputs empty.
	"""
	L = np.asarray(left_shoulder_y)
	R = np.asarray(right_shoulder_y)
	if L.size == 0 or R.size == 0:
		return 0.0
	meanL = float(np.mean(L))
	meanR = float(np.mean(R))
	denom = (meanL + meanR) / 2.0
	if denom == 0:
		return 0.0
	return abs(meanL - meanR) / denom


def arm_swing_amplitude_asymmetry(left_wrist_x: np.ndarray, left_shoulder_x: np.ndarray,
								  right_wrist_x: np.ndarray, right_shoulder_x: np.ndarray) -> float:
	"""Asymmetry of lateral arm swing amplitude between sides.

	Computes mean(|lw - ls|) and mean(|rw - rs|) then returns
	abs(left_mean - right_mean) / mean_shoulder_separation.
	If shoulder separation is zero or inputs empty returns 0.0.
	"""
	LW = np.asarray(left_wrist_x)
	LS = np.asarray(left_shoulder_x)
	RW = np.asarray(right_wrist_x)
	RS = np.asarray(right_shoulder_x)
	if LW.size == 0 or LS.size == 0 or RW.size == 0 or RS.size == 0:
		return 0.0
	nL = min(LW.size, LS.size)
	nR = min(RW.size, RS.size)
	left_amp = float(np.mean(np.abs(LW[:nL] - LS[:nL])))
	right_amp = float(np.mean(np.abs(RW[:nR] - RS[:nR])))
	# shoulder separation: mean absolute lateral distance between shoulders
	# use overlapping length
	ns = min(LS.size, RS.size)
	if ns == 0:
		return 0.0
	shoulder_sep = float(np.mean(np.abs(LS[:ns] - RS[:ns])))
	if shoulder_sep == 0:
		return 0.0
	return abs(left_amp - right_amp) / shoulder_sep


def leg_step_length_asymmetry(left_steps: np.ndarray, right_steps: np.ndarray) -> float:
	"""Mean step length asymmetry: |meanL - meanR| / ((meanL + meanR)/2).

	left_steps and right_steps are arrays of individual step lengths.
	"""
	L = np.asarray(left_steps)
	R = np.asarray(right_steps)
	if L.size == 0 or R.size == 0:
		return 0.0
	meanL = float(np.mean(L))
	meanR = float(np.mean(R))
	denom = (meanL + meanR) / 2.0
	if denom == 0:
		return 0.0
	return abs(meanL - meanR) / denom


def temporal_gait_asymmetry(stance_durations_left: np.ndarray, stance_durations_right: np.ndarray) -> float:
	"""Asymmetry of stance phase duration between left and right.

	Formula: |Tstance_L - Tstance_R| / ((Tstance_L + Tstance_R)/2) where each T is mean over strides.
	"""
	L = np.asarray(stance_durations_left)
	R = np.asarray(stance_durations_right)
	if L.size == 0 or R.size == 0:
		return 0.0
	meanL = float(np.mean(L))
	meanR = float(np.mean(R))
	denom = (meanL + meanR) / 2.0
	if denom == 0:
		return 0.0
	return abs(meanL - meanR) / denom


def peak_velocity_asymmetry(v_left_peaks: np.ndarray, v_right_peaks: np.ndarray) -> float:
	"""Asymmetry of peak wrist velocities: |vLmax - vRmax| / ((vLmax + vRmax)/2).

	Returns 0.0 when both peaks are zero or arrays empty.
	"""
	L = np.asarray(v_left_peaks)
	R = np.asarray(v_right_peaks)
	if L.size == 0 or R.size == 0:
		return 0.0
	vL = float(np.max(L))
	vR = float(np.max(R))
	denom = (vL + vR) / 2.0
	if denom == 0:
		return 0.0
	return abs(vL - vR) / denom


def symmetry_index_composite(left_vals: np.ndarray, right_vals: np.ndarray) -> float:
	"""Composite symmetry index over multiple paired metrics.

	Computes mean_k |Lk - Rk| / ((Lk + Rk)/2) across k. Pairs where denom==0 are skipped.
	Returns 0.0 if no valid pairs.
	"""
	L = np.asarray(left_vals)
	R = np.asarray(right_vals)
	n = min(L.size, R.size)
	if n == 0:
		return 0.0
	L = L[:n]
	R = R[:n]
	denom = (L + R) / 2.0
	valid = denom != 0
	if not np.any(valid):
		return 0.0
	vals = np.abs(L[valid] - R[valid]) / denom[valid]
	return float(np.mean(vals))

#B Tremor-Specific Features


def wrist_tremor_amplitude(wrist_y: np.ndarray) -> float:
	"""Frame-to-frame variability of wrist displacement: std(diff(y))."""
	y = np.asarray(wrist_y)
	if y.size < 2:
		return 0.0
	return float(np.std(np.diff(y)))


def tremor_dominant_frequency(wrist_y: np.ndarray, sr: float) -> float:
	"""Return dominant frequency (Hz) from FFT of wrist displacement."""
	y = np.asarray(wrist_y)
	if y.size < 2 or sr <= 0:
		return 0.0
	n = y.size
	Y = np.fft.rfft(y - np.mean(y))
	freqs = np.fft.rfftfreq(n, d=1.0 / sr)
	power = np.abs(Y) ** 2
	if power.size == 0:
		return 0.0
	idx = int(np.argmax(power))
	return float(freqs[idx])


def tremor_power_band(wrist_y: np.ndarray, sr: float, band: tuple = (3.0, 7.0)) -> float:
	"""Normalized spectral power in given band (e.g., 3-7 Hz)."""
	y = np.asarray(wrist_y)
	if y.size < 2 or sr <= 0:
		return 0.0
	n = y.size
	Y = np.fft.rfft(y - np.mean(y))
	freqs = np.fft.rfftfreq(n, d=1.0 / sr)
	power = np.abs(Y) ** 2
	total = np.sum(power)
	if total == 0:
		return 0.0
	band_mask = (freqs >= band[0]) & (freqs <= band[1])
	band_power = np.sum(power[band_mask])
	return float(band_power / total)


def tremor_intermittency(wrist_y: np.ndarray, sr: float, window_sec: float = 1.0) -> float:
	"""Fraction of windows where amplitude > mean+2*std across windows.

	Window amplitude measured as RMS within window.
	"""
	y = np.asarray(wrist_y)
	if y.size == 0 or sr <= 0 or window_sec <= 0:
		return 0.0
	win_len = max(1, int(window_sec * sr))
	# split into non-overlapping windows
	n_windows = y.size // win_len
	if n_windows == 0:
		return 0.0
	amps = []
	for i in range(n_windows):
		seg = y[i * win_len:(i + 1) * win_len]
		amps.append(np.sqrt(np.mean((seg - np.mean(seg)) ** 2)))
	amps = np.asarray(amps)
	mu = float(np.mean(amps))
	sigma = float(np.std(amps))
	if sigma == 0:
		return 0.0
	thresh = mu + 2 * sigma
	return float(np.sum(amps > thresh) / amps.size)


def tremor_spectral_entropy(wrist_y: np.ndarray, sr: float) -> float:
	"""Spectral entropy of wrist displacement spectrum. Lower = more periodic."""
	y = np.asarray(wrist_y)
	if y.size < 2 or sr <= 0:
		return 0.0
	Y = np.fft.rfft(y - np.mean(y))
	power = np.abs(Y) ** 2
	total = np.sum(power)
	if total == 0:
		return 0.0
	p = power / total
	# avoid log(0)
	p = p[p > 0]
	return float(-np.sum(p * np.log(p)))


def _magnitude_squared_coherence(x: np.ndarray, y: np.ndarray, fs: float,
								 nperseg: int = 256, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute magnitude-squared coherence via Welch's method (NumPy-only).

	Returns (freqs, Cxy) where Cxy is in [0,1]. If not enough data for at least one segment,
	returns (empty, empty).
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	n = min(x.size, y.size)
	if n == 0 or fs <= 0:
		return np.array([]), np.array([])
	x = x[:n]
	y = y[:n]
	if nperseg is None:
		nperseg = min(256, n)
	nperseg = int(max(16, min(n, nperseg)))  # keep reasonable bounds
	if noverlap is None:
		noverlap = nperseg // 2
	step = nperseg - int(noverlap)
	if step <= 0:
		step = max(1, nperseg // 2)
	starts = np.arange(0, n - nperseg + 1, step)
	if starts.size == 0:
		return np.array([]), np.array([])
	window = np.hanning(nperseg)
	# Scale factor cancels in coherence ratio, so we can omit exact density normalization
	Sxx = None
	Syy = None
	Sxy = None
	for s in starts:
		segx = x[s:s + nperseg]
		segy = y[s:s + nperseg]
		# remove segment mean to suppress DC
		segx = (segx - float(np.mean(segx))) * window
		segy = (segy - float(np.mean(segy))) * window
		X = np.fft.rfft(segx)
		Y = np.fft.rfft(segy)
		Px = (np.abs(X) ** 2)
		Py = (np.abs(Y) ** 2)
		Pxy = X * np.conjugate(Y)
		if Sxx is None:
			Sxx = Px
			Syy = Py
			Sxy = Pxy
		else:
			Sxx = Sxx + Px
			Syy = Syy + Py
			Sxy = Sxy + Pxy
	# average across segments
	Sxx = Sxx / starts.size
	Syy = Syy / starts.size
	Sxy = Sxy / starts.size
	freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
	denom = Sxx * Syy
	coh = np.zeros_like(denom, dtype=float)
	m = denom > 0
	coh[m] = (np.abs(Sxy[m]) ** 2) / denom[m]
	# numeric guard
	coh = np.clip(coh.real, 0.0, 1.0)
	return freqs, coh


def cross_wrist_coherence(left_y: np.ndarray, right_y: np.ndarray, sr: float, band: tuple = (3.0, 7.0)) -> float:
	"""Average magnitude-squared coherence between two wrist signals over band.

	Uses Welch-based estimate to avoid the trivial single-FFT identity that yields 1.0.
	"""
	x = np.asarray(left_y)
	y = np.asarray(right_y)
	if x.size < 2 or y.size < 2 or sr <= 0:
		return 0.0
	n = min(x.size, y.size)
	x = x[:n]
	y = y[:n]
	# choose segment length relative to sampling rate and data length
	# aim ~2s windows if possible
	target = int(max(32, min(n, round(sr * 2))))
	freqs, coh = _magnitude_squared_coherence(x, y, fs=sr, nperseg=target, noverlap=target // 2)
	if freqs.size == 0 or coh.size == 0:
		return 0.0
	band_mask = (freqs >= float(band[0])) & (freqs <= float(band[1]))
	if not np.any(band_mask):
		return 0.0
	return float(np.mean(coh[band_mask]))


def high_frequency_jerk_metric(wrist_y: np.ndarray) -> float:
	"""Standard deviation of second differences (jerk) to capture rapid oscillations."""
	y = np.asarray(wrist_y)
	if y.size < 3:
		return 0.0
	jerk2 = np.diff(y, n=2)
	return float(np.std(jerk2))


def phase_locking_value(signal1: np.ndarray, signal2: np.ndarray) -> float:
	"""Phase-locking value (PLV) between two signals using analytic signal via FFT.

	PLV = |mean(exp(i*(phi1 - phi2)))|
	"""
	def analytic_signal(sig: np.ndarray) -> np.ndarray:
		sig = np.asarray(sig)
		n = sig.size
		if n == 0:
			return np.array([])
		S = np.fft.fft(sig)
		H = np.zeros(n)
		if n % 2 == 0:
			H[0] = 1
			H[1:n//2] = 2
			H[n//2] = 1
		else:
			H[0] = 1
			H[1:(n+1)//2] = 2
		analytic = np.fft.ifft(S * H)
		return analytic

	s1 = np.asarray(signal1)
	s2 = np.asarray(signal2)
	n = min(s1.size, s2.size)
	if n == 0:
		return 0.0
	a1 = analytic_signal(s1[:n])
	a2 = analytic_signal(s2[:n])
	if a1.size == 0 or a2.size == 0:
		return 0.0
	phi1 = np.angle(a1)
	phi2 = np.angle(a2)
	plv = np.abs(np.mean(np.exp(1j * (phi1 - phi2))))
	return float(plv)


# Advanced Tremor Features

def wrist_tremor_amplitude(wrist_y: np.ndarray, sr: float, band: tuple = (3.0, 7.0)) -> float:
	"""Compute RMS amplitude of wrist tremor in the Parkinsonian tremor frequency band (3-7Hz).
	
	Uses bandpass filter followed by RMS calculation.
	"""
	y = np.asarray(wrist_y)
	if y.size < 10 or sr <= 0:
		return 0.0
	
	try:
		# Design bandpass Butterworth filter
		nyquist = sr / 2.0
		low = band[0] / nyquist
		high = band[1] / nyquist
		if low <= 0 or high >= 1 or low >= high:
			return 0.0
		
		sos = signal.butter(4, [low, high], btype='band', output='sos')
		filtered = signal.sosfiltfilt(sos, y)
		
		# RMS amplitude
		rms = np.sqrt(np.mean(filtered ** 2))
		return float(rms)
	except Exception:
		return 0.0


def dominant_tremor_frequency(wrist_y: np.ndarray, sr: float, band: tuple = (3.0, 7.0)) -> float:
	"""Find the dominant frequency of wrist tremor within the specified band using Welch's method.
	
	Returns the frequency with maximum power spectral density in the band.
	"""
	y = np.asarray(wrist_y)
	if y.size < 10 or sr <= 0:
		return 0.0
	
	try:
		# Welch's method for power spectral density
		nperseg = min(y.size, int(sr * 2))  # 2-second windows
		freqs, psd = signal.welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
		
		# Filter to band of interest
		band_mask = (freqs >= band[0]) & (freqs <= band[1])
		if not np.any(band_mask):
			return 0.0
		
		band_freqs = freqs[band_mask]
		band_psd = psd[band_mask]
		
		# Find peak frequency
		peak_idx = np.argmax(band_psd)
		return float(band_freqs[peak_idx])
	except Exception:
		return 0.0


def tremor_power_3_7Hz(wrist_y: np.ndarray, sr: float, band: tuple = (3.0, 7.0)) -> float:
	"""Total power in the Parkinsonian tremor frequency band (3-7Hz).
	
	Computes the integral of power spectral density in the specified band.
	"""
	y = np.asarray(wrist_y)
	if y.size < 10 or sr <= 0:
		return 0.0
	
	try:
		# Welch's method for power spectral density
		nperseg = min(y.size, int(sr * 2))  # 2-second windows
		freqs, psd = signal.welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
		
		# Filter to band of interest
		band_mask = (freqs >= band[0]) & (freqs <= band[1])
		if not np.any(band_mask):
			return 0.0
		
		# Integrate power using trapezoidal rule
		band_freqs = freqs[band_mask]
		band_psd = psd[band_mask]
		total_power = np.trapz(band_psd, band_freqs)
		return float(total_power)
	except Exception:
		return 0.0


def tremor_intermittency(wrist_y: np.ndarray, sr: float, band: tuple = (3.0, 7.0), 
                         window_sec: float = 1.0, threshold_percentile: float = 50.0) -> float:
	"""Measure tremor intermittency as the fraction of time windows with low tremor power.
	
	Divides signal into windows, computes power in each, and calculates the fraction
	of windows below the threshold (median power by default).
	
	Higher values indicate more intermittent tremor (more low-power windows).
	"""
	y = np.asarray(wrist_y)
	if y.size < int(sr * window_sec * 2) or sr <= 0:
		return 0.0
	
	try:
		# Bandpass filter to tremor band
		nyquist = sr / 2.0
		low = band[0] / nyquist
		high = band[1] / nyquist
		if low <= 0 or high >= 1 or low >= high:
			return 0.0
		
		sos = signal.butter(4, [low, high], btype='band', output='sos')
		filtered = signal.sosfiltfilt(sos, y)
		
		# Divide into windows
		window_size = int(sr * window_sec)
		n_windows = len(filtered) // window_size
		
		if n_windows < 2:
			return 0.0
		
		# Calculate power in each window
		window_powers = []
		for i in range(n_windows):
			start = i * window_size
			end = start + window_size
			window = filtered[start:end]
			power = np.mean(window ** 2)
			window_powers.append(power)
		
		window_powers = np.array(window_powers)
		threshold = np.percentile(window_powers, threshold_percentile)
		
		# Fraction of windows below threshold
		intermittency = np.mean(window_powers < threshold)
		return float(intermittency)
	except Exception:
		return 0.0


def tremor_regularity(wrist_y: np.ndarray, sr: float, band: tuple = (3.0, 7.0)) -> float:
	"""Measure tremor regularity using spectral entropy.
	
	Lower entropy indicates more regular (concentrated power at specific frequencies).
	Returns normalized entropy (0-1 scale), where lower values = more regular.
	"""
	y = np.asarray(wrist_y)
	if y.size < 10 or sr <= 0:
		return 0.0
	
	try:
		# Welch's method for power spectral density
		nperseg = min(y.size, int(sr * 2))
		freqs, psd = signal.welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
		
		# Filter to band of interest
		band_mask = (freqs >= band[0]) & (freqs <= band[1])
		if not np.any(band_mask):
			return 0.0
		
		band_psd = psd[band_mask]
		
		# Normalize to probability distribution
		if np.sum(band_psd) == 0:
			return 0.0
		
		psd_norm = band_psd / np.sum(band_psd)
		
		# Calculate Shannon entropy
		ent = entropy(psd_norm, base=2)
		
		# Normalize by maximum possible entropy (uniform distribution)
		max_entropy = np.log2(len(psd_norm))
		if max_entropy == 0:
			return 0.0
		
		normalized_entropy = ent / max_entropy
		return float(normalized_entropy)
	except Exception:
		return 0.0


def jerk_magnitude(wrist_coords: np.ndarray, sr: float) -> float:
	"""Compute mean magnitude of jerk (third derivative of position).
	
	wrist_coords: (N, 3) array of [x, y, z] coordinates over time
	Returns mean Euclidean norm of jerk vector.
	"""
	coords = np.asarray(wrist_coords)
	if coords.ndim != 2 or coords.shape[0] < 4 or coords.shape[1] != 3 or sr <= 0:
		return 0.0
	
	try:
		# Velocity (first derivative)
		velocity = np.diff(coords, axis=0) * sr
		
		# Acceleration (second derivative)
		acceleration = np.diff(velocity, axis=0) * sr
		
		# Jerk (third derivative)
		jerk = np.diff(acceleration, axis=0) * sr
		
		# Magnitude of jerk vectors
		jerk_magnitudes = np.linalg.norm(jerk, axis=1)
		
		# Mean magnitude
		return float(np.mean(jerk_magnitudes))
	except Exception:
		return 0.0


#C Coordination / Inter-Joint Kinematics (6)


def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
	"""Compute joint angle at b formed by points a-b-c for each time sample.

	Inputs a,b,c can be arrays of shape (N, D) or (N,) for 1D. Returns angles in radians (N,).
	"""
	A = np.asarray(a)
	B = np.asarray(b)
	C = np.asarray(c)
	# ensure shapes (N, D)
	if A.ndim == 1:
		A = A[:, None]
	if B.ndim == 1:
		B = B[:, None]
	if C.ndim == 1:
		C = C[:, None]
	n = min(A.shape[0], B.shape[0], C.shape[0])
	if n == 0:
		return np.array([])
	A = A[:n]
	B = B[:n]
	C = C[:n]
	BA = A - B
	BC = C - B
	# compute angle between BA and BC using arctan2 of cross and dot
	# support 2D or 3D
	if BA.shape[1] == 2:
		cross = BA[:, 0] * BC[:, 1] - BA[:, 1] * BC[:, 0]
		dot = np.sum(BA * BC, axis=1)
		angles = np.arctan2(np.abs(cross), dot)
	else:
		# 3D or higher: use norm of cross product
		cross_vec = np.cross(BA, BC)
		cross_norm = np.linalg.norm(cross_vec, axis=1)
		dot = np.sum(BA * BC, axis=1)
		angles = np.arctan2(cross_norm, dot)
	return angles


def elbow_wrist_angular_velocity_stats(a: np.ndarray, b: np.ndarray, c: np.ndarray, sr: float) -> Tuple[float, float]:
	"""Compute mean and std of angular velocity at joint B formed by A-B-C.

	Returns (mean, std) of theta_dot in rad/s. Handles empty inputs by returning (0.0, 0.0).
	"""
	theta = _angle_at_joint(a, b, c)
	if theta.size < 2 or sr <= 0:
		return 0.0, 0.0
	dt = 1.0 / sr
	theta_dot = np.diff(theta) / dt
	return float(np.mean(theta_dot)), float(np.std(theta_dot))


def inter_limb_cross_correlation(left_signal: np.ndarray, right_signal: np.ndarray, sr: float, max_lag_sec: float = 0.5) -> float:
	"""Max normalized cross-correlation between left and right signals within ±max_lag_sec.

	Returns the maximum Pearson correlation across lags.
	"""
	x = np.asarray(left_signal)
	y = np.asarray(right_signal)
	if x.size == 0 or y.size == 0 or sr <= 0:
		return 0.0
	n = min(x.size, y.size)
	x = x[:n] - np.mean(x[:n])
	y = y[:n] - np.mean(y[:n])
	max_lag = int(max_lag_sec * sr)
	# full cross-correlation
	corr = np.correlate(x, y, mode='full')
	lags = np.arange(-n + 1, n)
	# normalize to Pearson r for each lag
	denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
	if denom == 0:
		return 0.0
	corr = corr / denom
	# select lags within max_lag
	mask = (lags >= -max_lag) & (lags <= max_lag)
	if not np.any(mask):
		return 0.0
	return float(np.max(np.abs(corr[mask])))


def relative_phase(angle1: np.ndarray, angle2: np.ndarray) -> float:
	"""Mean relative phase between two angle time series (radians), wrapped to [-pi,pi]."""
	a1 = np.asarray(angle1)
	a2 = np.asarray(angle2)
	n = min(a1.size, a2.size)
	if n == 0:
		return 0.0
	phi1 = np.angle(np.exp(1j * a1[:n]))
	phi2 = np.angle(np.exp(1j * a2[:n]))
	diff = phi1 - phi2
	# wrap
	diff = (diff + np.pi) % (2 * np.pi) - np.pi
	return float(np.mean(diff))


def smoothness_normalized_jerk(trajectory: np.ndarray, sr: float) -> float:
	"""Normalized jerk metric for smoothness.

	trajectory: array shape (N, D) or (N,). Computes third derivative magnitude squared integral normalized by path length squared.
	"""
	traj = np.asarray(trajectory)
	if traj.ndim == 1:
		traj = traj[:, None]
	n = traj.shape[0]
	if n < 4 or sr <= 0:
		return 0.0
	dt = 1.0 / sr
	# third derivative (jerk) using finite differences: d3x/dt3 ~ diff(x, n=3) / dt^3
	jerk = np.diff(traj, n=3, axis=0) / (dt ** 3)
	# squared magnitude
	jerk_sq = np.sum(jerk ** 2, axis=1)
	integral = np.sum(jerk_sq) * dt  # approximate integral over time
	T = n * dt
	# path length
	diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
	path_length = np.sum(diffs)
	if path_length == 0:
		return 0.0
	return float((1.0 / T) * integral / (path_length ** 2))


def angular_excursion_range(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
	"""Range (max-min) of joint angle at b formed by a-b-c (radians)."""
	theta = _angle_at_joint(a, b, c)
	if theta.size == 0:
		return 0.0
	return float(np.max(theta) - np.min(theta))


def coordination_variability(relative_angles: np.ndarray) -> float:
	"""Frame-to-frame standard deviation of inter-joint relative angles (std of diff).

	Input is time series of relative angles (radians).
	"""
	ra = np.asarray(relative_angles)
	if ra.size < 2:
		return 0.0
	return float(np.std(np.diff(ra)))


def detect_steps(signal_1d: np.ndarray, sr: float, min_step_sec: float = 0.45) -> np.ndarray:
	"""Detect step events from a 1D cyclic gait signal.

	Applies light low-pass filtering + Savitzky-Golay smoothing, then peak detection.
	Returns frame indices of detected events.
	"""
	x = np.asarray(signal_1d, dtype=float).ravel()
	if x.size < 3 or sr <= 0:
		return np.array([], dtype=int)

	if np.all(np.isnan(x)):
		return np.array([], dtype=int)
	x = np.where(np.isnan(x), float(np.nanmean(x)), x)

	# Low-pass filter to reduce frame-level jitter (PKMAS-style preprocessing spirit).
	try:
		nyq = 0.5 * float(sr)
		# Keep gait-scale oscillations while suppressing high-frequency jitter.
		cutoff_hz = min(4.0, 0.45 * nyq)
		if cutoff_hz > 0 and cutoff_hz < nyq and x.size >= 9:
			b, a = signal.butter(2, cutoff_hz / nyq, btype='low')
			x = signal.filtfilt(b, a, x)
	except Exception:
		pass

	# Savitzky-Golay smoothing (odd window, capped by signal length).
	win = max(5, int(round(0.25 * sr)))
	if win % 2 == 0:
		win += 1
	if win >= x.size:
		win = x.size - 1 if x.size % 2 == 0 else x.size
	if win >= 5:
		try:
			x = signal.savgol_filter(x, window_length=win, polyorder=2)
		except Exception:
			pass

	dist = max(1, int(min_step_sec * sr))
	prom = max(0.0, 0.20 * float(np.std(x)))
	width = max(1, int(round(0.05 * sr)))
	peaks, _ = signal.find_peaks(x, distance=dist, prominence=prom, width=width)
	return peaks.astype(int)


def step_time_stats(step_indices: np.ndarray, sr: float) -> Tuple[float, float]:
	"""Return mean/std step time (seconds) from step indices."""
	idx = np.asarray(step_indices, dtype=int).ravel()
	if idx.size < 2 or sr <= 0:
		return 0.0, 0.0
	times = np.diff(np.sort(idx)) / float(sr)
	if times.size == 0:
		return 0.0, 0.0
	return float(np.mean(times)), float(np.std(times))


def cadence(step_indices: np.ndarray, duration_sec: float, per_minute: bool = True) -> float:
	"""Compute cadence from detected steps.

	Returns steps/min when per_minute=True; otherwise steps/sec.
	"""
	idx = np.asarray(step_indices)
	if duration_sec <= 0:
		return 0.0
	rate_hz = float(idx.size) / float(duration_sec)
	return float(rate_hz * 60.0) if per_minute else rate_hz


def step_length_stats(left_ankle: np.ndarray, right_ankle: np.ndarray) -> Tuple[float, float]:
	"""Approximate step-length stats using left-right ankle distance over time."""
	L = np.asarray(left_ankle, dtype=float)
	R = np.asarray(right_ankle, dtype=float)
	if L.ndim == 1:
		L = L[:, None]
	if R.ndim == 1:
		R = R[:, None]
	n = min(L.shape[0], R.shape[0])
	if n == 0:
		return 0.0, 0.0
	L = L[:n]
	R = R[:n]
	dist = np.linalg.norm(L - R, axis=1)
	if dist.size == 0:
		return 0.0, 0.0
	return float(np.mean(dist)), float(np.std(dist))


def center_of_mass(hip_l: np.ndarray, hip_r: np.ndarray) -> np.ndarray:
	"""Approximate COM from left/right hip midpoint."""
	L = np.asarray(hip_l, dtype=float)
	R = np.asarray(hip_r, dtype=float)
	if L.ndim == 1:
		L = L[:, None]
	if R.ndim == 1:
		R = R[:, None]
	n = min(L.shape[0], R.shape[0])
	if n == 0:
		return np.empty((0, max(L.shape[1] if L.ndim == 2 else 1, R.shape[1] if R.ndim == 2 else 1)))
	return (L[:n] + R[:n]) / 2.0


def velocity_stats(x: np.ndarray, sr: float) -> Tuple[float, float]:
	"""Return mean/std speed from trajectory x (N,) or (N,D)."""
	arr = np.asarray(x, dtype=float)
	if arr.ndim == 1:
		arr = arr[:, None]
	if arr.shape[0] < 2 or sr <= 0:
		return 0.0, 0.0
	v = np.diff(arr, axis=0) * float(sr)
	speed = np.linalg.norm(v, axis=1)
	if speed.size == 0:
		return 0.0, 0.0
	return float(np.mean(speed)), float(np.std(speed))


def acceleration_stats(x: np.ndarray, sr: float) -> Tuple[float, float]:
	"""Return mean/std acceleration magnitude from trajectory x (N,) or (N,D)."""
	arr = np.asarray(x, dtype=float)
	if arr.ndim == 1:
		arr = arr[:, None]
	if arr.shape[0] < 3 or sr <= 0:
		return 0.0, 0.0
	v = np.diff(arr, axis=0) * float(sr)
	a = np.diff(v, axis=0) * float(sr)
	mag = np.linalg.norm(a, axis=1)
	if mag.size == 0:
		return 0.0, 0.0
	return float(np.mean(mag)), float(np.std(mag))


def coefficient_of_variation(x: np.ndarray) -> float:
	"""Generic coefficient of variation: std / mean."""
	arr = np.asarray(x, dtype=float)
	if arr.size == 0:
		return 0.0
	mu = float(np.mean(arr))
	if mu == 0:
		return 0.0
	return float(np.std(arr) / mu)


def asymmetry_index(left: float, right: float) -> float:
	"""Asymmetry index: (R-L)/(0.5*(R+L))."""
	den = 0.5 * (float(left) + float(right))
	if den == 0:
		return 0.0
	return float((float(right) - float(left)) / den)


def euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
	"""Euclidean distance between two points."""
	return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
	"""Return angle between vectors in radians, clipped for numeric stability."""
	a = np.asarray(v1, dtype=float)
	b = np.asarray(v2, dtype=float)
	den = float(np.linalg.norm(a) * np.linalg.norm(b))
	if den <= 0:
		return 0.0
	cos = float(np.dot(a, b) / den)
	return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _pick_progress_signal(coords: np.ndarray) -> np.ndarray:
	"""Pick gait progression axis from 3D coordinates (prefer z, fallback x)."""
	arr = np.asarray(coords, dtype=float)
	if arr.ndim == 1:
		return arr
	if arr.shape[0] == 0:
		return np.array([])
	if arr.shape[1] > 2 and np.isfinite(np.std(arr[:, 2])) and np.std(arr[:, 2]) > 1e-8:
		return arr[:, 2]
	return arr[:, 0]


def smooth_signal(signal_1d: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
	"""Savitzky-Golay smoothing with safe fallback for short signals."""
	x = np.asarray(signal_1d, dtype=float).ravel()
	if x.size < 5:
		return x
	win = int(max(5, window))
	if win % 2 == 0:
		win += 1
	if win >= x.size:
		win = x.size - 1 if x.size % 2 == 0 else x.size
	if win < 5:
		return x
	try:
		return signal.savgol_filter(x, window_length=win, polyorder=int(max(1, poly)))
	except Exception:
		return x


def _smooth_kinematic_signal(signal_1d: np.ndarray, sr: float) -> np.ndarray:
	"""Denoise gait kinematic signal using low-pass + Savitzky-Golay."""
	x = np.asarray(signal_1d, dtype=float).ravel()
	if x.size < 3 or sr <= 0:
		return np.array([])
	if np.all(np.isnan(x)):
		return np.array([])
	x = np.where(np.isnan(x), float(np.nanmean(x)), x)

	try:
		nyq = 0.5 * float(sr)
		cutoff_hz = min(4.0, 0.45 * nyq)
		if cutoff_hz > 0 and cutoff_hz < nyq and x.size >= 9:
			b, a = signal.butter(2, cutoff_hz / nyq, btype='low')
			x = signal.filtfilt(b, a, x)
	except Exception:
		pass

	win = max(7, int(round(0.25 * sr)))
	x = smooth_signal(x, window=win, poly=2)
	return x


def _detect_gait_events_robust(signal_1d: np.ndarray, sr: float, event_type: str, min_interval_sec: float) -> np.ndarray:
	"""Detect gait events using local extrema + velocity zero-crossings.

	event_type:
	- "hs": local minimum + neg->pos velocity
	- "to": local maximum + pos->neg velocity
	"""
	x = _smooth_kinematic_signal(signal_1d, sr=sr)
	if x.size < 5:
		return np.array([], dtype=int)

	vel = np.gradient(x) * float(sr)
	if vel.size < 5:
		return np.array([], dtype=int)

	min_dist = max(1, int(round(float(min_interval_sec) * float(sr))))
	prom = max(1e-8, 0.05 * float(np.std(x)))
	width = max(1, int(round(0.03 * float(sr))))
	try:
		if event_type == "hs":
			cand, _ = signal.find_peaks(-x, distance=min_dist, prominence=prom, width=width)
		else:
			cand, _ = signal.find_peaks(x, distance=min_dist, prominence=prom, width=width)
	except Exception:
		cand = np.array([], dtype=int)

	if cand.size == 0:
		return np.array([], dtype=int)

	kept: list = []
	for idx in cand:
		i = int(idx)
		l = max(1, i - 2)
		r = min(vel.size - 1, i + 2)
		if event_type == "hs":
			zero_cross = any((vel[k - 1] < 0.0 and vel[k] >= 0.0) for k in range(l, r + 1))
		else:
			zero_cross = any((vel[k - 1] > 0.0 and vel[k] <= 0.0) for k in range(l, r + 1))
		if zero_cross:
			kept.append(i)

	return np.asarray(kept, dtype=int)


def filter_by_physiology(events: list, min_step_sec: float = 0.3, max_step_sec: float = 1.5) -> list:
	"""Filter events by adjacent time interval constraints."""
	if len(events) <= 1:
		return events
	filtered = [events[0]]
	for ev in events[1:]:
		dt = float(ev[1] - filtered[-1][1])
		if float(min_step_sec) < dt < float(max_step_sec):
			filtered.append(ev)
	return filtered


def enforce_alternation(
	events_left: list,
	events_right: list,
	min_step_sec: float = 0.3,
	max_step_sec: float = 1.5,
) -> Tuple[list, list, list]:
	"""Enforce L/R alternation and physiological timing on merged events."""
	merged = [(int(i), float(t), "L") for i, t in events_left] + [(int(i), float(t), "R") for i, t in events_right]
	if not merged:
		return [], [], []
	merged.sort(key=lambda it: it[1])
	filtered = []
	last_side = None
	for item in merged:
		if item[2] != last_side:
			filtered.append(item)
			last_side = item[2]

	if not filtered:
		return [], [], []

	phys = [filtered[0]]
	for item in filtered[1:]:
		dt = float(item[1] - phys[-1][1])
		if float(min_step_sec) < dt < float(max_step_sec):
			phys.append(item)

	left = [(i, t) for i, t, side in phys if side == "L"]
	right = [(i, t) for i, t, side in phys if side == "R"]
	combined = [(i, t) for i, t, _ in phys]
	return left, right, combined


def detect_heel_strikes(
	heel_positions: np.ndarray,
	timestamps: np.ndarray,
	ankle_positions: Optional[np.ndarray] = None,
	min_interval_sec: float = 0.35,
	heel_weight: float = 0.7,
) -> list:
	"""Detect heel strikes using extrema + velocity criteria with optional heel/ankle fusion."""
	heel = np.asarray(heel_positions, dtype=float)
	ts = np.asarray(timestamps, dtype=float).ravel()
	ankle = np.asarray(ankle_positions, dtype=float) if ankle_positions is not None else None
	n = min(heel.shape[0] if heel.ndim > 1 else heel.size, ts.size)
	if ankle is not None:
		n = min(n, ankle.shape[0] if ankle.ndim > 1 else ankle.size)
	if n < 3:
		return []
	heel = heel[:n] if heel.ndim > 1 else heel[:n, None]
	if ankle is not None:
		ankle = ankle[:n] if ankle.ndim > 1 else ankle[:n, None]
	ts = ts[:n]
	dt = np.diff(ts)
	valid = dt[np.isfinite(dt) & (dt > 0)]
	if valid.size == 0:
		return []
	sr = float(1.0 / np.median(valid))
	heel_sig = _pick_progress_signal(heel)
	if ankle is not None and ankle.shape[0] == n:
		ankle_sig = _pick_progress_signal(ankle)
		w = float(np.clip(heel_weight, 0.0, 1.0))
		sig = w * heel_sig + (1.0 - w) * ankle_sig
	else:
		sig = heel_sig
	events = _detect_gait_events_robust(sig, sr=sr, event_type="hs", min_interval_sec=min_interval_sec)
	out = [(int(i), float(ts[i])) for i in events if 0 <= int(i) < ts.size]
	# Same-foot HS intervals are stride-like and usually slower than alternating step intervals.
	return filter_by_physiology(out, min_step_sec=0.45, max_step_sec=2.5)


def detect_toe_offs(
	toe_positions: np.ndarray,
	timestamps: np.ndarray,
	ankle_positions: Optional[np.ndarray] = None,
	min_interval_sec: float = 0.35,
	toe_weight: float = 0.7,
) -> list:
	"""Detect toe offs using extrema + velocity criteria with optional toe/ankle fusion."""
	toe = np.asarray(toe_positions, dtype=float)
	ts = np.asarray(timestamps, dtype=float).ravel()
	ankle = np.asarray(ankle_positions, dtype=float) if ankle_positions is not None else None
	n = min(toe.shape[0] if toe.ndim > 1 else toe.size, ts.size)
	if ankle is not None:
		n = min(n, ankle.shape[0] if ankle.ndim > 1 else ankle.size)
	if n < 3:
		return []
	toe = toe[:n] if toe.ndim > 1 else toe[:n, None]
	if ankle is not None:
		ankle = ankle[:n] if ankle.ndim > 1 else ankle[:n, None]
	ts = ts[:n]
	dt = np.diff(ts)
	valid = dt[np.isfinite(dt) & (dt > 0)]
	if valid.size == 0:
		return []
	sr = float(1.0 / np.median(valid))
	toe_sig = _pick_progress_signal(toe)
	if ankle is not None and ankle.shape[0] == n:
		ankle_sig = _pick_progress_signal(ankle)
		w = float(np.clip(toe_weight, 0.0, 1.0))
		sig = w * toe_sig + (1.0 - w) * ankle_sig
	else:
		sig = toe_sig
	events = _detect_gait_events_robust(sig, sr=sr, event_type="to", min_interval_sec=min_interval_sec)
	out = [(int(i), float(ts[i])) for i in events if 0 <= int(i) < ts.size]
	return filter_by_physiology(out, min_step_sec=0.45, max_step_sec=2.5)


def compute_stride_time(heel_strikes: list) -> list:
	"""Stride time from consecutive same-foot heel strikes."""
	if len(heel_strikes) < 2:
		return []
	out = []
	for i in range(len(heel_strikes) - 1):
		d = float(heel_strikes[i + 1][1] - heel_strikes[i][1])
		if d > 0:
			out.append(d)
	return out


def compute_stride_length(heel_positions: np.ndarray, heel_strikes: list) -> list:
	"""Stride length from consecutive same-foot heel strike positions."""
	heel = np.asarray(heel_positions, dtype=float)
	if heel.ndim == 1:
		heel = heel[:, None]
	if len(heel_strikes) < 2 or heel.shape[0] == 0:
		return []
	out = []
	for i in range(len(heel_strikes) - 1):
		i0 = int(heel_strikes[i][0])
		i1 = int(heel_strikes[i + 1][0])
		if 0 <= i0 < heel.shape[0] and 0 <= i1 < heel.shape[0]:
			out.append(euclidean(heel[i0], heel[i1]))
	return out


def compute_stride_speed(stride_length: list, stride_time: list) -> list:
	"""Stride speed = stride length / stride time."""
	if not stride_length or not stride_time:
		return []
	out = []
	for l, t in zip(stride_length, stride_time):
		tf = float(t)
		if tf > 0:
			out.append(float(l) / tf)
	return out


def compute_stance_time(heel_strikes: list, toe_offs: list) -> list:
	"""Stance duration in each gait cycle (HS -> next TO before next HS)."""
	if len(heel_strikes) < 2 or len(toe_offs) == 0:
		return []
	out = []
	to_idx = 0
	for i in range(len(heel_strikes) - 1):
		hs_idx, hs_t = heel_strikes[i]
		next_hs_idx = heel_strikes[i + 1][0]
		while to_idx < len(toe_offs) and toe_offs[to_idx][0] <= hs_idx:
			to_idx += 1
		if to_idx >= len(toe_offs):
			break
		to_event_idx, to_t = toe_offs[to_idx]
		if to_event_idx >= next_hs_idx:
			continue
		d = float(to_t - hs_t)
		if d > 0:
			out.append(d)
	return out


def compute_swing_time(heel_strikes: list, toe_offs: list) -> list:
	"""Swing duration in each gait cycle (TO -> next HS)."""
	if len(heel_strikes) < 2 or len(toe_offs) == 0:
		return []
	out = []
	to_idx = 0
	for i in range(len(heel_strikes) - 1):
		hs_idx = heel_strikes[i][0]
		next_hs_t = heel_strikes[i + 1][1]
		next_hs_idx = heel_strikes[i + 1][0]
		while to_idx < len(toe_offs) and toe_offs[to_idx][0] <= hs_idx:
			to_idx += 1
		if to_idx >= len(toe_offs):
			break
		to_event_idx, to_t = toe_offs[to_idx]
		if to_event_idx >= next_hs_idx:
			continue
		d = float(next_hs_t - to_t)
		if d > 0:
			out.append(d)
	return out


def compute_stance_ratio(stance: list, stride: list) -> list:
	"""Stance ratio per cycle = stance_time / stride_time."""
	if not stance or not stride:
		return []
	out = []
	for s, st in zip(stance, stride):
		stf = float(st)
		if stf > 0:
			out.append(float(s) / stf)
	return out


def compute_foot_progression_angle(heel_positions: np.ndarray, toe_positions: np.ndarray, heel_strikes: list) -> list:
	"""Foot progression angle (radians) at heel-strike frames."""
	heel = np.asarray(heel_positions, dtype=float)
	toe = np.asarray(toe_positions, dtype=float)
	if heel.ndim == 1:
		heel = heel[:, None]
	if toe.ndim == 1:
		toe = toe[:, None]
	n = min(heel.shape[0], toe.shape[0])
	if len(heel_strikes) < 3 or n == 0:
		return []
	out = []
	for i in range(1, len(heel_strikes) - 1):
		curr_idx = int(heel_strikes[i][0])
		next_idx = int(heel_strikes[i + 1][0])
		prev_idx = int(heel_strikes[i - 1][0])
		if not (0 <= prev_idx < n and 0 <= curr_idx < n and 0 <= next_idx < n):
			continue
		progress_vec = heel[next_idx] - heel[curr_idx]
		foot_vec = toe[curr_idx] - heel[curr_idx]
		if np.linalg.norm(progress_vec) <= 1e-8 or np.linalg.norm(foot_vec) <= 1e-8:
			continue
		out.append(angle_between(foot_vec, progress_vec))
	return out


def extract_gait_features(landmarks: Dict[str, np.ndarray], sr: float) -> Dict[str, float]:
	"""Extract PKMAS-style gait summary features from landmark dict arrays.

	Expected keys include: left_ankle, right_ankle, left_hip, right_hip.
	"""
	out: Dict[str, float] = {}
	if not isinstance(landmarks, dict) or sr <= 0:
		return out

	left_ankle = np.asarray(landmarks.get("left_ankle", np.array([])), dtype=float)
	right_ankle = np.asarray(landmarks.get("right_ankle", np.array([])), dtype=float)
	hip_l = np.asarray(landmarks.get("left_hip", np.array([])), dtype=float)
	hip_r = np.asarray(landmarks.get("right_hip", np.array([])), dtype=float)

	if left_ankle.size == 0 or right_ankle.size == 0:
		return out

	n = min(left_ankle.shape[0], right_ankle.shape[0])
	if n < 3:
		return out

	# Prefer z-axis for progression; fallback to x-axis.
	left_sig = left_ankle[:n, 2] if left_ankle.ndim == 2 and left_ankle.shape[1] > 2 else left_ankle[:n, 0]
	right_sig = right_ankle[:n, 2] if right_ankle.ndim == 2 and right_ankle.shape[1] > 2 else right_ankle[:n, 0]

	left_steps = detect_steps(left_sig, sr, min_step_sec=0.45)
	right_steps = detect_steps(right_sig, sr, min_step_sec=0.45)
	all_steps = np.sort(np.concatenate([left_steps, right_steps])) if (left_steps.size or right_steps.size) else np.array([], dtype=int)

	duration = float(n) / float(sr)
	step_mean, step_std = step_time_stats(all_steps, sr)
	step_l_mean, step_l_std = step_time_stats(left_steps, sr)
	step_r_mean, step_r_std = step_time_stats(right_steps, sr)

	step_len_mean, step_len_std = step_length_stats(left_ankle[:n], right_ankle[:n])
	com = center_of_mass(hip_l, hip_r)
	vel_mean, vel_std = velocity_stats(com, sr)
	acc_mean, acc_std = acceleration_stats(com, sr)

	out.update({
		"step_time_mean": step_mean,
		"step_time_std": step_std,
		"step_time_left_mean": step_l_mean,
		"step_time_left_std": step_l_std,
		"step_time_right_mean": step_r_mean,
		"step_time_right_std": step_r_std,
		"cadence_spm": cadence(all_steps, duration, per_minute=True),
		"cadence_hz": cadence(all_steps, duration, per_minute=False),
		"step_length_mean": step_len_mean,
		"step_length_std": step_len_std,
		"velocity_mean": vel_mean,
		"velocity_std": vel_std,
		"acceleration_mean": acc_mean,
		"acceleration_std": acc_std,
		"cv_step_time": coefficient_of_variation(np.diff(all_steps) / float(sr)) if all_steps.size > 1 else 0.0,
		"step_time_asymmetry_index": asymmetry_index(step_l_mean, step_r_mean),
	})
	return out


#D Spatiotemporal Gait & Global Movement (6)
# D.1 Mean gait speed
def mean_gait_speed(pelvis_x: np.ndarray, sr: float) -> float:
	"""Estimate mean gait speed (m/s) from pelvis x-coordinates.

	Uses mean frame-to-frame absolute velocity: mean(|dx|)*sr
	Returns 0.0 for empty input or non-positive sr.
	"""
	x = np.asarray(pelvis_x)
	if x.size < 2 or sr <= 0:
		return 0.0
	dx = np.diff(x)
	mean_speed = float(np.mean(np.abs(dx)) * sr)
	return mean_speed


# D.2 Step time variability (CV of step durations)
def step_time_variability(step_times: np.ndarray) -> float:
	"""Coefficient of variation of step durations: std / mean."""
	arr = np.asarray(step_times)
	if arr.size == 0:
		return 0.0
	mean = float(np.mean(arr))
	if mean == 0:
		return 0.0
	return float(np.std(arr) / mean)


# D.3 Stride length statistics (mean and CV)
def stride_length_stats(stride_lengths: np.ndarray) -> Tuple[float, float]:
	"""Return (mean_stride_length, cv_stride_length).

	If input is empty returns (0.0, 0.0).
	"""
	s = np.asarray(stride_lengths)
	if s.size == 0:
		return 0.0, 0.0
	mean = float(np.mean(s))
	if mean == 0:
		cv = 0.0
	else:
		cv = float(np.std(s) / mean)
	return mean, cv


# D.4 Double support time fraction
def double_support_time_fraction(double_support_times: np.ndarray, cycle_times: np.ndarray) -> float:
	"""Mean fraction of gait cycle spent in double support: mean(Tdouble)/mean(Tcycle)."""
	D = np.asarray(double_support_times)
	C = np.asarray(cycle_times)
	if D.size == 0 or C.size == 0:
		return 0.0
	meanD = float(np.mean(D))
	meanC = float(np.mean(C))
	if meanC == 0:
		return 0.0
	return float(meanD / meanC)


# D.5 Center-of-mass sway amplitude
def com_sway_amplitude(hips_x: np.ndarray) -> float:
	"""Lateral trunk sway amplitude: std of mean hip x-position across time.

	hips_x can be 1D (center) or 2D (N x 2 with left/right hip columns). If 2D,
	the hips center is computed as mean across columns for each frame.
	"""
	arr = np.asarray(hips_x)
	if arr.size == 0:
		return 0.0
	if arr.ndim == 1:
		center = arr
	else:
		# mean across hip columns (axis 1) to get center x per frame
		center = np.mean(arr, axis=1)
	if center.size == 0:
		return 0.0
	return float(np.std(center))


# D.6 Path curvature / lateral deviation
def path_lateral_deviation(path_y: np.ndarray) -> float:
	"""Mean absolute lateral change of the walking path: mean(|Δy|).

	path_y is lateral coordinate (per-frame). Empty input returns 0.0.
	"""
	y = np.asarray(path_y)
	if y.size < 2:
		return 0.0
	dy = np.abs(np.diff(y))
	return float(np.mean(dy))


#E Energy / Dynamics (3)
# E.1 Kinetic energy proxy
def kinetic_energy_proxy(velocities: np.ndarray) -> float:
	"""Proxy for kinetic energy: mean(0.5 * speed^2).

	velocities can be (N,) or (N,D). If (N,D) per-frame speed is norm across D.
	"""
	v = np.asarray(velocities)
	if v.size == 0:
		return 0.0
	if v.ndim == 1:
		speed_sq = v ** 2
	else:
		speed = np.linalg.norm(v, axis=1)
		speed_sq = speed ** 2
	return float(np.mean(0.5 * speed_sq))


# E.2 Acceleration bursts per minute
def acceleration_bursts_per_minute(accel: np.ndarray, sr: float, tau: float) -> float:
	"""Count of high-acceleration events per minute.

	accel may be 1D (scalar acceleration per frame) or 2D (N,D) where frame norm is used.
	tau is threshold; sr is sampling rate (Hz). Returns 0.0 when duration is zero or inputs invalid.
	"""
	a = np.asarray(accel)
	if a.size == 0 or sr <= 0 or tau <= 0:
		return 0.0
	if a.ndim == 1:
		mags = np.abs(a)
	else:
		mags = np.linalg.norm(a, axis=1)
	count = int(np.sum(mags > tau))
	duration_sec = float(mags.size / sr)
	if duration_sec <= 0:
		return 0.0
	bursts_per_min = float(count / (duration_sec / 60.0))
	return bursts_per_min


# E.3 Movement entropy (spatial)
def spatial_movement_entropy(positions: np.ndarray, bins: int = 32) -> float:
	"""Spatial entropy of positions: -sum p * log(p).

	positions may be 1D (scalar positions) or 2D (N,D). For 2D we compute a 2D histogram
	with (bins,bins) and compute entropy of the occupancy distribution.
	"""
	pos = np.asarray(positions)
	if pos.size == 0:
		return 0.0
	try:
		if pos.ndim == 1:
			hist, _ = np.histogram(pos, bins=bins)
		else:
			# 2D histogram over first two columns
			x = pos[:, 0]
			y = pos[:, 1] if pos.shape[1] > 1 else np.zeros_like(x)
			hist, _, _ = np.histogram2d(x, y, bins=(bins, bins))
	except Exception:
		return 0.0
	total = np.sum(hist)
	if total == 0:
		return 0.0
	p = hist / total
	p = p[p > 0]
	return float(-np.sum(p * np.log(p)))


# ======== BrainWalk landmark-array feature bridge ========

def _np_clean1d(a: np.ndarray, method: str = "forward_fill") -> np.ndarray:
	"""Clean 1D array with missing values using specified method.
	
	Args:
		a: Input array with possible NaN values
		method: Cleaning strategy
			- "forward_fill": Forward fill then backward fill (preserves more structure)
			- "mean": Replace NaN with mean (original, deprecated approach)
	
	Returns:
		Cleaned array with NaN values handled
	"""
	a = np.asarray(a, dtype=float)
	if a.size == 0:
		return np.array([])
	if np.all(np.isnan(a)):
		return np.array([])
	
	if method == "forward_fill":
		# Forward fill: propagate last valid value forward
		mask = np.isnan(a)
		if not np.any(mask):
			return a
		idx = np.where(~mask, np.arange(mask.size), 0)
		idx = np.maximum.accumulate(idx)
		filled = a[idx]
		# Backward fill for leading NaNs
		mask_after_ff = np.isnan(filled)
		if np.any(mask_after_ff):
			idx_back = np.where(~mask_after_ff, np.arange(mask_after_ff.size), mask_after_ff.size-1)
			idx_back = np.minimum.accumulate(idx_back[::-1])[::-1]
			filled = filled[idx_back]
		return filled
	else:  # method == "mean"
		# Original mean-based imputation
		m = float(np.nanmean(a))
		return np.where(np.isnan(a), m, a)


def _interpolate_landmarks(landmarks: np.ndarray, max_gap: int = 10) -> tuple:
	"""Linearly interpolate short NaN gaps (<= max_gap) per landmark axis.

	Returns (interp_array, quality_metrics_dict).
	quality_metrics keys:
	- landmark_mean_nan_ratio: original NaN fraction
	- landmark_gap_filled_ratio: fraction of NaNs filled by interpolation
	- landmark_large_gap_count: number of NaN runs > max_gap (not filled)
	"""
	arr = np.array(landmarks, dtype=float)
	if arr.ndim != 3:
		return arr, {
			"landmark_mean_nan_ratio": 0.0,
			"landmark_gap_filled_ratio": 0.0,
			"landmark_large_gap_count": 0.0,
		}
	n_frames, n_landmarks, n_axes = arr.shape
	orig_nans = np.isnan(arr).sum()
	filled = 0
	large_gap_count = 0
	for li in range(n_landmarks):
		for ax in range(n_axes):
			col = arr[:, li, ax]
			is_nan = np.isnan(col)
			if not np.any(is_nan):
				continue
			# find runs of NaNs
			idx = np.arange(n_frames)
			in_run = False
			run_start = 0
			for i in range(n_frames + 1):
				end_run = (i == n_frames) or (i < n_frames and not is_nan[i] and in_run)
				start_new = (i < n_frames and is_nan[i] and not in_run)
				if start_new:
					in_run = True
					run_start = i
				elif end_run and in_run:
					run_end = i  # exclusive
					run_len = run_end - run_start
					# interpolation conditions: run bounded by valid values on both sides and length <= max_gap
					left_idx = run_start - 1
					right_idx = run_end
					can_interp = (run_len <= max_gap and left_idx >= 0 and right_idx < n_frames and not np.isnan(col[left_idx]) and not np.isnan(col[right_idx]))
					if can_interp:
						left_val = float(col[left_idx])
						right_val = float(col[right_idx])
						span = right_idx - left_idx
						for j in range(run_start, run_end):
							t = (j - left_idx) / span
							col[j] = left_val + t * (right_val - left_val)
						filled += run_len
					else:
						large_gap_count += 1
					in_run = False
			arr[:, li, ax] = col
	quality = {
		"landmark_mean_nan_ratio": float(orig_nans) / float(n_frames * n_landmarks * n_axes) if n_frames > 0 else 0.0,
		"landmark_gap_filled_ratio": float(filled) / float(orig_nans) if orig_nans > 0 else 0.0,
		"landmark_large_gap_count": float(large_gap_count),
	}
	return arr, quality


def features_from_landmark_array(landmarks: np.ndarray, fps: float = 30.0) -> Dict[str, float]:
	"""Compute a subset of engineered features from BrainWalk landmark array.

	landmarks: shape (n_frames, n_landmarks, 3), axes = (x,y,z). Missing values allowed.
	Returns a flat dict with keys matching existing feature naming where possible.

	Note: step_time_variability and double_support_time_fraction require step events; returns NaN placeholders.
	"""
	out: Dict[str, float] = {}
	if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.ndim != 3 or landmarks.shape[0] == 0:
		return out
	# interpolate short gaps for robustness
	interp, q = _interpolate_landmarks(landmarks, max_gap=10)
	for k, v in q.items():
		out[k] = float(v)
	n = interp.shape[0]
	# indices (MediaPipe Pose)
	L_SHOULDER, R_SHOULDER = 11, 12
	L_ELBOW, R_ELBOW = 13, 14
	L_WRIST, R_WRIST = 15, 16
	L_HIP, R_HIP = 23, 24
	L_KNEE, R_KNEE = 25, 26
	L_ANKLE, R_ANKLE = 27, 28
	L_HEEL, R_HEEL = 29, 30
	L_TOE, R_TOE = 31, 32
	# guard
	def _get(idx, ax):
		if idx < interp.shape[1]:
			return interp[:, idx, ax]
		return np.full((n,), np.nan, dtype=float)
	# coordinates (cleaned)
	ls_x = _np_clean1d(_get(L_SHOULDER, 0)); ls_y = _np_clean1d(_get(L_SHOULDER, 1)); ls_z = _np_clean1d(_get(L_SHOULDER, 2))
	rs_x = _np_clean1d(_get(R_SHOULDER, 0)); rs_y = _np_clean1d(_get(R_SHOULDER, 1)); rs_z = _np_clean1d(_get(R_SHOULDER, 2))
	le_x = _np_clean1d(_get(L_ELBOW, 0)); le_y = _np_clean1d(_get(L_ELBOW, 1)); le_z = _np_clean1d(_get(L_ELBOW, 2))
	re_x = _np_clean1d(_get(R_ELBOW, 0)); re_y = _np_clean1d(_get(R_ELBOW, 1)); re_z = _np_clean1d(_get(R_ELBOW, 2))
	lw_x = _np_clean1d(_get(L_WRIST, 0)); lw_y = _np_clean1d(_get(L_WRIST, 1)); lw_z = _np_clean1d(_get(L_WRIST, 2))
	rw_x = _np_clean1d(_get(R_WRIST, 0)); rw_y = _np_clean1d(_get(R_WRIST, 1)); rw_z = _np_clean1d(_get(R_WRIST, 2))
	lh_x = _np_clean1d(_get(L_HIP, 0)); lh_y = _np_clean1d(_get(L_HIP, 1)); lh_z = _np_clean1d(_get(L_HIP, 2))
	rh_x = _np_clean1d(_get(R_HIP, 0)); rh_y = _np_clean1d(_get(R_HIP, 1)); rh_z = _np_clean1d(_get(R_HIP, 2))
	lk_x = _np_clean1d(_get(L_KNEE, 0)); lk_y = _np_clean1d(_get(L_KNEE, 1)); lk_z = _np_clean1d(_get(L_KNEE, 2))
	rk_x = _np_clean1d(_get(R_KNEE, 0)); rk_y = _np_clean1d(_get(R_KNEE, 1)); rk_z = _np_clean1d(_get(R_KNEE, 2))
	la_x = _np_clean1d(_get(L_ANKLE, 0)); la_y = _np_clean1d(_get(L_ANKLE, 1)); la_z = _np_clean1d(_get(L_ANKLE, 2))
	ra_x = _np_clean1d(_get(R_ANKLE, 0)); ra_y = _np_clean1d(_get(R_ANKLE, 1)); ra_z = _np_clean1d(_get(R_ANKLE, 2))
	lheel_x = _np_clean1d(_get(L_HEEL, 0)); lheel_y = _np_clean1d(_get(L_HEEL, 1)); lheel_z = _np_clean1d(_get(L_HEEL, 2))
	rheel_x = _np_clean1d(_get(R_HEEL, 0)); rheel_y = _np_clean1d(_get(R_HEEL, 1)); rheel_z = _np_clean1d(_get(R_HEEL, 2))
	ltoe_x = _np_clean1d(_get(L_TOE, 0)); ltoe_y = _np_clean1d(_get(L_TOE, 1)); ltoe_z = _np_clean1d(_get(L_TOE, 2))
	rtoe_x = _np_clean1d(_get(R_TOE, 0)); rtoe_y = _np_clean1d(_get(R_TOE, 1)); rtoe_z = _np_clean1d(_get(R_TOE, 2))

	def _stack_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=float)
		y = np.asarray(y, dtype=float)
		z = np.asarray(z, dtype=float)
		n_xyz = min(x.size, y.size, z.size)
		if n_xyz == 0:
			return np.empty((0, 3), dtype=float)
		return np.column_stack([x[:n_xyz], y[:n_xyz], z[:n_xyz]])

	# pelvis center (x,z) smoothed (simple moving average)
	pelvis_x_raw = (lh_x + rh_x) / 2.0
	pelvis_z_raw = (lh_z + rh_z) / 2.0
	def _smooth(sig):
		s = np.asarray(sig, dtype=float)
		w = 5
		if s.size < w:
			return s
		ker = np.ones(w, dtype=float) / w
		return np.convolve(s, ker, mode='same')
	pelvis_x = _np_clean1d(_smooth(pelvis_x_raw))
	pelvis_z = _np_clean1d(_smooth(pelvis_z_raw))

	# Tremor / coordination features
	out['tremor_cross_wrist_coherence'] = cross_wrist_coherence(lw_y, rw_y, sr=float(fps))
	
	# Advanced tremor features - wrist Y-axis (vertical motion) analysis
	# Use left wrist as primary for individual tremor metrics, average for bilateral metrics
	out['wrist_tremor_amplitude'] = (wrist_tremor_amplitude(lw_y, sr=float(fps)) + 
	                                  wrist_tremor_amplitude(rw_y, sr=float(fps))) / 2.0
	out['dominant_frequency'] = (dominant_tremor_frequency(lw_y, sr=float(fps)) + 
	                             dominant_tremor_frequency(rw_y, sr=float(fps))) / 2.0
	out['power_3_7Hz'] = (tremor_power_3_7Hz(lw_y, sr=float(fps)) + 
	                      tremor_power_3_7Hz(rw_y, sr=float(fps))) / 2.0
	out['tremor_intermittency'] = (tremor_intermittency(lw_y, sr=float(fps)) + 
	                               tremor_intermittency(rw_y, sr=float(fps))) / 2.0
	out['tremor_regularity'] = (tremor_regularity(lw_y, sr=float(fps)) + 
	                            tremor_regularity(rw_y, sr=float(fps))) / 2.0
	
	# Jerk magnitude requires 3D coordinates
	lw_coords = np.column_stack([lw_x, lw_y, lw_z])
	rw_coords = np.column_stack([rw_x, rw_y, rw_z])
	out['jerk_magnitude'] = (jerk_magnitude(lw_coords, sr=float(fps)) + 
	                         jerk_magnitude(rw_coords, sr=float(fps))) / 2.0
	
	# Phase locking value between left and right wrists
	out['phase_locking_value'] = phase_locking_value(lw_y, rw_y)
	
	out['shoulder_height_asymmetry'] = shoulder_height_asymmetry(ls_y, rs_y)
	out['arm_swing_amplitude_asymmetry'] = arm_swing_amplitude_asymmetry(lw_x, ls_x, rw_x, rs_x)

	# Arm swing peak-to-peak (choose axis with larger variance per side between x and z relative motions)
	left_rel_x = lw_x - ls_x; left_rel_z = lw_z - ls_z
	right_rel_x = rw_x - rs_x; right_rel_z = rw_z - rs_z
	left_axis = left_rel_z if np.var(left_rel_z) > np.var(left_rel_x) else left_rel_x
	right_axis = right_rel_z if np.var(right_rel_z) > np.var(right_rel_x) else right_rel_x
	arm_left_pp = float(np.max(left_axis) - np.min(left_axis)) if left_axis.size else 0.0
	arm_right_pp = float(np.max(right_axis) - np.min(right_axis)) if right_axis.size else 0.0
	out['arm_swing_peak_to_peak_left'] = arm_left_pp
	out['arm_swing_peak_to_peak_right'] = arm_right_pp
	den_pp = (arm_left_pp + arm_right_pp) / 2.0
	out['arm_swing_peak_to_peak_asymmetry'] = (abs(arm_left_pp - arm_right_pp) / den_pp) if den_pp > 0 else 0.0

	# Enhanced speed (2D path length over x,z)
	if pelvis_x.size > 1 and pelvis_z.size == pelvis_x.size:
		path_diffs = np.sqrt(np.diff(pelvis_x)**2 + np.diff(pelvis_z)**2)
		path_len = float(np.sum(path_diffs))
		duration_sec = float(pelvis_x.size / float(fps)) if fps > 0 else 0.0
		out['gait_speed_enhanced'] = (path_len / duration_sec) if duration_sec > 0 else 0.0
	else:
		out['gait_speed_enhanced'] = 0.0
	out['mean_gait_speed'] = mean_gait_speed(pelvis_x, sr=float(fps))

	# Stride width proxy (mean lateral ankle separation)
	out['stride_width_mean'] = stride_width_mean(la_x, ra_x)

	# Step / gait-cycle events from smoothed progression signal (prefer z then x).
	def _pick_step_signal(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
		p = np.asarray(primary, dtype=float)
		s = np.asarray(secondary, dtype=float)
		if p.size >= 3 and np.isfinite(np.std(p)) and np.std(p) > 1e-8:
			return p
		return s

	left_sig = _pick_step_signal(la_z, la_x)
	right_sig = _pick_step_signal(ra_z, ra_x)
	left_events = detect_steps(left_sig, sr=float(fps), min_step_sec=0.45)
	right_events = detect_steps(right_sig, sr=float(fps), min_step_sec=0.45)
	out['step_events_left_count'] = float(left_events.size)
	out['step_events_right_count'] = float(right_events.size)
	all_events = np.sort(np.concatenate([left_events, right_events])) if (left_events.size or right_events.size) else np.array([], dtype=int)
	l_step_mean, l_step_std = step_time_stats(left_events, sr=float(fps))
	r_step_mean, r_step_std = step_time_stats(right_events, sr=float(fps))
	step_mean, step_std = step_time_stats(all_events, sr=float(fps))
	out['step_time_mean'] = step_mean
	out['step_time_std'] = step_std
	out['step_time_variability'] = coefficient_of_variation(
		(np.diff(all_events) / float(fps)) if all_events.size > 1 and fps > 0 else np.array([])
	)
	out['step_time_left_mean'] = l_step_mean
	out['step_time_left_std'] = l_step_std
	out['step_time_right_mean'] = r_step_mean
	out['step_time_right_std'] = r_step_std
	out['step_time_asymmetry_index'] = asymmetry_index(l_step_mean, r_step_mean)
	duration_sec = float(n / float(fps)) if fps > 0 else 0.0
	out['cadence_spm'] = cadence(all_events, duration_sec, per_minute=True)
	out['cadence_hz'] = cadence(all_events, duration_sec, per_minute=False)

	# Gait-cycle core features from heel/toe events.
	def _mean_or_zero(values: list) -> float:
		arr = np.asarray(values, dtype=float)
		return float(np.mean(arr)) if arr.size else 0.0

	timestamps = np.arange(n, dtype=float) / float(fps) if fps > 0 else np.array([], dtype=float)
	left_ankle_xyz = _stack_xyz(la_x, la_y, la_z)
	right_ankle_xyz = _stack_xyz(ra_x, ra_y, ra_z)
	left_heel_xyz = _stack_xyz(lheel_x, lheel_y, lheel_z)
	right_heel_xyz = _stack_xyz(rheel_x, rheel_y, rheel_z)
	left_toe_xyz = _stack_xyz(ltoe_x, ltoe_y, ltoe_z)
	right_toe_xyz = _stack_xyz(rtoe_x, rtoe_y, rtoe_z)

	left_hs_raw = detect_heel_strikes(left_heel_xyz, timestamps, ankle_positions=left_ankle_xyz)
	right_hs_raw = detect_heel_strikes(right_heel_xyz, timestamps, ankle_positions=right_ankle_xyz)
	left_to_raw = detect_toe_offs(left_toe_xyz, timestamps, ankle_positions=left_ankle_xyz)
	right_to_raw = detect_toe_offs(right_toe_xyz, timestamps, ankle_positions=right_ankle_xyz)

	left_hs_alt, right_hs_alt, _ = enforce_alternation(left_hs_raw, right_hs_raw)
	left_to_alt, right_to_alt, _ = enforce_alternation(left_to_raw, right_to_raw)

	# Fallback to per-side robust detections if alternation becomes too sparse (e.g., one-side occlusion).
	left_hs = left_hs_alt if (len(left_hs_alt) + len(right_hs_alt)) >= 4 else left_hs_raw
	right_hs = right_hs_alt if (len(left_hs_alt) + len(right_hs_alt)) >= 4 else right_hs_raw
	left_to = left_to_alt if (len(left_to_alt) + len(right_to_alt)) >= 4 else left_to_raw
	right_to = right_to_alt if (len(left_to_alt) + len(right_to_alt)) >= 4 else right_to_raw

	left_stride_time = compute_stride_time(left_hs)
	right_stride_time = compute_stride_time(right_hs)
	stride_time = left_stride_time + right_stride_time

	left_stride_length = compute_stride_length(left_heel_xyz, left_hs)
	right_stride_length = compute_stride_length(right_heel_xyz, right_hs)
	stride_length = left_stride_length + right_stride_length
	stride_speed = compute_stride_speed(stride_length, stride_time)

	left_stance = compute_stance_time(left_hs, left_to)
	right_stance = compute_stance_time(right_hs, right_to)
	stance_time = left_stance + right_stance
	left_swing = compute_swing_time(left_hs, left_to)
	right_swing = compute_swing_time(right_hs, right_to)
	swing_time = left_swing + right_swing
	left_ratio = compute_stance_ratio(left_stance, left_stride_time)
	right_ratio = compute_stance_ratio(right_stance, right_stride_time)
	stance_ratio = left_ratio + right_ratio

	left_fpa = compute_foot_progression_angle(left_heel_xyz, left_toe_xyz, left_hs)
	right_fpa = compute_foot_progression_angle(right_heel_xyz, right_toe_xyz, right_hs)
	foot_progression_angles = left_fpa + right_fpa

	out['stride_time_mean'] = _mean_or_zero(stride_time)
	out['stride_time_cv'] = coefficient_of_variation(np.asarray(stride_time, dtype=float)) if stride_time else 0.0
	out['stride_length_mean'] = _mean_or_zero(stride_length)
	out['stride_length_cv'] = coefficient_of_variation(np.asarray(stride_length, dtype=float)) if stride_length else 0.0
	out['stride_speed_mean'] = _mean_or_zero(stride_speed)
	out['stance_time_mean'] = _mean_or_zero(stance_time)
	out['swing_time_mean'] = _mean_or_zero(swing_time)
	out['stance_ratio_mean'] = _mean_or_zero(stance_ratio)
	out['foot_progression_angle_mean'] = _mean_or_zero(foot_progression_angles)
	out['foot_progression_angle_std'] = float(np.std(np.asarray(foot_progression_angles, dtype=float))) if foot_progression_angles else 0.0

	# PKMAS-like spatial and COM dynamics summaries.
	step_len_mean, step_len_std = step_length_stats(left_ankle_xyz, right_ankle_xyz)
	out['step_length_mean'] = step_len_mean
	out['step_length_std'] = step_len_std
	out['step_length_cv'] = coefficient_of_variation(
		np.linalg.norm(left_ankle_xyz - right_ankle_xyz, axis=1)
	) if left_ankle_xyz.shape[0] and right_ankle_xyz.shape[0] else 0.0

	hip_l_xyz = _stack_xyz(lh_x, lh_y, lh_z)
	hip_r_xyz = _stack_xyz(rh_x, rh_y, rh_z)
	com_xyz = center_of_mass(hip_l_xyz, hip_r_xyz)
	com_v_mean, com_v_std = velocity_stats(com_xyz, sr=float(fps))
	com_a_mean, com_a_std = acceleration_stats(com_xyz, sr=float(fps))
	out['com_velocity_mean'] = com_v_mean
	out['com_velocity_std'] = com_v_std
	out['com_acceleration_mean'] = com_a_mean
	out['com_acceleration_std'] = com_a_std
	# Prefer log-compressed acceleration for training stability under heavy tails.
	out['com_acceleration_mean_log1p'] = float(np.log1p(max(0.0, com_a_mean)))
	out['com_acceleration_std_log1p'] = float(np.log1p(max(0.0, com_a_std)))

	# Double support approximation: stance window of 0.15s post-event
	stance_win = int(0.15 * fps) if fps > 0 else 0
	if stance_win > 0 and n > 0:
		left_stance = np.zeros(n, dtype=bool)
		right_stance = np.zeros(n, dtype=bool)
		for ev in left_events:
			end = min(n, ev + stance_win)
			left_stance[ev:end] = True
		for ev in right_events:
			end = min(n, ev + stance_win)
			right_stance[ev:end] = True
		double_support = left_stance & right_stance
		out['double_support_time_fraction'] = float(np.sum(double_support) / n)
	else:
		out['double_support_time_fraction'] = 0.0

	# Step length asymmetry: based on horizontal distances between consecutive events per foot
	def _compute_step_lengths(ankle_x: np.ndarray, events: np.ndarray) -> np.ndarray:
		"""Compute step lengths as horizontal (x) distances between consecutive events."""
		if events.size < 2:
			return np.array([])
		ankle_x = np.asarray(ankle_x, dtype=float)
		step_lengths = []
		for i in range(len(events) - 1):
			x_dist = abs(ankle_x[events[i+1]] - ankle_x[events[i]])
			step_lengths.append(x_dist)
		return np.asarray(step_lengths, dtype=float)
	
	left_step_lengths = _compute_step_lengths(la_x, left_events)
	right_step_lengths = _compute_step_lengths(ra_x, right_events)
	out['leg_step_length_asymmetry'] = leg_step_length_asymmetry(left_step_lengths, right_step_lengths)
	out['step_length_asymmetry_index'] = asymmetry_index(
		float(np.mean(left_step_lengths)) if left_step_lengths.size else 0.0,
		float(np.mean(right_step_lengths)) if right_step_lengths.size else 0.0,
	)

	# Joint angular-velocity summaries for gait coordination.
	left_shoulder_xyz = _stack_xyz(ls_x, ls_y, ls_z)
	right_shoulder_xyz = _stack_xyz(rs_x, rs_y, rs_z)
	left_elbow_xyz = _stack_xyz(le_x, le_y, le_z)
	right_elbow_xyz = _stack_xyz(re_x, re_y, re_z)
	left_wrist_xyz = _stack_xyz(lw_x, lw_y, lw_z)
	right_wrist_xyz = _stack_xyz(rw_x, rw_y, rw_z)
	left_hip_xyz = _stack_xyz(lh_x, lh_y, lh_z)
	right_hip_xyz = _stack_xyz(rh_x, rh_y, rh_z)
	left_knee_xyz = _stack_xyz(lk_x, lk_y, lk_z)
	right_knee_xyz = _stack_xyz(rk_x, rk_y, rk_z)

	elb_l_mean, elb_l_std = elbow_wrist_angular_velocity_stats(
		left_shoulder_xyz, left_elbow_xyz, left_wrist_xyz, sr=float(fps)
	)
	elb_r_mean, elb_r_std = elbow_wrist_angular_velocity_stats(
		right_shoulder_xyz, right_elbow_xyz, right_wrist_xyz, sr=float(fps)
	)
	hip_l_mean, hip_l_std = elbow_wrist_angular_velocity_stats(
		left_shoulder_xyz, left_hip_xyz, left_knee_xyz, sr=float(fps)
	)
	hip_r_mean, hip_r_std = elbow_wrist_angular_velocity_stats(
		right_shoulder_xyz, right_hip_xyz, right_knee_xyz, sr=float(fps)
	)
	knee_l_mean, knee_l_std = elbow_wrist_angular_velocity_stats(
		left_hip_xyz, left_knee_xyz, left_ankle_xyz, sr=float(fps)
	)
	knee_r_mean, knee_r_std = elbow_wrist_angular_velocity_stats(
		right_hip_xyz, right_knee_xyz, right_ankle_xyz, sr=float(fps)
	)
	out['elbow_angle_velocity_mean_left'] = elb_l_mean
	out['elbow_angle_velocity_std_left'] = elb_l_std
	out['elbow_angle_velocity_mean_right'] = elb_r_mean
	out['elbow_angle_velocity_std_right'] = elb_r_std
	out['hip_angle_velocity_mean_left'] = hip_l_mean
	out['hip_angle_velocity_std_left'] = hip_l_std
	out['hip_angle_velocity_mean_right'] = hip_r_mean
	out['hip_angle_velocity_std_right'] = hip_r_std
	out['knee_angle_velocity_mean_left'] = knee_l_mean
	out['knee_angle_velocity_std_left'] = knee_l_std
	out['knee_angle_velocity_mean_right'] = knee_r_mean
	out['knee_angle_velocity_std_right'] = knee_r_std
	out['hip_angle_velocity_asymmetry_index'] = asymmetry_index(hip_l_mean, hip_r_mean)
	out['knee_angle_velocity_asymmetry_index'] = asymmetry_index(knee_l_mean, knee_r_mean)

	# Balance features: head and trunk sway/tilt (FGA includes balance assessment)
	# Head position variability (use nose as proxy; MediaPipe landmark 0)
	NOSE = 0
	if NOSE < interp.shape[1]:
		nose_x = _np_clean1d(_get(NOSE, 0), method='forward_fill')
		nose_y = _np_clean1d(_get(NOSE, 1), method='forward_fill')
		if nose_x.size > 0:
			out['head_lateral_sway'] = float(np.std(nose_x))  # Lateral head movement
			out['head_vertical_sway'] = float(np.std(nose_y))  # Vertical head bobbing
			# Head path curvature: indicator of balance instability
			if nose_x.size > 1:
				head_path = np.sqrt(np.diff(nose_x)**2 + np.diff(nose_y)**2)
				out['head_path_length'] = float(np.sum(head_path))
	
	# Trunk tilt variability: use shoulder-hip angle as proxy for trunk lean
	# Compute trunk angle from shoulders to hips in sagittal plane (x-z)
	ls_x_clean = _np_clean1d(_get(L_SHOULDER, 0), method='forward_fill')
	lh_x_clean = _np_clean1d(_get(L_HIP, 0), method='forward_fill')
	ls_z_clean = _np_clean1d(_get(L_SHOULDER, 2), method='forward_fill')
	lh_z_clean = _np_clean1d(_get(L_HIP, 2), method='forward_fill')
	
	if ls_x_clean.size > 0 and lh_x_clean.size > 0 and min(len(ls_z_clean), len(lh_z_clean)) > 0:
		n_trunk = min(ls_x_clean.size, lh_x_clean.size, ls_z_clean.size, lh_z_clean.size)
		trunk_lean = np.arctan2(ls_z_clean[:n_trunk] - lh_z_clean[:n_trunk], 
								ls_x_clean[:n_trunk] - lh_x_clean[:n_trunk])
		out['trunk_tilt_mean'] = float(np.mean(trunk_lean))
		out['trunk_tilt_std'] = float(np.std(trunk_lean))  # Trunk sway variability
	
	# Turning features: head rotation speed (proxy for turning ability)
	# Compute head yaw from shoulder positions (horizontal angle)
	rs_x_clean = _np_clean1d(_get(R_SHOULDER, 0), method='forward_fill')
	if ls_x_clean.size > 0 and rs_x_clean.size > 0:
		n_head = min(ls_x_clean.size, rs_x_clean.size)
		head_yaw = np.arctan2(rs_x_clean[:n_head] - ls_x_clean[:n_head], 1.0)  # Shoulder separation -> yaw proxy
		if n_head > 1:
			head_yaw_velocity = np.abs(np.diff(head_yaw)) * fps
			out['head_rotation_speed_mean'] = float(np.mean(head_yaw_velocity))
			out['head_rotation_speed_max'] = float(np.max(head_yaw_velocity))
			# Turning smoothness: low variation in rotation speed indicates coordinated turning
			out['head_rotation_variability'] = float(np.std(head_yaw_velocity) / (np.mean(head_yaw_velocity) + 1e-6))
	
	# Stepping asymmetry during turns: if head rotation correlates with asymmetric stepping
	if left_events.size > 0 and right_events.size > 0:
		left_step_count = float(left_events.size)
		right_step_count = float(right_events.size)
		if left_step_count + right_step_count > 0:
			stepping_symmetry = 1.0 - abs(left_step_count - right_step_count) / (left_step_count + right_step_count)
			out['stepping_symmetry_ratio'] = stepping_symmetry

	return out


