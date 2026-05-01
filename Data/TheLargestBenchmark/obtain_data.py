import os
import numpy as np
import pandas as pd

# please set the file path
file_path = ''

# --- metadata / region filter ---
ca_meta = pd.read_csv(os.path.join(file_path, 'ca_meta.csv'))
# gba_meta = ca_meta[ca_meta.District == 4].reset_index(drop=True)
gba_meta = ca_meta
gba_meta.to_csv(os.path.join(file_path, 'gba_meta.csv'), index=False)
print('duplicate lat/lng rows in gba_meta:')
print(gba_meta[gba_meta.duplicated(subset=['Lat', 'Lng'])])

# --- adjacency extraction for GBA (still kept for reference / other uses) ---
gba_meta_id2 = gba_meta.ID2.values.tolist()
print('gba node count (by ID2):', len(gba_meta_id2))

ca_rn_adj_path = os.path.join(file_path, 'ca_rn_adj.npy')
ca_rn_adj = np.load(ca_rn_adj_path)
print('ca_rn_adj shape:', ca_rn_adj.shape)
gba_rn_adj = ca_rn_adj[gba_meta_id2]
gba_rn_adj = gba_rn_adj[:, gba_meta_id2]
print('gba_rn_adj shape:', gba_rn_adj.shape)
np.save(os.path.join(file_path, 'gba_rn_adj.npy'), gba_rn_adj)

# --- distance-based node subset: every pairwise distance > threshold ---
# We compute pairwise great-circle (Haversine) distances between detectors using their latitude and longitude.
# Then we greedily select a subset of detectors such that any two selected detectors are more than
# `threshold_miles` apart. This means all retained detectors are mutually well-separated.
threshold_miles = 5  # distance threshold in miles

lat_rad = np.radians(gba_meta['Lat'].values)
lng_rad = np.radians(gba_meta['Lng'].values)

# Use broadcasting to form all pairwise differences
lat1 = lat_rad[:, None]
lat2 = lat_rad[None, :]
dlng = lng_rad[None, :] - lng_rad[:, None]
dlat = lat2 - lat1

# Haversine formula (radius of Earth in miles)
R_miles = 3958.8
a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance_matrix = R_miles * c  # shape (N, N)

N = distance_matrix.shape[0]

# too_close[i, j] = True means detectors i and j are within threshold (and cannot co-exist)
too_close = (distance_matrix < threshold_miles)
np.fill_diagonal(too_close, False)

remaining = np.ones(N, dtype=bool)
selected_indices = []

for i in range(N):
	if not remaining[i]:
		continue
	# Select this detector
	selected_indices.append(i)
	# Remove all detectors that are too close to i (including i itself)
	conflict = too_close[i] | (np.arange(N) == i)
	remaining[conflict] = False

selected_indices = np.array(selected_indices, dtype=int)
kept_node_count = selected_indices.size
print(f'selected node count with all pairwise distances > {threshold_miles} miles: {kept_node_count} / {N}')

# Optionally save selected node IDs
selected_ids = [gba_meta.ID.iloc[idx] for idx in selected_indices]
selected_id2 = [gba_meta_id2[idx] for idx in selected_indices]
selected_df = pd.DataFrame({
	'ID': selected_ids,
	'ID2': selected_id2
})
selected_out_path = os.path.join(file_path, f'gba_nodes_pairwise_over_{threshold_miles}mi.csv')
selected_df.to_csv(selected_out_path, index=False)
print('saved selected node IDs to:', selected_out_path)
# exit()
# --- prepare ID list for selecting historical data ---
gba_meta = gba_meta.iloc[selected_indices].reset_index(drop=True)  # Restrict meta information to the selected, well-separated detectors only.
gba_meta.ID = gba_meta.ID.astype(str)
gba_meta_id = gba_meta.ID.values.tolist()

# --- process years 2017-2021, save per-year HDF and CSV, then combine ---
years = range(2017, 2022)
processed = []
for y in years:
	y_str = str(y)
	in_h5 = os.path.join(file_path, f'ca_his_raw_{y_str}.h5')
	ca_his = pd.read_hdf(in_h5)
	# Ensure the index is a datetime index and exposed as a 'date' column for downstream loaders.
	if not isinstance(ca_his.index, pd.DatetimeIndex):
		try:
			ca_his.index = pd.to_datetime(ca_his.index)
		except Exception:
			pass
	# Select only the (distance-filtered) GBA detector columns.
	gba_his = ca_his[gba_meta_id]
	# Add explicit 'date' column from the datetime index so later code can use it.
	gba_his = gba_his.copy()
	gba_his['date'] = gba_his.index
	out_h5 = os.path.join(file_path, f'gba_his_{y_str}.h5')
	out_csv = os.path.join(file_path, f'gba_his_{y_str}.csv')

	# Save per-year data including the 'date' column.
	gba_his.to_hdf(out_h5, key='t', mode='w')
	gba_his.to_csv(out_csv, index=False)  # index is now in 'date'
	print(f'saved year {y_str}: {out_h5} and {out_csv}, shape={gba_his.shape}')

	processed.append(gba_his)

if processed:
	# Combine along time index (rows). Drop duplicated timestamps if any.
	combined = pd.concat(processed)
	# After per-year step, 'date' is a column; keep it as the **first** column.
	# Ensure no duplicated timestamps based on 'date'.
	combined = combined.drop_duplicates(subset=['date'])
	# Reorder columns so that 'date' is the first column, followed by all feature columns.
	feature_cols = [c for c in combined.columns if c != 'date']
	combined = combined[['date'] + feature_cols]

	out_combined_csv = os.path.join(file_path, 'traffic.csv')
	out_combined_h5 = os.path.join(file_path, 'gba_his_2017-2021.h5')

	# --- missing value handling & optional normalization on features ---
	# We only operate on feature columns (exclude 'date'). First, handle NaNs by
	# linear interpolation over time (using 'date' ordering), then fill any
	# remaining NaNs with forward/backward fill.

	# Work on a copy to avoid modifying the original reference unexpectedly
	features = combined[feature_cols].copy()

	# Sort by date to ensure correct temporal order
	combined = combined.sort_values('date')
	features = combined[feature_cols].copy()

	# Interpolate missing values column-wise (linear in index order)
	features = features.interpolate(method='linear', limit_direction='both', axis=0)
	# Fill any leading/trailing NaNs
	features = features.fillna(method='ffill').fillna(method='bfill')

	# Optional: standardize features to zero mean / unit variance
	# (comment out this block if you prefer raw values)
	feature_means = features.mean()
	feature_stds = features.std().replace(0, 1.0)
	features_norm = (features - feature_means) / feature_stds
	# features_norm = features - feature_means

	# Reassemble combined dataframe with date as the first column
	combined_norm = pd.concat([combined[['date']].reset_index(drop=True),
	                         features_norm.reset_index(drop=True)], axis=1)

	try:
		# Save with 'date' as the **first column**, no index in CSV.
		combined_norm.to_csv(out_combined_csv, index=False)
		combined_norm.set_index('date').to_hdf(out_combined_h5, key='t', mode='w')
		print('saved combined (normalized) file:', out_combined_csv, 'shape=', combined_norm.shape)
		print('feature column count (excluding date):', len(feature_cols))
	except Exception as e:
		print('failed to save combined outputs:', e)
else:
	print('no yearly data processed; combined file not created')