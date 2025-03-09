import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,  squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score  # Tambahkan ini
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import os
import seaborn as sns
from matplotlib import cm

wd = "C:/WIWIN/UNHAS/PSHA/BuatPAPERDewi/segmenYopi/b-valueMap/Palukoro-Matano/"
wd1 = "C:/WIWIN/UNHAS/PSHA/BuatPAPERDewi/segmenYopi/"
cat_file = "(WardMethod)Cluster_10_1.4.xlsx"  # Ganti dengan nama file Excel yang diinginkan
sheet_name = "Sebelum"

# Baca katalog dari file Excel
cat = pd.read_excel(wd + cat_file, sheet_name=sheet_name)

# Filter data berdasarkan magnitude
cat_filtered = cat[cat['magnitude'] >= 3.2]

# Menggunakan data lat dan lon dari catatan
lat = cat_filtered['latitude']
lon = cat_filtered['longitude']
depth = cat['depth'] 
X = np.column_stack((lat, lon))  # Menggabungkan lat dan lon ke dalam satu array

# Menghitung matriks jarak menggunakan metode Euclidean
dist_matrix = pdist(X, metric='euclidean')

# Membuat dendrogram menggunakan metode linkage ward
linkage_matrix = linkage(dist_matrix, method='ward')
# Convert distance matrix to a square form for clustering
dist_matrix_square = squareform(dist_matrix)

# Create a DataFrame for the distance matrix
dist_df = pd.DataFrame(dist_matrix_square)

# Create a clustermap with seaborn
#clustermap=sns.clustermap(dist_df, method='ward', cmap='hot', figsize=(10, 10))
# Membuat clustermap dengan dendrogram yang sesuai
clustermap = sns.clustermap(dist_df, row_linkage=linkage_matrix, col_linkage=linkage_matrix, method='ward', cmap='hot', figsize=(10, 10))


# output_file = os.path.join(wd1, "SimilarityMatrixComplete.png")
# plt.savefig(output_file, dpi=300, bbox_inches='tight')

# Menambahkan judul
# clustermap.fig.suptitle('Metode Linkage = Ward', fontsize=16)

Threshold= 10
#Menampilkan dendrogram
plt.figure(figsize=(12, 6))
dendro=dendrogram(linkage_matrix)
plt.axhline(y=Threshold, color='r', linestyle='--', label="threshold")

plt.xlabel('Data Points',fontsize=20)
plt.ylabel('Distance',fontsize=20)
plt.title(f'Dendogram with Threshold:{Threshold}',fontsize=20)


plt.xticks(fontsize=20)  # Memperbesar ukuran angka di sumbu x
plt.yticks(fontsize=20)  # Memperbesar ukuran angka di sumbu y
plt.legend(fontsize="20")
plt.show()

##############################################################################################################

# Tentukan jumlah cluster yang diinginkan berdasarkan threshold
clusters = fcluster(linkage_matrix, Threshold, criterion='distance')
num_clusters = len(np.unique(clusters))

# Menampilkan dendrogram dengan warna sesuai jumlah cluster
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, color_threshold=Threshold, 
            above_threshold_color='gray', 
            truncate_mode='lastp', p=num_clusters,
            show_contracted=True)

plt.axhline(y=Threshold, color='r', linestyle='--',label="threshold")
plt.xlabel('Data Points',fontsize=20)
plt.ylabel('Distance',fontsize=20)
plt.title(f'Dendogram with {num_clusters} Clusters (Threshold: {Threshold})',fontsize=20)
plt.xticks(fontsize=20)  # Memperbesar ukuran angka di sumbu x
plt.yticks(fontsize=20)  # Memperbesar ukuran angka di sumbu y
plt.legend(fontsize="20")
plt.show()

#-------------------------------------------------------------------------------------------------

# Membuat objek AgglomerativeClustering dengan jumlah cluster yang diinginkan
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=Threshold, linkage='ward')

# Melakukan klasterisasi
labels = clustering.fit_predict(X)

# Menghitung Silhouette Score
silhouette_avg = silhouette_score(X, labels)

# Latitude and longitude limits
lat_min, lat_max = -7.5, 3
lon_min, lon_max = 117, 127

# Baca file GeoJSON
gdf_faults = gpd.read_file('IndonesiaFaults.geojson')

# Plot the data
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
# Menghitung centroid untuk setiap cluster
centroids = []
for label in np.unique(labels):
    cluster_points = X[labels == label]
    centroid = np.mean(cluster_points, axis=0)
    centroids.append((label, centroid))


# Plot centroids pada peta dengan nomor masing-masing cluster
for label, centroid in centroids:
    plt.text(centroid[1], centroid[0], str(label+1), color='blue', fontsize=12, ha='center', transform=ccrs.PlateCarree(), zorder=6)

# Konversi centroids ke dalam array numpy untuk scatter plot
centroid_coords = np.array([centroid for _, centroid in centroids])


# Custom coastline with increased linewidth
coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black', facecolor='none', linewidth=0.5)

ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=0)
ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
ax.add_feature(coastline, zorder=3)
ax.gridlines(draw_labels=True, zorder=4)

# Plot fault lines from GeoJSON
gdf_faults.plot(ax=ax, edgecolor='red', linewidth=1, transform=ccrs.PlateCarree(), zorder=5 ,label='Fault')

#============== PLOT POLYGON YOPI ===================
from shapely.geometry import Point, Polygon
# Fungsi untuk mengubah string koordinat menjadi Polygon
def parse_coordinates(coord_str):
    coord_list = [
        tuple(map(float, pair.strip("()").split(",")))
        for pair in coord_str.split("),(")
    ]
    return Polygon(coord_list)

# File dan sheet name
wd = "C:/WIWIN/UNHAS/PSHA/BuatPAPERDewi/"
wd1 = "C:/WIWIN/UNHAS/PSHA/BuatPAPERDewi/segmenYopi/"
cat_file = "SulawesiData.xlsx"
region_sheet = "Region"
data_sheet = "Combine"

# Baca data dari Excel
region_df = pd.read_excel(f"{wd}{cat_file}", sheet_name=region_sheet)
#data_df = pd.read_excel(f"{wd}{cat_file}", sheet_name=data_sheet)

# Buat dictionary untuk menyimpan poligon
regions = {}
for _, row in region_df.iterrows():
    region_name = row['RegionName']  # Sesuaikan dengan kolom yang relevan
    coords = row['Coordinates']  # Kolom koordinat dalam format string
    regions[region_name] = parse_coordinates(coords)
    
import matplotlib.colors as mcolors
# Generate colors for each region polygon
cmap = plt.get_cmap('Set1')  # You can change the colormap here
color_map = mcolors.ListedColormap(cmap.colors[:len(regions)])  # Adjusting to the number of regions

# Plot polygons with different colors
for i, (region_name, polygon) in enumerate(regions.items()):
    ax.plot(*polygon.exterior.xy,color="gray", alpha=0.5, zorder=4, linewidth=7) 
    centroid = polygon.centroid
    #ax.text(centroid.x, centroid.y, region_name,fontsize=12, ha='center', color='white', zorder=5)
#============== BATAS BAWAH !!!! PLOT POLYGON YOPI ================

import rasterio
from rasterio.plot import show
from matplotlib.lines import Line2D

def interpolate_coords(coords, num_points):
    """
    Interpolates a list of coordinates to have a specified number of points.
    """
    coords = np.array(coords)
    distance = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    distance = np.insert(distance, 0, 0).cumsum()
    fx = np.interp(np.linspace(0, distance[-1], num_points), distance, coords[:, 0])
    fy = np.interp(np.linspace(0, distance[-1], num_points), distance, coords[:, 1])
    return np.column_stack((fx, fy))

# Filter out filled markers and marker settings that do nothing.
unfilled_markers = [m for m, func in Line2D.markers.items()
                    if func != 'nothing' and m not in Line2D.filled_markers]

# Choose the 20th marker
marker_20 = unfilled_markers[21]#20
                  #[119.33232, 1.62531], 
custom_coords = [[119.41092, 1.79997], [119.58559, 1.92224], [119.79519, 2.04451], 
                  [120.09213, 2.07944], [120.51133, 2.09691], [120.87813, 2.18424], [121.41087, 2.08817], 
                  [121.97854, 2.11437], [122.35407, 2.07944], [123.26234, 1.81744], [123.74497, 1.69827], 
                  [124.19682, 1.56417]]

custom_coords_lat = [coord[1] for coord in custom_coords]
custom_coords_lon = [coord[0] for coord in custom_coords]

marker_16 = unfilled_markers[16]
custom_coords1 = [[124.8446, 0.2717], [125.1039, 0.6073], [125.3937, 0.9886], [125.5309, 1.2174], 
                 [125.7292, 2.1325], [125.958, 2.9867], [126.3241, 3.7341], [126.5376, 4.3747], 
                 [126.6444, 4.9848]]

custom_coords_lat1 = [coord[1] for coord in custom_coords1]
custom_coords_lon1 = [coord[0] for coord in custom_coords1]

marker_18 = unfilled_markers[19] #18
custom_coords2 = [[127.4872, 5.4651], [127.4109, 4.5194], [127.4567, 3.9703], [127.6092, 3.2839], 
                 [127.5635, 2.5823], [127.3499, 1.8502], [126.9839, 1.1485], [126.7398, 0.1876], 
                 [126.8161, -0.4683], [127.0296, -0.8953]]

custom_coords_lat2 = [coord[1] for coord in custom_coords2]
custom_coords_lon2 = [coord[0] for coord in custom_coords2]

# Interpolate coordinates to have 10 points each
num_points_list = [10, 10, 10, 4] 
interp_coords = interpolate_coords(custom_coords, num_points_list[0])
interp_coords1 = interpolate_coords(custom_coords1, num_points_list[1])
interp_coords2 = interpolate_coords(custom_coords2, num_points_list[2])

# Plotting
plt.plot(interp_coords[:, 0], interp_coords[:, 1],  color='red', marker=marker_20, linewidth=1, markersize=8, markeredgewidth=1, zorder=5, label='Subduction')
#plt.plot(interp_coords1[:, 0], interp_coords1[:, 1],  color='red', marker=marker_16, linewidth=1, markersize=8, markeredgewidth=1, zorder=5)
plt.plot(interp_coords2[:, 0], interp_coords2[:, 1],  color='red', marker=marker_18, linewidth=1, markersize=8, markeredgewidth=1, zorder=5)

# Tentukan palet warna dan buat array warna yang unik
num_clusters = len(np.unique(labels))
colormap = cm.get_cmap('tab20b', num_clusters)  # Menggunakan palet warna 'tab20' dengan variasi warna

for label in np.unique(labels):
    cluster_color = colormap(label)
    plt.scatter(lon[labels == label], lat[labels == label], label=f'Cluster {label+1}',color=cluster_color,edgecolor="black",linewidth=0.5, transform=ccrs.PlateCarree())

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Hierarchy Clustering (Ward Method)\nThreshold: {Threshold}\nSilhouette Score: {silhouette_avg}')  # Tambahkan nilai Silhouette Score di judul

# Letakkan legenda di bawah plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

plt.show()

# Membagi data berdasarkan label klaster
clustered_data = {}
for label in np.unique(labels):
    clustered_data[label] = cat_filtered.iloc[labels == label]



#Simpan setiap bagian data ke dalam file Excel terpisah
# for label, data in clustered_data.items():
#     file_name = f"(WardMethod)Cluster_{label + 1}_3.xlsx"
#     data.to_excel(wd1 + file_name, index=False)
#     print(f"Data untuk Cluster {label + 1} tersimpan dalam file {file_name}")

# output_file = os.path.join(wd1, "Region2.png")
# plt.savefig(output_file, dpi=300, bbox_inches='tight')