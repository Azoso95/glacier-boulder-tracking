"""Metric reprojection and broad downslope fields for the study DEMs."""

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import gaussian_filter


def utm_crs_for_lonlat(lon, lat):
    """CRS UTM (WGS84) piu' vicino al punto lon/lat, per calcoli metrici."""
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def compute_flow_direction(dem_path, sigma_px=10):
    """
    sigma_px: forza dello smoothing gaussiano applicato all'elevazione, in
    pixel DEM. Con DEM a 30 m, sigma_px=10 equivale a smussare su una scala
    di circa 300 m: una direzione di flusso "macro", non locale al singolo
    masso. Aumentare per un dato ancora piu' lasco, diminuire per seguire
    meglio la curvatura del ghiacciaio.
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        width, height = src.width, src.height

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    if np.isnan(dem).any():
        dem = np.where(np.isnan(dem), np.nanmean(dem), dem)

    # Il DEM Copernicus e' in coordinate geografiche (gradi): transform.a/e
    # sono un passo pixel in gradi (~0.00027 per 30 m). Calcolare il
    # gradiente direttamente su quello divide un dislivello in METRI per un
    # passo in GRADI, gonfiando la pendenza di ~4 ordini di grandezza (quasi
    # sempre satura vicino a 90 gradi, anche su aree pianeggianti). Si
    # riproietta quindi in UTM (metri) prima di smussare e derivare.
    if crs.is_geographic:
        lon_c = (bounds.left + bounds.right) / 2
        lat_c = (bounds.top + bounds.bottom) / 2
        dst_crs = utm_crs_for_lonlat(lon_c, lat_c)
        meters_per_deg_lon = 111_320 * np.cos(np.radians(lat_c))
        approx_res_m = abs(transform.a) * meters_per_deg_lon

        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, dst_crs, width, height, *bounds, resolution=approx_res_m
        )
        dst_dem = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=dem,
            destination=dst_dem,
            src_transform=transform,
            src_crs=crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        if np.isnan(dst_dem).any():
            dst_dem = np.where(np.isnan(dst_dem), np.nanmean(dst_dem), dst_dem)

        dem = dst_dem
        transform = dst_transform
        crs = dst_crs

    dem_smooth = gaussian_filter(dem, sigma=sigma_px)

    px_size_x = transform.a
    px_size_y = -transform.e  # transform.e e' negativo

    dz_dy, dz_dx = np.gradient(dem_smooth, px_size_y, px_size_x)

    # direzione di flusso = verso il basso lungo il gradiente (downslope)
    flow_dx = -dz_dx
    flow_dy = -dz_dy

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    direction_rad = np.arctan2(flow_dy, flow_dx)
    sin_dir = np.sin(direction_rad).astype(np.float32)
    cos_dir = np.cos(direction_rad).astype(np.float32)

    return slope_deg, sin_dir, cos_dir, transform, crs


def native_res_transform(src_transform, src_crs, dst_crs, width, height):
    """
    Transform/shape di destinazione per riproiettare da src_crs a dst_crs
    mantenendo la risoluzione nativa della sorgente (nessun ricampionamento
    sul pixel-grid dell'ortofoto): serve solo a portare i dati nello stesso
    CRS, il file resta piccolo quanto la sorgente invece di gonfiarsi alla
    risoluzione dell'ortofoto.
    """
    left = src_transform.c
    top = src_transform.f
    right = left + width * src_transform.a
    bottom = top + height * src_transform.e
    return calculate_default_transform(src_crs, dst_crs, width, height, left, bottom, right, top)


def clip_dem_to_orthophoto(dem_path, orthophoto_path, out_path):
    """
    Il DEM scaricato copre bbox ortofoto + buffer (serve per lo smoothing
    senza artefatti di bordo in compute_flow_direction). Questa funzione
    produce in piu' un DEM riproiettato nel CRS dell'ortofoto, a risoluzione
    nativa (non ricampionato sul pixel-grid dell'ortofoto): e' gia'
    georeferenziato correttamente, letture per finestra/coordinate
    funzionano senza bisogno di un file enorme.
    """
    with rasterio.open(orthophoto_path) as ortho:
        dst_crs = ortho.crs

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        src_height, src_width = src.height, src.width

    if src_nodata is not None:
        dem = np.where(dem == src_nodata, np.nan, dem)

    dst_transform, dst_width, dst_height = native_res_transform(
        src_transform, src_crs, dst_crs, src_width, src_height
    )

    dst_dem = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=dem,
        destination=dst_dem,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    out_profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(dst_dem, 1)

    print(f"DEM riproiettato (risoluzione nativa) salvato in: {out_path}")


def align_to_orthophoto(bands, src_transform, src_crs, orthophoto_path, out_path):
    with rasterio.open(orthophoto_path) as ortho:
        dst_crs = ortho.crs

    height, width = bands[0].shape
    dst_transform, dst_width, dst_height = native_res_transform(
        src_transform, src_crs, dst_crs, width, height
    )

    out_profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": len(bands),
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **out_profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst_band = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=band,
                destination=dst_band,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            dst.write(dst_band, i)

    print(f"Raster di direzione di flusso riproiettato (risoluzione nativa) salvato in: {out_path}")
