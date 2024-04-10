from osgeo import gdal

image_path = r"J:\GF5B_AHSI_W104.1_N32.8_20220209_002267_L10000074984\result\GF5B_AHSI_W104.1_N32.8_20220209_002267_L10000074984_SW.tif"
dataset = gdal.Open(image_path)

if dataset.GetMetadata('RPC') is None:
    print("RPC file is not found.")
else:
    corrected_image_path = image_path.replace('.tif','_corrected.tif')
    warp_options = gdal.WarpOptions(rpc=True)
    corrected_dataset = gdal.Warp(corrected_image_path,dataset,options=warp_options)
    corrected_dataset = None
    dataset = None
    print("校正完成")

