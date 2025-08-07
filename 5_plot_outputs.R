library(terra)
threshold <- 0.7
district = 'Nyabihu'
season = 'B'
eyear = '2025'
nodata = 255
nclasses = 4
path <- '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/data/outputs/' 
root <- '/home/bkenduiywo/GFM_Galileo/results/' 
file_ending <- '_merged_masked_labels_without2025.tif'
generateLabels = FALSE 
filename = paste0(path,district,'_',season,eyear,file_ending)
#labels <- rast(paste0(path,district,'_',season,eyear,'_merged_masked_probs.tif')) #/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/data/outputs/Nyagatare_B2025_merged_masked_labels_with2025.tif
if(generateLabels==TRUE){
  probs <- rast(filename) #Nyagatare_B2025_merged_masked_labels_with2025
  out_label <- sprintf("%s%s_%s%s_merged_masked_labels_%.2f.tif",
                       path, district, season, eyear, threshold)
  ###Apply probability threshold to class probability
  if (nlyr(probs) != nclasses) stop(sprintf("Expected %d probability bands (classes 1–%d)", nclasses, nclasses))
  
  # Compute per-pixel max prob and winning class
  winner_class <- which.max(probs)    # integer 1–4
  winner_prob  <- max(probs)          # float
  
  # Initialize label map as NA
  labels <- winner_class
  values(labels) <- NA                # all NA initially
  
  # Keep only where max prob ≥ threshold AND winner_prob is not NA
  valid_mask <- !is.na(winner_prob) & (winner_prob >= threshold)
  labels[valid_mask] <- winner_class[valid_mask]
  
  # Set NoData value explicitly (for saving)
  NAflag(labels) <- nodata
  # Write to disk
  writeRaster(
    labels,
    filename = out_label,
    datatype = "INT1U",
    gdal = c("COMPRESS=LZW"),
    overwrite = TRUE
  )
  
  cat("[✓] Land-cover map saved to:", out_label, "\n")
  ######
  
}else{
  labels <- rast(filename) 
  NAflag(labels) <- nodata
}



#==============================================================
# Display map and compute area
#=============================================================
freq(labels)
classnames <- c('Bean', 'Irish Potato', 'Maize', 'Rice')  # 0-3 and 255
class_colors <- c("#55FF00","#732600", "#FFD400" , "#00A9E6")  # match exactly #9ACD32
# Display function


display <- function(map, method){
  par(mar = c(7, 2, 1.6, 6)) #c(bottom, left, top, right)
  image(map, col = class_colors, axes = TRUE, ann = FALSE)
  #add_legend("bottomright", legend = classnames, fill = class_colors, ncol = 3, bty = 'n', cex = 1.1)
  legend("topright", legend = classnames, fill = class_colors, ncol = 1, bty = 'n', cex = 1.1)
  title(paste0(method, " classification"))
}
output_file <- paste0(root,district,'_',season,eyear,'_merged_labels_unmasked.png')
png(filename = output_file, width = 8.2, height = 6.6, units = "in", res = 300)
display(labels, "Finetuned Galileo GFM")
dev.off()

######Display 2: OPTIONAL 
library(tmap)
map_raster <- raster::raster(labels)  # Convert from SpatRaster to RasterLayer
# Assign labels to raster values (assuming 0:4 = classes)
levels(map_raster) <- data.frame(ID = 0:4, class = classnames)
#output_file <- paste0(root, district, '_', season, eyear, '_tmapped_labels.png')

tmap_mode("plot")  # use static map rendering

tm <- tm_shape(map_raster) +
  tm_raster(style = "cat", palette = class_colors, labels = classnames, title = "Class") +
  tm_layout(
    title = "Finetuned Galileo GFM Classification",
    title.size = 1.2,
    title.position = c("center", "top"),
    legend.outside = TRUE,
    frame = TRUE,
    legend.title.size = 1,
    legend.text.size = 0.9
  ) +
  tm_scale_bar(position = c("left", "bottom")) +
  tm_compass(type = "arrow", position = c("left", "top"))

# Save as high-resolution PNG
tmap_save(tm, filename = output_file, width = 8.2, height = 6.6, units = "in", dpi = 300)


#####
# -------------------- compute per‑class area --------------------
labels_factor <- as.factor(labels)
area_ha <-  expanse(labels_factor, unit = "ha", byValue = TRUE)
area_km2 <- expanse(labels_factor, unit = "km", byValue = TRUE)
# -------------------- tidy result --------------------
df_ha <- as.data.frame(area_ha, xy = FALSE, cells = FALSE)
df_km2 <- as.data.frame(area_km2, xy = FALSE, cells = FALSE)
area_df <- merge(df_ha[,c('value', 'area')], df_km2[,c('value', 'area')], by = "value")
colnames(area_df) <- c("code", "area_ha", "area_km2")
classnames <- c("Bean", "Irish Potato", "Maize", "Rice")
area_df$code <- as.numeric(area_df$code)
area_df$crop_type <- classnames[area_df$code ]
print(area_df)




