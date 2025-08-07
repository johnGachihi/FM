# ------------------------------------------------------------
# Author: Benson Kenduiywo (Translated to R)
# Assess accuracy of classification using unseen validation polygons
# ------------------------------------------------------------

library(terra)
library(sf)
library(dplyr)
library(foreach)
library(doParallel)
library(yardstick)
library(readxl)
library(openxlsx)

# ---------- 1. Paths -------------------------------------------------
threshold <- 0.6
map_path <- '/cluster/archiving/GIZ/maps/'
outpath = '/cluster/archiving/GIZ/accuracy/'
name_end <- 'with_2025'
file_ending = paste0('merged_probs_', name_end)
out_xlsx = paste0(outpath,'Metrics_',file_ending,'_threshold_',threshold,'.xlsx')
root = '/cluster/archiving/GIZ/data/'
districts <- c("Ruhango", "Nyagatare", "Musanze", "Nyabihu")
season <- "B"
eyear <- 2025
nodata_val <- 255
classnames <- c('Bean', 'Irish Potato', 'Maize', 'Rice')
nclasses <- 4

raster_paths <- file.path(root, "outputs", paste0(districts, "_", season, eyear, "_", file_ending,".tif"))
vector_path <- file.path(root, "shapefiles", paste0("RWA_", season, eyear, "_Merge_v1_ValidSet.shp"))

# ---------- 2. Merge Rasters -----------------------------------------
rasters <- lapply(raster_paths, rast)
merged_probs <- do.call(merge, rasters)
names(merged_probs) <- classnames
raster_crs <- crs(merged_probs)


# ---------- 2. Determine wining class -----------------------------------------
# Step 1: Compute the max probability per pixel
max_prob <- app(merged_probs, fun = max)

# Step 2: Get class index (argmax) per pixel
predicted_class <- which.max(merged_probs) - 1  # class labels from 0

# Step 3: Apply threshold: if max prob < threshold, assign 255
final_class <- classify(c(predicted_class, max_prob), 
                        rcl = matrix(c(-Inf, threshold, NA,  # mask out low confidence
                                       threshold, Inf, NA), 
                                     ncol=3, byrow=TRUE),
                        include.lowest=TRUE, others=TRUE)

# Combine result: use mask to assign 255 where max_prob < threshold
labels <- mask(predicted_class, max_prob < threshold, maskvalue=TRUE)
names(labels) <- 'class'
# Save labels to disk
writeRaster(labels, paste0(map_path,'RWA_',season,eyear,'_threshold_',threshold,'_',name_end,'.tif'), overwrite = TRUE,
            wopt = list(datatype = "INT1U", gdal = c("COMPRESS=DEFLATE")))


# ---------- 3. Read Vector Data and Reproject ------------------------
ncores <- max(1, parallel::detectCores() - 40)
polys <- vect(vector_path)
#polys <- polys[polys$code != nodata_val, ]
polys <- project(polys, raster_crs)  
polys <- polys[!is.na(geomtype(polys)), ] # drop missing geometries
# ---------- 4. Parallel Pixel Extraction -----------------------------

start_time <- Sys.time() 
cl <- makeCluster(ncores)
registerDoParallel(cl)

# Function to extract pixel values inside each polygon
extract_pixels <- function(geom, ref) {
  mask <- terra::mask(labels, vect(geom))
  values <- terra::values(mask, na.rm = TRUE)
  values <- values[!is.na(values) & values != nodata_val]
  data.frame(true = rep(ref, length(values)), pred = values)
}

# Export necessary objects to each worker
clusterExport(cl, varlist = c("labels", "nodata_val", "extract_pixels"))

# Run parallel extraction
results <- foreach(i = 1:nrow(polys),
                   .combine = rbind,
                   .packages = c("terra", "sf", "dplyr")) %dopar% {
                     extract_pixels(polys[i, ], as.integer(polys$code[i]))
                   }

end_time <- Sys.time()  # Record end time
# Compute time difference in minutes and hours
time_taken <- difftime(end_time, start_time, units = "mins") 
stopCluster(cl)
#######################

df <- terra::extract(labels, polys, parallel=ncores)

# ---------- 5. Confusion Matrix & Metrics ----------------------------
results <- na.omit(results)
classes <- sort(unique(c(results$true, results$pred)))
results$true <- factor(results$true, levels = classes)
results$pred <- factor(results$pred, levels = classes)

conf_mat <- table(results$true, results$pred)
acc_metrics <- yardstick::metrics(results, truth = true, estimate = pred)
f1_scores <- yardstick::f_meas(results, truth = true, estimate = pred)

# Class-wise stats
per_class <- data.frame(
  Class = classes,
  ProducerAcc = diag(conf_mat) / colSums(conf_mat),
  UserAcc = diag(conf_mat) / rowSums(conf_mat),
  F1_score = f1_scores$.estimate
)

# Overall stats
overall <- data.frame(
  Overall_accuracy = sum(diag(conf_mat)) / sum(conf_mat),
  Kappa = yardstick::kap(results, truth = true, estimate = pred)$.estimate
)

# ---------- 6. Export to Excel ---------------------------------------
wb <- createWorkbook()
addWorksheet(wb, "ConfusionMatrix")
writeData(wb, "ConfusionMatrix", as.data.frame.matrix(conf_mat), rowNames = TRUE)

addWorksheet(wb, "PerClass")
writeData(wb, "PerClass", per_class)

addWorksheet(wb, "Overall")
writeData(wb, "Overall", overall)

saveWorkbook(wb, out_xlsx, overwrite = TRUE)
cat("✅ Results saved to:", out_xlsx, "\n")
