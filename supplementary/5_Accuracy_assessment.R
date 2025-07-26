## -----------------------------------------------------------
## 1.  Libraries  (only spatial & accuracy utilities)
## -----------------------------------------------------------
rm(list=ls(all=TRUE))
library(terra)          # raster handling
library(sf)             # vector (shapefile)
library(caret)          # confusionMatrix() for convenience

## -----------------------------------------------------------
## 2.  Read raster & validation polygons
## -----------------------------------------------------------
threshold = 0.0
district = 'Nyagatare'
season = 'B'
eyear = '2025'
nodata = 255
nclasses = 5
path <- '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/data/'
root <- '/home/bkenduiywo/GFM_Galileo/results/' 
pfilename <- paste0(path,'outputs/',district,'_',season,eyear,'_merged_probs_with2025.tif')
probs <- rast(pfilename) 
#ofilename <- tools::file_path_sans_ext(basename(pfilename))
#out_label <- sprintf("%s%s_%s%s_merged_masked_labels_%.2f.tif", path, district, season, eyear, threshold)
shp_path <- paste0(path,'shapefiles/RWA_B2025_Merge_v1_ValidSet.shp') 

###Apply probability threshold to class probability
if (nlyr(probs) != nclasses) stop(sprintf("Expected %d probability bands (classes 1–%d)", nclasses, nclasses))

# Compute per-pixel max prob and winning class
winner_class <- which.max(probs)    # integer 1–nclasses
winner_prob  <- max(probs)          # float

# Initialize label map as NA
labels <- winner_class
values(labels) <- NA                # all NA initially

# Keep only where max prob ≥ threshold AND winner_prob is not NA
valid_mask <- !is.na(winner_prob) & (winner_prob >= threshold)
labels[valid_mask] <- winner_class[valid_mask]
names(labels) = 'class'

## -----------------------------------------------------------
## 3.  Extract EVERY pixel inside polygons
## -----------------------------------------------------------
polys_vect <- vect(shp_path)

ncores = parallel::detectCores() - 30
pix_df <- terra::extract(labels, polys_vect, parallel=ncores)
names(pix_df)[2] <- 'class'
head(pix_df)
## -----------------------------------------------------------
## 4.  Attach reference class using ID
## -----------------------------------------------------------
# pix_df$ID matches row numbers of polys_sf (1 … nrow(polys_sf))
reference <- polys_vect$code[pix_df$ID]#+1      # Add 1 because python index starts from Zero while R is 1
predicted <- pix_df$class                   # raster class for each pixel

# drop NA predictions (original no‑data)
keep <- !is.na(predicted)
reference <- reference[keep]
predicted <- predicted[keep]
acc = accuracy(reference, predicted)


  accuracy <- function(valid, pred){
    conmat <- table(observed=valid, predicted=pred)
    print(conmat)
    n <- sum(conmat)
    # Overall Accuracy
    OA <- sum(diag(conmat)) / n
    # number of total cases/samples
    print(OA)
    #kappa
    # observed (true) cases per class
    rowsums <- apply(conmat, 1, sum)
    p <- rowsums / n
    # predicted cases per class
    colsums <- apply(conmat, 2, sum)
    q <- colsums / n
    expAccuracy <- sum(p*q)
    kappa <- (OA - expAccuracy) / (1 - expAccuracy)
    print(kappa)
    # Producer accuracy
    PA <- diag(conmat) / colsums
    # User accuracy
    UA <- diag(conmat) / rowsums
    #F1-Score
    F1 <- (2*PA*UA)/(PA+UA)
    #outAcc <- data.frame(producerAccuracy = PA, userAccuracy = UA, F1_Score=F1, OA=OA,Kappa=kappa)
    List <- list("ConfusionMatrix" = conmat, "Producer"=PA, "user"=UA, "F1-Score" = F1,
                 "OverallAccuracy" = OA,"kappa" = kappa)
    print(List)
    return(List)
  }

## -----------------------------------------------------------
## 5.  Build confusion matrix
## -----------------------------------------------------------
cls_levels <- sort(unique(c(reference, predicted)))
reference  <- factor(reference,  levels = cls_levels)
predicted  <- factor(predicted,  levels = cls_levels)

cm <- table(Predicted = predicted, Reference = reference)
print(cm)

## -----------------------------------------------------------
## 6.  Accuracy metrics  (base R)
## -----------------------------------------------------------
diag_vals   <- diag(cm)

prod_acc <- diag_vals / colSums(cm)           # producer's (recall)
user_acc <- diag_vals / rowSums(cm)           # user's (precision)
f1        <- 2 * prod_acc * user_acc / (prod_acc + user_acc)

overall_acc <- sum(diag_vals) / sum(cm)

row_marg <- rowSums(cm)
col_marg <- colSums(cm)
exp_acc  <- sum(row_marg * col_marg) / sum(cm)^2
kappa    <- (overall_acc - exp_acc) / (1 - exp_acc)

## -----------------------------------------------------------
## 7.  Display results
## -----------------------------------------------------------
cat("\nOverall accuracy :", round(overall_acc, 3),
    "\nKappa statistic  :", round(kappa, 3), "\n\n")

cat("Class | ProducerAcc | UserAcc |   F1\n")
cat("---------------------------------------\n")
for(i in seq_along(cls_levels)){
  cat(sprintf("%5s |     %6.3f |  %6.3f | %6.3f\n",
              cls_levels[i], prod_acc[i], user_acc[i], f1[i]))
}

## -----------------------------------------------------------
## 8. (Optional) caret::confusionMatrix for cross‑check
## -----------------------------------------------------------
# cm_caret <- caret::confusionMatrix(predicted, reference)
# print(cm_caret)