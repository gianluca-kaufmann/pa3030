# Are the stored Mouillot et al. (2024) land scores in-sample or out-of-bag?
#
# Refits the strict IUCN-I land model (formula and log transforms from analyses/Fig3.R) on the
# stored training sample, then compares the stored scores of the unprotected background cells
# with this refit's out-of-bag and in-sample predictions.
#
# Usage: Rscript mouillot_in_sample_check.R <path to FRBCesab-pelagic-f4a2b42>

library(ranger)

pkg <- commandArgs(trailingOnly = TRUE)[1]
train <- get(load(file.path(pkg, "data", "IUCN.I.VI.ATP.RData")))
names(train) <- make.unique(names(train))
stopifnot(identical(as.logical(train$Reserve), c(rep(TRUE, 258725), rep(FALSE, 258725))))

for (v in c("NGO", "conflicts", "DistToOcean", "tt", "Altitude", "NaturalResources", "AnnualPrecipitation", "GDP")) {
  train[[paste0("log", v)]] <- log10(train[[v]] + 1)
}
train$logHF <- log10(train$hf + 1)
train$Island <- as.factor(train$Island)
train$Reserve <- as.factor(train$Reserve)
strict <- train[train$IUCN_CAT %in% c("Ia", "Ib", "NONE"), ]

fit <- ranger(
  Reserve ~ logNGO + logconflicts + logDistToOcean + logtt + logHF + logAltitude + FreshWater +
    logAnnualPrecipitation + logGDP + HDI + logNaturalResources + Island + MeanTemperature + NDVI,
  data = strict, num.trees = 500, probability = TRUE, seed = 42, num.threads = parallel::detectCores()
)
strict$oob <- fit$predictions[, "TRUE"]
strict$in_sample <- predict(fit, strict, num.threads = parallel::detectCores())$predictions[, "TRUE"]

key <- function(d) paste(round(d$long, 4), round(d$lat, 4), d$ISO3)
stored <- get(load(file.path(pkg, "data", "res_prob_terrestrial.Rdata")))
final <- get(load(file.path(pkg, "data", "final_predictions_terrestrial.Rdata")))
stored$PredictedProtection <- final$PredictedProtection[match(key(stored), key(final))]
m <- merge(strict, stored[, c("long", "lat", "ISO3", "corr.rf.response", "PredictedProtection")],
           by = c("long", "lat", "ISO3"))
cat("rows: strict sample", nrow(strict), "| stored", nrow(stored), "| matched", nrow(m), "\n\n")

auc <- function(score, y) {
  r <- rank(score); n1 <- as.numeric(sum(y)); n0 <- length(y) - n1
  (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
y <- m$Reserve == "TRUE"
bg <- !y
cols <- c("oob", "in_sample", "PredictedProtection", "corr.rf.response")
qs <- sapply(cols, function(c) quantile(m[[c]][bg], c(0.05, 0.25, 0.5, 0.75, 0.95)))
cat("Background (unprotected) cell scores, quantiles:\n"); print(round(qs, 4))
cat("\nAUC protected (Ia/Ib) vs background:\n"); print(round(sapply(cols, function(c) auc(m[[c]], y)), 4))
cat("\nSpearman correlation on background cells:\n")
print(round(cor(m[bg, cols], method = "spearman"), 3))
