library('xgboost')

# matrix with n rows (observations) and p columns (features)
x <- as.matrix(read.table("data/train_data.txt", colClasses = "numeric"))

#x <- read.matrix.csr("data/train_data.csr", fac = FALSE)
# FBR: in the sparse case, we should use a dgCMatrix

# vector of size n and values +1 or -1 only
y <- as.vector(read.table("data/train_labels.txt"), mode = "numeric")

# transform [-1,1] to [0,1]
lut <- data.frame(old = c(-1.0,1.0), new = c(0.0,1.0))
labels <- lut$new[match(y, lut$old)]

# check number of rows
stopifnot(nrow(x) == length(labels))

# train
gbtree <- xgboost(data = x, label = labels, nrounds = 10, objective = "binary:logistic")

save(gbtree, file="r_gbtree_model.bin")

load("r_gbtree_model.bin")

# stupid test on training data; don't do this at home !!!
values <- predict(gbtree, x)

write.table(values, file = "data/predictions.txt", sep = "\n", row.names = F, col.names = F)

quit()
