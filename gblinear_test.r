
library(xgboost)

train_fn <- 'data/chembl1868.csv'

training_set <- as.matrix(read.table(train_fn, colClasses = 'numeric',
                                     header = TRUE))
cols_count = dim(training_set)[2]

x <- training_set[, 2:cols_count] # all lines, all columns except 1st
y <- training_set[, 1:1] # all lines, only 1st column (resp. var)

# check number of rows
stopifnot(cols_count == length(y))

# train
gbtree <- xgboost(data = x, label = y, booster='gblinear', eta = 0.2, objective = 'reg:squarederror', eval_metric = 'rmse', nrounds = 50, lambda = 0.0, alpha = 0.0)

xgb.save(gbtree, 'r_gbtree_model.bin')

xgb.load('r_gbtree_model.bin')

# stupid test on training data; don't do this at home !!!
values <- predict(gbtree, x)

write.table(values, file = 'data/predictions.txt', sep = '\n',
                    row.names = F, col.names = F)
quit()
