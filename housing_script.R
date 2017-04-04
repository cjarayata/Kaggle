# 4/4/2017 - kaggle housing prices testing

setwd("~/GitHub/Kaggle Housing")
train <- read.csv(file="train.csv", header = T, stringsAsFactors = F)
pacman::p_load(ggplot2, reshape2, caret, lattice, randomForest, extraTrees, broom, plyr, e1071, kernlab, glmnet, lars, data.table, xgboost, fastICA, rgl, ape, pls)

# subset cols
subset_colclasses <- function(DF, colclasses="numeric") {
  DF[,sapply(DF, function(vec, test) class(vec) %in% test, test=colclasses)]
}

blah.chr <- subset_colclasses(train, c("character"))
blah.int <- subset_colclasses(train, c("integer"))

blah.chr[is.na(blah.chr)]  <- "MISSING"
blah.int[is.na(blah.int)] <- -1

# unclass to strip the character, then making it dataframe will turn all to factor
blah.chr <- as.data.frame(unclass(blah.chr))


train <- cbind(blah.int, blah.chr)

train <- train[, -1] # cut ID
colnames(train) <- tolower(colnames(train))
# move sale price to front of dataset
train <- train[, c(37, 1:36, 38:ncol(train))]

# run something straight up and just check it out

fitControl <- trainControl(method = "repeatedcv",
                           number = 5, repeats = 2,
                           savePredictions=TRUE)

# partition data but make the sets identical for all models
set.seed(50)
trainIndex <- createDataPartition(train$saleprice, p = .8, list = FALSE, times = 1)
training <- train[ trainIndex,]
testing  <- train[-trainIndex,]


# wont run because NA

ptm <- proc.time()
set.seed(50)
lm.fit <- train(saleprice ~ ., data = training,
                method = "BstLm",
                trControl = fitControl,
                metric = "Rsquared")

# check accuracy of training set
lm.results <- data.frame(lm.fit$pred)
cor(lm.results$obs, lm.results$pred) # 0.8154546
plot(lm.results$obs, lm.results$pred)
as.numeric((proc.time() - ptm)[3]) # 7.43

# check on test set
lm.test.pred <- predict(lm.fit, testing[, -1])
cor(testing$saleprice, lm.test.pred) # 0.8470298
as.numeric((proc.time() - ptm)[3]) # this will return only elapsed time number
# user  system elapsed 
# 5.65    0.14    5.80 

plot(testing$saleprice, lm.test.pred)
lm.imp <- varImp(lm.fit)$importance # make importance a df
# need to sort; then concatenate rownames, then cut to top 10
lm.imp <- setDT(lm.imp, keep.rownames = T)
lm.imp <- lm.imp[order(-lm.imp$Overall), ]
lm.imp <- lm.imp[1:20, ]

# OK this is the top twenty for the boosted linear, so this will go into the second spreadsheet
lm.imp$top.var <- paste(lm.imp$rn, round(lm.imp$Overall, digits = 2), sep = " - ")

# now make predictions with new data

hold.out <- read.csv(file = "test.csv", header = T, stringsAsFactors = F)

# do same cleaning
hold.chr <- subset_colclasses(hold.out, c("character"))
hold.int <- subset_colclasses(hold.out, c("integer"))

hold.chr[is.na(hold.chr)]  <- "MISSING"
hold.int[is.na(hold.int)] <- -1

# unclass to strip the character, then making it dataframe will turn all to factor
hold.chr <- as.data.frame(unclass(hold.chr))


hold.out <- cbind(hold.int, hold.chr)

hold.out <- hold.out[, -1] # cut ID
colnames(hold.out) <- tolower(colnames(hold.out))
hold.out$pred <- predict(lm.fit, hold.out)
# Error in model.frame.default(Terms, newdata, na.action = na.action, xlev = object$xlevels) : 
# factor mszoning has new levels MISSING

table(testing$mszoning)
# C (all)      FV      RH      RL      RM 
# 0      19       2     225      45 
table(training$mszoning)
# C (all)      FV      RH      RL      RM 
# 10      46      14     926     173 
table(hold.out$mszoning)
# C (all)      FV MISSING      RH      RL      RM 
# 15      74       4      10    1114     242 

# let's change the missing in the hold.out set to RL
hold.out$mszoning[hold.out$mszoning == 'MISSING'] <- 'RL'
hold.out$mszoning <- factor(hold.out$mszoning)
table(hold.out$mszoning) # OK dealt with

# try again
hold.out$pred <- predict(lm.fit, hold.out) # now utilities

# extract important variables from first model
important <- lm.imp[lm.imp$Overall > 10]
list <- important$rn

trim <- train[colnames(train) %in% c("saleprice", list)]

fitControl <- trainControl(method = "repeatedcv",
                           number = 5, repeats = 2,
                           savePredictions=TRUE)

# partition data but make the sets identical for all models
set.seed(50)
trainIndex <- createDataPartition(trim$saleprice, p = .8, list = FALSE, times = 1)
training <- trim[ trainIndex,]
testing  <- trim[-trainIndex,]

# rerun training model using slimmed down predictors
ptm <- proc.time()
set.seed(50)
lm.fit <- train(saleprice ~ ., data = training,
                method = "BstLm",
                trControl = fitControl,
                metric = "Rsquared")

# check accuracy of training set
lm.results <- data.frame(lm.fit$pred)
cor(lm.results$obs, lm.results$pred) # 0.8154546, now 0.7925229
plot(lm.results$obs, lm.results$pred)
as.numeric((proc.time() - ptm)[3]) # 7.43

# check on test set
lm.test.pred <- predict(lm.fit, testing[, -1])
cor(testing$saleprice, lm.test.pred) # 0.8470298, now 0.8156968
as.numeric((proc.time() - ptm)[3])



hold.out <- read.csv(file = "test.csv", header = T, stringsAsFactors = F)

# do same cleaning
hold.chr <- subset_colclasses(hold.out, c("character"))
hold.int <- subset_colclasses(hold.out, c("integer"))

hold.chr[is.na(hold.chr)]  <- "MISSING"
hold.int[is.na(hold.int)] <- -1

# unclass to strip the character, then making it dataframe will turn all to factor
hold.chr <- as.data.frame(unclass(hold.chr))


hold.out <- cbind(hold.int, hold.chr)

hold.out <- hold.out[, -1] # cut ID
colnames(hold.out) <- tolower(colnames(hold.out))

trim.hold <- hold.out[colnames(hold.out) %in% c(list)]

# hopefully i dont keep running into this problem - welp i am
trim.hold$pred <- predict(lm.fit, trim.hold)

table(trim.hold$kitchenqual)

trim.hold$kitchenqual[trim.hold$kitchenqual == 'MISSING'] <- 'TA'
trim.hold$kitchenqual <- factor(trim.hold$kitchenqual)
table(trim.hold$kitchenqual) # OK dealt with

# this predict works
trim.hold$pred <- predict(lm.fit, trim.hold)

trim.hold$Id <- hold.out$Id

final <- trim.hold[, c(31, 30)]
colnames(final) <- c("Id", "SalePrice")

write.csv(final, file="housing_submit_1.csv", row.names = F)
## if i do, try pre-processing with near-zero predictors, then re-running on clean dataset
