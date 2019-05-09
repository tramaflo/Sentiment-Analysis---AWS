# AmazonWebService - Sentiment Analysis #
# Floriana Trama #
# April 2019 #
# Random Forest #



# Libraries ---------------------------------------------------------------

if(!require(pacman))install.packages("pacman")

pacman::p_load('doParallel', 'plotly', 'caret', 'corrplot', 'dplyr',
               "e1071", "kknn", 'ROSE')


# Prepare rstudio for sentiment analysis ----------------------------------

# Find how many cores are on your machine
# Result = Typically 4 to 6
detectCores()

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(2)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers()


# Upload galaxy small matrix and large matrix -----------------------------

galaxyData <- read_csv("galaxy_smallmatrix_labeled_9d.csv")

LargeMatrixFinalGalaxy <- read_delim("LargeMatrixFinalGalaxy.csv", 
                                     ";", escape_double = FALSE, trim_ws = TRUE)


# First data exploration --------------------------------------------------

# Summary
summary(galaxyData)

str(galaxyData)

hist(galaxyData$galaxysentiment)

# Missing values
is.na(galaxyData)

# Outliers
boxplot(galaxyData)$out


# Selct only the variables related to galaxy -------------------------------------

galaxyData2 <- select(galaxyData, samsungcampos, samsungcamneg,
                      samsungcamunc, samsungdispos, samsungdisneg, samsungdisunc,
                      samsungperpos, samsungperneg, samsungperunc, galaxysentiment)

ggplot(galaxyData2, aes(x=galaxysentiment, fill=galaxysentiment))+
  geom_bar(fill="#CC5544")+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank())+
  labs(title= 'Number of observations by Sentiment', x= 'Galaxy Sentiment ', y= 'Frequency')


# Create a new variable indicating the total number of observation by row

galaxyData2["TotReviews"] <- NA

galaxyData2$TotReviews <- as.numeric(galaxyData2$TotReviews)

galaxyData2$TotReviews <- (rowSums( galaxyData2[,1:9]))

summary(galaxyData2)


# Deleting rows with a total number of observations lower than 2

galaxyData3 <- galaxyData2[!(galaxyData2$TotReviews < "2"),]

galaxyData3$TotReviews <- NULL

ggplot(galaxyData3, aes(x=galaxysentiment, fill=galaxysentiment))+
  geom_bar(fill="#CC5544")+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank())+
  labs(title= 'Number of observations by Sentiment', x= 'Galaxy Sentiment ', y= 'Frequency')


# Recoding galaxysentiment from 6 levels to only 2 ------------------------

galaxyData3$galaxysentiment <- recode(galaxyData3$galaxysentiment, 
                                      '0' = 0, '1' = 0, '2' = 0,
                                      '3' = 5, '4' = 5, '5' = 5)


# Transform Sentiment variable in Factor ----------------------------------

galaxyData3$galaxysentiment <- as.factor(galaxyData3$galaxysentiment)

LargeMatrixFinalGalaxy$galaxysentiment <- as.factor(LargeMatrixFinalGalaxy$galaxysentiment)


# Set seed ----------------------------------------------------------------

set.seed(123)


# Galaxy sampling (both = oversampling and undersampling at the same time -

galaxyData.both <- ovun.sample(galaxysentiment~., 
                               data = galaxyData3,
                               N = nrow(galaxyData3), 
                               p = 0.5, 
                               seed = 1, 
                               method = "both")$data


# Create 75%/25% training and test sets -----------------------------------

inTraining <- createDataPartition(galaxyData.both$galaxysentiment, p = .75, list = FALSE)

training <- galaxyData.both[inTraining,]

testing <- galaxyData.both[-inTraining,]


# Cross validation --------------------------------------------------------

fitcontrol <- trainControl(method = "repeatedcv",
                           number = 10,
                           preProc = c("center", "scale"),
                           repeats = 1,
                           verboseIter = TRUE)


# Train RF model ----------------------------------------------------------

RFfit <- train(galaxysentiment ~ .,
               data = training,
               method = "rf",
               trControl = fitcontrol)


# Training results --------------------------------------------------------

RFfit

varImp(RFfit)


# Predictions on testing set ----------------------------------------------

prediction <- predict(RFfit, testing)

confusionMatrix(prediction, testing$galaxysentiment)

postResample(prediction, testing$galaxysentiment)


# Errors ------------------------------------------------------------------

Specialtable <- cbind(testing, prediction)

Specialtable$galaxysentiment <- as.numeric(Specialtable$galaxysentiment)

Specialtable$prediction <- as.numeric(Specialtable$prediction)

error <- abs(Specialtable$galaxysentiment - Specialtable$prediction)

Errorplot <- cbind(Specialtable, error)

ggplot(Errorplot, aes(galaxysentiment, prediction, color = as.factor(error)))+ 
  geom_jitter(alpha = 0.5)

write.csv(Errorplot, file = "RFErrorplotGALAXY.csv")


# Make sentiment predictions on Large matrix ------------------------------

FinalPrediction <- predict(object = RFfit, newdata = LargeMatrixFinalGalaxy)

FinalPrediction

Specialtable2 <- cbind(LargeMatrixFinalGalaxy, FinalPrediction)

summary(FinalPrediction)

write.csv(Specialtable2, file = "FinalPredictionsRFgalaxy.csv")


# Stop Cluster. After performing your tasks, stop your cluster
stopCluster(cl)