# AmazonWebService - Sentiment Analysis #
# Floriana Trama #
# April 2019 #
# KKNN #



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


# Upload iphone small matrix and large matrix -----------------------------

iphoneData <- read_csv("C:/Users/T450S/Desktop/Floriana/Ubiqum/Big Data/Sentiment Analysis/iphone_smallmatrix_labeled_8d.csv")


# First data exploration --------------------------------------------------

# Summary
summary(iphoneData)

str(iphoneData)

hist(iphoneData$iphonesentiment)

# Missing values
is.na(iphoneData)

# Outliers
boxplot(iphoneData)$out


# Selct only the variables related to iPhones -----------------------------

iphoneData2 <- select(iphoneData, iphone, iphonecampos, iphonecamneg, 
                         iphonecamunc, iphonedispos, iphonedisneg, iphonedisunc,
                         iphoneperpos, iphonesentiment)

# Deleting rows with values > 0 in the "iphone" var and 0 in others var ---
# There is a problem in the prediction of the sentiment in this rows = webpages because
# the model has to decide the sentiment just knowing that in the page there is the term "iphone"
# if you do it manually like in the small matrix, you can understand which is the sentiment
# but the model alone, cannot know it, it can just guess (that's why K is low if you keep that rows)

iphoneData3 <- iphoneData2[!(iphoneData2$iphone > "0" &
                                     iphoneData2$iphonecampos == "0" & iphoneData2$iphonecamneg == "0" & 
                                     iphoneData2$iphonecamunc == "0" & iphoneData2$iphonedispos == "0" & 
                                     iphoneData2$iphonedisneg == "0" & iphoneData2$iphonedisunc == "0" & 
                                     iphoneData2$iphoneperpos == "0"),]


# Recoding iphonesentiment from 6 levels to only 2 ------------------------

iphoneData3$iphonesentiment <- recode(iphoneData3$iphonesentiment, 
                                         '0' = 0, '1' = 0, '2' = 0,
                                         '3' = 5, '4' = 5, '5' = 5)


# Correlation -------------------------------------------------------------

corrData <- cor(iphoneData3)

corrData

corrplot(corrData)


# Transform Sentiment variable in Factor ----------------------------------

iphoneData3$iphonesentiment <- as.factor(iphoneData3$iphonesentiment)


# Set seed ----------------------------------------------------------------

set.seed(123)


# Create 75%/25% training and test sets -----------------------------------

inTraining <- createDataPartition(iphoneData3$iphonesentiment, p = .75, list = FALSE)

training <- iphoneData3[inTraining,]

testing <- iphoneData3[-inTraining,]


# Cross validation --------------------------------------------------------

fitcontrol <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 1)


# Train KKNN model ----------------------------------------------------------

KKNNfit <- train(iphonesentiment ~ .,
                data = training,
                method = "kknn",
                tuneLength = 1,
                trControl = fitcontrol)


# Training results --------------------------------------------------------

KKNNfit

varImp(KKNNfit)


# Predictions on testing set ----------------------------------------------

prediction <- predict(KKNNfit, testing)

confusionMatrix(prediction, testing$iphonesentiment)

postResample(prediction, testing$iphonesentiment)


# Errors ------------------------------------------------------------------

Specialtable <- cbind(testing, prediction)

Specialtable$iphonesentiment <- as.numeric(Specialtable$iphonesentiment)

Specialtable$prediction <- as.numeric(Specialtable$prediction)

error <- abs(Specialtable$iphonesentiment - Specialtable$prediction)

Errorplot <- cbind(Specialtable, error)

ggplot(Errorplot, aes(iphonesentiment, prediction, color = as.factor(error)))+ 
  geom_jitter(alpha = 0.5)

write.csv(Errorplot, file = "KKNNErrorplotIPHONE.csv")







# Stop Cluster. After performing your tasks, stop your cluster
stopCluster(cl)