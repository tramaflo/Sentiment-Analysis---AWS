# AmazonWebService - Sentiment Analysis #
# Floriana Trama #
# April 2019 #



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


# Libraries ---------------------------------------------------------------

if(!require(pacman))install.packages("pacman")

pacman::p_load('doParallel', 'plotly', 'caret', 'corrplot', 'dplyr',
               "e1071", "kknn", 'ROSE')


# Upload iphone small matrix and large matrix -----------------------------

iphoneData <- read_csv("C:/Users/T450S/Desktop/Floriana/Ubiqum/Big Data/Sentiment Analysis/iphone_smallmatrix_labeled_8d.csv")

largeMatrix <- read_csv("C:/Users/T450S/Desktop/Floriana/Ubiqum/Big Data/Sentiment Analysis/Large Matrix.csv")


# First data exploration --------------------------------------------------

# Summary
summary(iphoneData)

str(iphoneData)

hist(iphoneData$iphonesentiment)

# Missing values
is.na(iphoneData)

# Outliers
boxplot(iphoneData)$out


# Create new datasets by "Sentiment" --------------------------------------

iphoneData0 <- iphoneData %>% 
  filter(iphonesentiment == 0)

iphoneData1 <- iphoneData %>% 
  filter(iphonesentiment == 1)

iphoneData2 <- iphoneData %>% 
  filter(iphonesentiment == 2)

iphoneData3 <- iphoneData %>% 
  filter(iphonesentiment == 3)

iphoneData4 <- iphoneData %>% 
  filter(iphonesentiment == 4)

iphoneData5 <- iphoneData %>% 
  filter(iphonesentiment == 5)


# Take a random sample from each subset -----------------------------------
# Get around 1200 observations per subset
# Data1 = iphoneData1 = 390 obs
# Data3 = iphoneData3 = 1188 obs

Data0 <- iphoneData0[sample(1:nrow(iphoneData0), 1200,
                            replace=FALSE),]

Data1 <- iphoneData1[sample(1:nrow(iphoneData1), 390,
                            replace=FALSE),]

Data2 <- iphoneData2[sample(1:nrow(iphoneData2), 400,
                            replace=FALSE),]

Data3 <- iphoneData3[sample(1:nrow(iphoneData3), 1188,
                            replace=FALSE),]

Data4 <- iphoneData4[sample(1:nrow(iphoneData4), 1200,
                            replace=FALSE),]

Data5 <- iphoneData5[sample(1:nrow(iphoneData5), 1200,
                            replace=FALSE),]


# Oversampling - create a new balanced dataset ----------------------------
# Balanced in ternms of sentiment

iphoneDataNew <- bind_rows(Data0, Data1, Data1, Data1, Data2, Data2, Data2, Data3, Data4, Data5)

hist(iphoneDataNew$iphonesentiment)


# Selct only the variables related to iPhones -----------------------------

iphoneDataNew2 <- select(iphoneData, iphone, iphonecampos, iphonecamneg, 
                         iphonecamunc, iphonedispos, iphonedisneg, iphonedisunc,
                         iphoneperpos, iosperpos, iosperneg, iosperunc, iphonesentiment)

# Deleting rows with values > 0 in the "iphone" var and 0 in others var ---
# There is a problem in the prediction of the sentiment in this rows = webpages because
# the model has to decide the sentiment just knowing that in the page there is the term "iphone"
# if you do it manually like in the small matrix, you can understand which is the sentiment
# but the model alone, cannot know it, it can just guess (that's why K is low if you keep that rows)

iphoneDataNew3 <- iphoneDataNew2[!(iphoneDataNew2$iphone > "0" &
                                     iphoneDataNew2$iphonecampos == "0" & iphoneDataNew2$iphonecamneg == "0" & 
                                     iphoneDataNew2$iphonecamunc == "0" & iphoneDataNew2$iphonedispos == "0" & 
                                     iphoneDataNew2$iphonedisneg == "0" & iphoneDataNew2$iphonedisunc == "0" & 
                                     iphoneDataNew2$iphoneperpos == "0" & iphoneDataNew2$iosperpos == "0" & 
                                     iphoneDataNew2$iosperneg == "0" & iphoneDataNew2$iosperunc == "0"),]


# Recoding iphonesentiment from 6 levels to only 2 ------------------------

iphoneDataNew3$iphonesentiment <- recode(iphoneDataNew3$iphonesentiment, 
                                         '0' = 0, '1' = 0, '2' = 0,
                                         '3' = 5, '4' = 5, '5' = 5)


# Correlation -------------------------------------------------------------

corrData <- cor(iphoneDataNew3)

corrData

corrplot(corrData)

# Transform Sentiment variable in Factor ----------------------------------

iphoneDataNew3$iphonesentiment <- as.factor(iphoneDataNew3$iphonesentiment)


# Set seed ----------------------------------------------------------------

set.seed(123)


# Create 75%/25% training and test sets -----------------------------------

inTraining <- createDataPartition(iphoneDataNew3$iphonesentiment, p = .75, list = FALSE)

training <- iphoneDataNew3[inTraining,]

testing <- iphoneDataNew3[-inTraining,]


# Cross validation --------------------------------------------------------

fitcontrol <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 1)


# Train RF model ----------------------------------------------------------

RFfit <- train(iphonesentiment ~ .,
               data = training,
               method = "rf",
               tuneLength = 1,
               trControl = fitcontrol)


# Training results --------------------------------------------------------

RFfit

varImp(RFfit)


# Predictions on testing set ----------------------------------------------

prediction <- predict(RFfit, testing)

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

write.csv(Errorplot, file = "ErrorplotIPHONE.csv")







# Stop Cluster. After performing your tasks, stop your cluster
stopCluster(cl)