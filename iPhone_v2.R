# AmazonWebService - Sentiment Analysis #
# Floriana Trama #
# April 2019 #



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




# Upload iphone small matrix ----------------------------------------------

iphoneData <- read_csv("C:/Users/T450S/Desktop/Floriana/Ubiqum/Big Data/Sentiment Analysis/iphone_smallmatrix_labeled_8d.csv")

summary(iphoneData)

str(iphoneData)

hist(iphoneData$iphonesentiment)


# Missing values ----------------------------------------------------------

is.na(iphoneData)

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


# Take a random sample of size 390 from each subset 
# Data1 = iphoneData1 = 390 obs

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

iphoneDataNew <- bind_rows(Data0, Data1, Data1, Data1, Data2, Data2, Data2, Data3, Data4, Data5)

hist(iphoneDataNew$iphonesentiment)


# Selct only the variables related to iPhones -----------------------------

iphoneDataNew2 <- select(iphoneData, iphone, ios, iphonecampos, iphonecamneg, iphonecamunc,  
                         iphonedispos, iphonedisneg, iphonedisunc,
                         iphoneperpos, iphoneperneg, iphoneperunc, 
                         iosperpos, iosperneg, iosperunc, iphonesentiment)


# Recoding iphonesentiment from 6 to only 2 levels ------------------------

iphoneDataNew2$iphonesentiment <- recode(iphoneDataNew2$iphonesentiment, 
                                         '0' = 0, '1' = 0, '2' = 0,
                                         '3' = 5, '4' = 5, '5' = 5)


# Outliers ----------------------------------------------------------------

boxplot(iphoneDataNew2)$out

boxplot(iphoneDataNew2$iphonecampos)$out
hist(iphoneDataNew2$iphonecampos)

summary(iphoneDataNew2$iphonecampos)
summary(iphoneDataNew2$iphonecamneg)
summary(iphoneDataNew2$iphonecamunc)
summary(iphoneDataNew2$iphonedispos)
summary(iphoneDataNew2$iphonedisneg)
summary(iphoneDataNew2$iphonedisunc)
summary(iphoneDataNew2$iphoneperpos)
summary(iphoneDataNew2$iphoneperneg)
summary(iphoneDataNew2$iphoneperunc)
summary(iphoneDataNew2$iosperpos)
summary(iphoneDataNew2$iosperneg)
summary(iphoneDataNew2$iphonesentiment)

# Correlation -------------------------------------------------------------

corrData <- cor(iphoneDataNoVar)

corrData

corrplot(corrData)


# Transform Sentiment variable in Factor ----------------------------------

iphoneDataNew2$iphonesentiment <- as.factor(iphoneDataNew2$iphonesentiment)


# Set seed ----------------------------------------------------------------

set.seed(123)


# Create 75%/25% training and test sets -----------------------------------

inTraining <- createDataPartition(iphoneDataNew2$iphonesentiment, p = .75, list = FALSE)

training <- iphoneDataNew2[inTraining,]

testing <- iphoneDataNew2[-inTraining,]


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


# Predictions -------------------------------------------------------------

prediction <- predict(RFfit, testing)

confusionMatrix(prediction, testing$iphonesentiment)

postResample(prediction, testing$iphonesentiment)


# Error -------------------------------------------------------------------

Specialtable <- cbind(testing, prediction)

Specialtable$iphonesentiment <- as.numeric(Specialtable$iphonesentiment)

Specialtable$prediction <- as.numeric(Specialtable$prediction)

error <- abs(Specialtable$iphonesentiment - Specialtable$prediction)

Errorplot <- cbind(Specialtable, error)

plot(Specialtable$prediction, Specialtable$iphonesentiment)

write.csv(Errorplot, file = "ErrorplotIPHONECIAO.csv")

ggplot(Errorplot, aes(prediction, iphonesentiment, color = as.factor(error)))+ 
  geom_jitter(alpha = 0.5)

# Stop Cluster. After performing your tasks, stop your cluster
stopCluster(cl)