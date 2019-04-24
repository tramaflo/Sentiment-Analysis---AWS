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

# Stop Cluster. After performing your tasks, stop your cluster
stopCluster(cl)


# Upload iphone small matrix ----------------------------------------------

iphoneData <- read_csv("C:/Users/T450S/Desktop/Floriana/Ubiqum/Big Data/Sentiment Analysis/iphone_smallmatrix_labeled_8d.csv")

summary(iphoneData)

str(iphoneData)

is.na(iphoneData)

hist(iphoneData$iphonesentiment)

plot_ly(iphoneData, x= ~iphoneData$iphonesentiment, type='histogram')

plot(iphoneData$iphonecampos, iphoneData$iphonesentiment)
plot(iphoneData$iphonecamneg, iphoneData$iphonesentiment)
plot(iphoneData$iphonecamunc, iphoneData$iphonesentiment)
plot(iphoneData$iphonedispos, iphoneData$iphonesentiment)
plot(iphoneData$iphonedisneg, iphoneData$iphonesentiment)
plot(iphoneData$iphonedisunc, iphoneData$iphonesentiment)
plot(iphoneData$iphoneperpos, iphoneData$iphonesentiment)
plot(iphoneData$iphoneperneg, iphoneData$iphonesentiment)
plot(iphoneData$iphoneperunc, iphoneData$iphonesentiment)
plot(iphoneData$iosperpos, iphoneData$iphonesentiment)
plot(iphoneData$iosperneg, iphoneData$iphonesentiment)


plot(iphoneData$iphonecampos, iphoneData$iphonesentiment)# Selct only the variables related to iPhones -----------------------------

iphoneData2 <- select(iphoneData, iphonecampos, iphonecamneg, iphonecamunc,  
                      iphonedispos, iphonedisneg, iphonedisunc,
                      iphoneperpos, iphoneperneg, iphoneperunc, 
                      iosperpos, iosperneg, iphonesentiment)


# Delete websites not related to iPhone -----------------------------------

listofnoiPhonewords <- apply(iphoneData2,1 , var) == 0

iphoneDataNoVar <-iphoneData2[!listofnoiPhonewords,]


# Recoding iphonesentiment from 6 to only 2 levels ------------------------

iphoneDataNoVar$iphonesentiment <- recode(iphoneDataNoVar$iphonesentiment, 
                                       '0' = 1, '1' = 1, '2' = 1,
                                       '3' = 5, '4' = 5, '5' = 5)


# Explore the data --------------------------------------------------------

hist(iphoneDataNoVar$iphonesentiment)

plot_ly(iphoneDataNoVar, x= ~iphoneDataNoVar$iphonesentiment, type='histogram')

plot(iphoneDataNoVar$iphonecampos, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iphonecamneg, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iphonecamunc, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iphonedispos, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iphonedisneg, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iphonedisunc, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iphoneperpos, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iphoneperneg, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iphoneperunc, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iosperpos, iphoneDataNoVar$iphonesentiment)
plot(iphoneDataNoVar$iosperneg, iphoneDataNoVar$iphonesentiment)

ggplot(iphoneDataNoVar, aes(iphonecampos, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iphonecamneg, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iphonecamunc, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iphonedispos, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iphonedisneg, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iphonedisunc, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iphoneperpos, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iphoneperneg, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iphoneperunc, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iosperpos, iphonesentiment))+
  
  geom_jitter(alpha = 1)

ggplot(iphoneDataNoVar, aes(iosperneg, iphonesentiment))+
  
  geom_jitter(alpha = 1)


# Correlation -------------------------------------------------------------

corrData <- cor(iphoneDataNoVar)

corrData

corrplot(corrData)

