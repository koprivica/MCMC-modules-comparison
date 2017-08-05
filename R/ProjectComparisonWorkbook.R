# Project comparison workbook 
#------------------------------------------------------------------------------- 

# Reading omegas
myData = read.csv("/home/marko/phdspace/MCMC-modules-comparison/results/jags/omega/test_3.csv", skip = 1, header = FALSE, col.names=c('omega1', 'omega2', 'omega3'))

# Number of test files
nFiles = seq(from=1, to=100, by=1)

allData <- array(dim=c(6667, 3, 100))
print(dim(allData))
for (val in nFiles) {
  print(sprintf("-----------------------------------------%i-----------------------------------------", val))
  x = read.csv(sprintf("/home/marko/phdspace/MCMC-modules-comparison/results/jags/omega/test_%i.csv", val), skip = 1, header = FALSE, col.names=c('omega1', 'omega2', 'omega3'))
  print(dim(x))

  allData[,,val] = matrix( unlist(x), ncol=3)#matrix(c(x$omega1,x$omega2,x$omega3),ncol=3)
}