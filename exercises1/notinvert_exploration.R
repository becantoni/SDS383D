library(matlib) 
library(matrixcalc) 
library(microbenchmark) 

#Function that inverts the matrix
invert_sol <- function(A, b, S){
  beta_hat <- inv(t(A)%*%S%*%A)%*%t(A)%*%S%*%b 
  return(beta_hat)
}

#Function that uses LU decomposition 
lin_sistem_sol <- function(A, b, S){
  to_invert <- t(A)%*%S%*%A
  lu_decomp = lu.decomposition(to_invert)
  beta_hat = solve(lu_decomp$U, solve(lu_decomp$L, t(A)%*%b))
  return(beta_hat)
}

#try
n <- 100
p <- 10
W <- diag(n)
X <- matrix(rnorm(n*p), n, p) 
y <- rnorm(n, 0.1*X[,1]+0.2*X[,2] + 0.4*X[,3], 1)

try_1 <- invert_sol(X, y, W)
try_2 <- lin_sistem_sol(X, y, W)

try_compare <- microbenchmark(invert_sol(X, y, W), lin_sistem_sol(X, y, W), times=10)

#repeat for a range of N and P:
N <- c(10, 100, 500)
P <- c(2, 50, 100)
#evaluate over the different pairs:
for(i in 1:length(N)){
  ## create random dataset
  n <- N[i]
  p <- P[i]
  W <- diag(n) # identity matrix for W for
  X <- matrix(rnorm(n*p), n, p) # design matrix
  y <- rnorm(n, 0.3*X[,1]+0.5*X[,2], 1)
  ## Implementation
  assign(paste0("benchmark",i),microbenchmark(invert_sol(X, y, W), lin_sistem_sol(X, y, W), times=10)) 
}

for(i in 1:length(N)){
  print(paste0("Benchmark when N=", N[i]," and P=", P[i]))
  print(get(paste0("benchmark", i)))
}

