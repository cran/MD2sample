#' This function creates data for Dalitz plot
#' @param nyadd =0
#' @param which type of Daliz plot.
#' @param addon what to add to uniform
#' @param n =200, sample size of both data sets.
#' @param nx =n, sample size of x data set.
#' @param ny =n, sample size of y data set.
#' @keywords internal
#' @return a list of functions
#' @export
rDalitz=function(nyadd=0, which=1, addon, n=200, nx=n, ny=n) {
  
  if(which==1) {M=1.97;m1=0.4937;m2=0.4937;m3=0.1396;delta=0.3}
  if(which==2)  {M=3;m1=0.1396;m2=0.4976;m3=0.938;delta=0.2}
  if(which==3)  {M=3;m1=0.3099;m2=0.3101;m3=0.3094;delta=0.2}
  sq=function(x) {
     y=rep(NA, length(x))
     for(i in seq_along(x)) if(x[i]>=0) y[i]=sqrt(x[i])
     y
  }
  doCut=function() {
    E2=(m12s-m1^2+m2^2)/(2*sqrt(m12s))
    E3=(M^2-m12s-m3^2+m2^2)/(2*sqrt(m12s))
    m23max=(E2+E3)^2-(sq(E2^2-m2^2)-sq(E3^2-m3^2))^2
    m23min=(E2+E3)^2-(sq(E2^2-m2^2)+sq(E3^2-m3^2))^2
    u=cbind(m12s, m23s)
    u=u[u[,2]>m23min & u[,2]<m23max, ]
    u=u[!is.na(u[,1]),]
    u[!is.na(u[,2]),]
  }
  # Find border    
  xrange=c( (m1+m2)^2,(M-m3)^2)+delta*c(-1,1)
  yrange=c( (m2+m3)^2,(M-m1)^2)+delta*c(-1,1)    
  m12s=seq(xrange[1], xrange[2], length=3000)
  E2=(m12s-m1^2+m2^2)/(2*sqrt(m12s))
  E3=(M^2-m12s-m3^2+m2^2)/(2*sqrt(m12s))
  m23max=(E2+E3)^2-(sq(E2^2-m2^2)-sq(E3^2-m3^2))^2
  m23min=(E2+E3)^2-(sq(E2^2-m2^2)+sq(E3^2-m3^2))^2
  borders=cbind(m12s, m23min, m23max)
  borders=borders[!is.na(borders[,2]), ]
  borders=borders[!is.na(borders[,3]), ]
  mbor=nrow(borders)
  # Find uniform data
  u=NULL 
  repeat {
      m12s=runif(nx+ny, xrange[1], xrange[2])
      m23s=runif(nx+ny, yrange[1], yrange[2])
      u=rbind(u, doCut())
      if(nrow(u)>nx+ny) break
  }
  x=u[1:nx,]
  y=u[(1+nx):(nx+ny),]
  if(nyadd==0) return(list(x=x, y=y, borders=borders))
  add_data=as.list(1:4)
  names(add_data)=c("Left Stripe", "Right Stripe",
                    "Bottom Stripe", "Diagonal Stripe")
# left stripe    
    m12s=runif(ny, borders[1,1], borders[100,1])
    E2=(m12s-m1^2+m2^2)/(2*sqrt(m12s))
    E3=(M^2-m12s-m3^2+m2^2)/(2*sqrt(m12s))
    m23max=(E2+E3)^2-(sq(E2^2-m2^2)-sq(E3^2-m3^2))^2
    m23min=(E2+E3)^2-(sq(E2^2-m2^2)+sq(E3^2-m3^2))^2 
    m23s=m23min+rbeta(ny, 0.1, 0.1)*(m23max-m23min)
    add_data[[1]]=cbind(m12s, m23s)
#bottom stripe  
    m12s=sort(runif(ny, borders[1,1], borders[0.5*mbor,1]))
    ab=c(borders[0.6*mbor, 2], borders[0.1*mbor,2])
    q=(m12s-min(m12s))/(max(m12s)-min(m12s))/2.3
    m23s=runif(ny, ab[1]+q*diff(ab),ab[2]-q*diff(ab))
    add_data[[3]]=doCut()
#right stripe    
    tmp=rexp(2*ny,3)
    tmp=1-(tmp[tmp<1])[1:ny]
    m12s=sort(runif(ny, borders[0.6*mbor,1], borders[mbor,1]))
    m12s=borders[0.6*mbor,1]+(borders[mbor,1]-borders[0.6*mbor,1])*tmp
    ab=c(borders[0.3*mbor, 2], borders[0.1*mbor,2])
    q=(m12s-min(m12s))/(max(m12s)-min(m12s))/2.5
    q=max(q)-q
    m23s=runif(ny, ab[1]+q*diff(ab),ab[2]-q*diff(ab))
    add_data[[2]]=doCut()
#Diagonal
  m12s=runif(3*nyadd, borders[1,1],borders[mbor,1])
  q=c(range(borders[,1]),range(borders[,2:3]))
  lne=function(x) (q[4]-q[3])/(q[1]-q[2])*(x-q[1])+q[4]
  m23s=lne(m12s)+runif(3*nyadd, -0.05, 0.05)
  add_data[[4]]=doCut()
  k=length(addon)
  y=y[1:(ny-nyadd*k), ]
  for(i in addon) y=rbind(y, add_data[[i]][1:nyadd,])
  list(x=x, y=y, borders=borders)
}