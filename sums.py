# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 19:45:40 2023

@author: Krithika
"""

#INTERVAL 
x       = c(0,10,20,30,40)
y       = (x+10)

#FREQUENCY
fq      = c(2,12,22,8,6)
cf      = 0
n       = sum(fq)
summ    = 0
mid     = 0

#TO PRINT THE INTERVAL 
print("The data set is given as")

for (i in 1:5)
{
  print(paste(x[i]," - ",y[i],"  ",fq[i]))
  cf[i]=0
}


#MEAN

for(i in 1:5)
{
  mid[i] = c((x[i]+y[i])/2)
  summ   = (summ+ (mid[i]*fq[i]))
}
paste(" THE MEAN IS : ",round(summ/sum(fq)))


#MEDIAN

n = sum(fq)
cf = 0
l = 0
h = 0
f = 0

for(i in 1:5)
{
  if((n/2)>=x[i])
  {
    cf = fq[i-1]
    f  = fq[i]
    l  = x[i]
  }
  else
  {next}
}

h = (y[2]-x[2])

paste("THE MEDIAN IS : ",round((l + (((n/2)-cf)/f)*h),digits=2))


#MODE

maximum  <- max(fq)
index<-0
for(i in 1:5)
{
  if(fq[i]==maximum)
  {
    index <- i 
  }
}
h=y[2]-x[2]
f0 = (fq[index-1])
f1 = (fq[index])
f2 = (fq[index+1])

paste("THE MODE IS :  ",round((l + ((f1-f0)/(2*f1 - f0 -f2)*h))),digit=1)


#VARIANCE

summ = 0
m1 = 0
m2 = 0
mid = (x+y)/2
m1= mid*fq
n= sum(fq)
mean= sum(m1)/n
m2=fq*(mid-mean)*(mid-mean)
var=sum(m2/n)
paste("THE VARIANCE IS : ",round(var,digit=2))


#STANDARD DEVIATION
sd = sqrt(var)
paste("THE STANDARD DEVIATION IS : ",round(sd),digit=2)


