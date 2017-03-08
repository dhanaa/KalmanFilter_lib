
# coding: utf-8


'''
Kalman filter 

This module demonstrates the standard Kalman filter implementation.

Assume the system model:
x(k+1) = A*x(k) + w(k)
y(k) = C*x(k) + v(k)
where x is the state, y is the measurement, k is time index, w,v are Gaussian noises with covariance Q,R respectively.

For Kalman filter, please refer to 
http://www.cs.unc.edu/~tracker/media/pdf/SIGGRAPH2001_CoursePack_08.pdf  (pp.24)

---
Example

If you have the system parameters, e.g., A,C,Q,R and the initial conditions, e.g. x_0 and P_0 which are the initial estimate
on state x and initial estimation error covariance, you can use the Kalman filter in this module in two ways:

Case 1:
If you are running a real-time system and want to filter the signal x(k) from y(k), y(k-1).... y(0), you can do
a) at time 0, you have the initial guess as the priori estimate and covarariance for the next-step estimation.
b) at time k, you input your last-step estimate e(k-1) and error covariance P(k-1), your measurement y(k), and system 
    parameters into the function KalmanFilterUpdate(e(k-1), P(k-1), y(k), A, C, Q, R) and obtain a new estimate e(k) and 
    its error covariance P(k), e.g., 
        e(k), P(k) = KalmanFilterUpdate(e(k-1), P(k-1), y(k), A, C, Q, R)
c) jump to step b) and loop.

Case 2:
If you have all the measurements already, you can get all estimate and its corresponding error covariance in a lump sum.
You can use
        x_est, P_est = KalmanFilterBatch(y, A, C, Q, R, xInit, pInit)
If the initial conditions are missing, default conditions will be used.
Note that this is NOT a smoothing algorithm.

---

Example code:

# the measurements here are generated randomly
y = np.random.rand(100,3)
A = [[1,0],[0,1.1]]
C = np.array([[2,1],[1,2],[3,2.3]])
Q = np.eye(2)
R = np.eye(3)
xx, pp = KalmanFilterBatch(y,A,C,Q,R)
print(pp[100])
pri, pos = SteadyCov(A,C,Q,R)
print(pos)
# we can see that pp[100] and pos are indeed the same.


---

Author: Duo HAN
Email: dhanaa@connect.ust.hk
Date: 2017.3.8
'''

import numpy as np
from scipy import linalg

def KalmanFilterUpdate(xPri,pPri,y,A,C,Q,R):
    '''
    One-step Kalman filter
    
    Input
    -------
    xPri: numpy.ndarray or list
        n-dim vector representing the priori estimate
    pPri: numpy.ndarray or list
        n*n matrix which is the priori error covariance
    y: numpy.ndarray or list
        m-dim vector representing the measurement
    A, Q: numpy.ndarray or list
        n*n matrix
    C: numpy.ndarray or list
        m*n matrix
    R: numpy.ndarray or list
        m*m matrix
    
    Output
    -------
    xPos: numpy.ndarray or list
        n-dim vector representing the posteriori estimate
    pPos:
        n*n matrix representing the posteriori error covariance

    
    '''
    # transform the input into np.array if they are lists
    xPri =  np.asarray(xPri)
    pPri = np.asarray(pPri)
    y = np.asarray(y)
    A = np.asarray(A)
    C = np.asarray(C)
    Q = np.asarray(Q)
    R = np.asarray(R)
    
    # initialization
    xPri = xPri.squeeze()
    y = y.squeeze()
    n = len(xPri)
    m = len(y)
    
    # time update
    xPos = A.dot(xPri)
    pPos = A.dot(pPri).dot(A.T) + Q
    
    # measurement update
    K = pPos.dot(C.T).dot(np.linalg.inv(C.dot(pPos).dot(C.T) + R))
    xPos = xPos + K.dot( y - C.dot(xPos) )
    pPos = (np.eye(n) - K.dot(C)).dot(pPos)
    
    return xPos,pPos

def KalmanFilterBatch(y, A, C, Q, R, xInit = np.zeros(len(A),), pInit = np.eye(len(A))):
    '''
    Kalman filter for a batch of measurements
    
    Input
    -------
    y: numpy.ndarray or list
        A total T number of m-dim vectors representing a sequence of measurements
    A, Q: numpy.ndarray or list
        n*n matrix
    C: numpy.ndarray or list
        m*n matrix
    R: numpy.ndarray or list
        m*m matrix
    xInit: numpy.ndarray or list
        n-dim vector representing the initial estimate
    pInit: numpy.ndarray or list
        n*n matrix which is the initial error covariance
    
    Output
    -------
    xEst: numpy.ndarray or list
        T number of n-dim vectors representing the estimate at time k
    pCov:
        T number of n*n matrices representing the error covariance at time k

    
    '''
    # transform the input into np.array if they are lists
    y = np.asarray(y)
    A = np.asarray(A)
    C = np.asarray(C)
    Q = np.asarray(Q)
    R = np.asarray(R)

    # initialization
    n,_ = A.shape
    x = xInit
    p = pInit
    xEst = [x]
    pCov = [pInit]
    
    # check if the measurement is 1-dimensional
    if (y.ndim==1):
        x,p = KalmanFilterUpdate(x,p,y,A,C,Q,R)
        return x,p
    
    # recursively run the Kalman filter update using the measurement
    T,m = y.shape
    for k in range(T):
        x,p = KalmanFilterUpdate(x,p,y[k],A,C,Q,R)
        xEst.append(x)
        pCov.append(p)
    return xEst, pCov

def SteadyCov(A, C, Q, R):
    '''
    Compute the steady-state estimation covariance of Kalman filter, independent of measurements.
    
    Input
    -------
    A, Q: numpy.ndarray or list
        n*n matrix
    C: numpy.ndarray or list
        m*n matrix
    R: numpy.ndarray or list
        m*m matrix
    
    Output
    -------
    pPri: numpy.ndarray or list
        n*n matrix, steady-state priori error covariance
    pPos: numpy.ndarray or list
        n*n matrix, steady-state posteriori error covariance

    
    '''
    # transform the input into np.array if they are lists
    A = np.asarray(A)
    C = np.asarray(C)
    Q = np.asarray(Q)
    R = np.asarray(R)
    
    # solve the discrete algebraic Riccati equation
    pPri = linalg.solve_discrete_are(A.T,C.T,Q,R)
    # compute the posteriori covariance from the priori covariance
    K = pPri.dot(C.T).dot(np.linalg.inv(C.dot(pPri).dot(C.T) + R))
    pPos = pPri - (K.dot(C)).dot(pPri)
    
    return pPri,pPos

