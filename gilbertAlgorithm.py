"""
Simple implementation of Gilbert's algorithm

This script contains the functions used in https://arxiv.org/pdf/2207.08850.pdf .
This basic implementation is given in python, however, further speed ups can be obtained
including pre-compilation in cython. 

The Bell scenario is given by two parties, m- measurements and 2 outcomes. Thus, 
the correlations (behaviours) are represented by vectors of dimension D = m^2 + 2m.
This is achieved after applying the normalization and no-signalling conditions. In
particular P = {p(1,1|x,y),p_A(1|x),p_B(1|y)} for x,y in {1,..., m}.

Moreover, the script uses two noise models, the visibility (W) and the symmetric 
detection efficiency (eta). In this case the Alice's and Bob's detectors have
the same efficiency.

"""

import numpy as np
from math import sqrt, log2, log, pi, cos, sin
from sympy.physics.quantum.dagger import Dagger
import mosek
import qutip as qtp
from joblib import Parallel, delayed, parallel_backend
from glob import glob
from scipy.optimize import minimize, minimize_scalar
from datetime import datetime
import matplotlib.pyplot as plt


##############################################################################
################################# YuOh Set ##################################
##############################################################################

def correlationsYuOh(W,eta):
    """
    Generating the correlations using the YuOh set and the maximally entangled state (d=3)
    with visibility `W' and the detection efficiency by `eta'
    """
    
    [id, sx, sy, sz] = [qtp.qeye(3), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]

    rho0 = ((1/sqrt(3))*(qtp.ket('00',dim=3) + qtp.ket('11',dim=3)+qtp.ket('22',dim=3))).proj()
    rho= W*rho0 + ((1-W)/9)*qtp.tensor(id, id)
    
    #v1 = (1,0,0)
    v1 = qtp.ket('0',dim=3).unit()
    
    #v2 = (0,1,0)
    v2 = qtp.ket('1',dim=3).unit()
    
    #v3 = (0,0,1)
    v3 = qtp.ket('2',dim=3).unit()
    
    #v4 = (0,1,1)
    v4 = (qtp.ket('1',dim=3) + qtp.ket('2',dim=3)).unit()
    
    #v5 = (0,1,-1)
    v5 = (qtp.ket('1',dim=3) - qtp.ket('2',dim=3)).unit()
    
    #v6 = (1,0,1)
    v6 = (qtp.ket('0',dim=3) + qtp.ket('2',dim=3)).unit()
    
    #v7 = (1,0,-1)
    v7 = (qtp.ket('0',dim=3) - qtp.ket('2',dim=3)).unit()
    
    #v8 = (1,1,0)
    v8 = (qtp.ket('0',dim=3) + qtp.ket('1',dim=3)).unit()
    
    #v9 = (1,-1,0)
    v9 = (qtp.ket('0',dim=3) - qtp.ket('1',dim=3)).unit()
    
    #v10 = (1,1,1)
    v10 = (qtp.ket('0',dim=3) + qtp.ket('1',dim=3) + qtp.ket('2',dim=3)).unit()
    
    #v11 = (-1,1,1)
    v11 = (-qtp.ket('0',dim=3) + qtp.ket('1',dim=3) + qtp.ket('2',dim=3)).unit()
    
    #v12 = (1,-1,1)
    v12 = (qtp.ket('0',dim=3) - qtp.ket('1',dim=3) + qtp.ket('2',dim=3)).unit()
    
    #v13 = (1,1,-1)
    v13 = (qtp.ket('0',dim=3) + qtp.ket('1',dim=3) - qtp.ket('2',dim=3)).unit()
    
    allvectors = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13]
    A_meas = []
    B_meas = []
    
    for v in allvectors:
        A_meas += [[v.proj() , id - v.proj()]]
        B_meas += [[(v.proj()).trans() , id - ((v.proj()).trans())]]
    
    datitaJoint = np.zeros((13+2,13))
    
    #Generating the joint distribution p(1,1|x,y)
    for x in range(13):
        for y in range(13):
            datitaJoint[x,y] = (eta**2) * ((rho*qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real)
    
    #Generating the marginal distribution of Alice p_A(1|x)       
    for x in range(13):
        datitaJoint[13,x] = eta*((rho*qtp.tensor(A_meas[x][0], id)).tr().real)
    
    #Generating the marginal distribution of Bob p_B(1|y) 
    for y in range(13):
        datitaJoint[13+1,y] = eta*((rho*qtp.tensor(id, B_meas[y][0])).tr().real)
    
    return datitaJoint

##############################################################################
################################# KS-18 Set ##################################
##############################################################################
def correlationsKS18(W,eta):
    """
    Generating the correlations using the KS18 set and the maximally entangled state (d=4)
    with visibility `W' and the detection efficiency by `eta'
    """

    [id, sx, sy, sz] = [qtp.qeye(4), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    
    rho0 = ((1/sqrt(4))*(qtp.ket('00',dim=4) + qtp.ket('11',dim=4) +\
                        qtp.ket('22',dim=4) + qtp.ket('33',dim=4))).proj()
    rho= W*rho0 + ((1-W)/16)*qtp.tensor(id, id)


    # The set of vectors
    #1 = (1,-1,0,0)
    v1= (qtp.ket('0',dim=4) - qtp.ket('1',dim=4)).unit() 
    
    #2 = (0,0,1,1)
    v2 = (qtp.ket('2',dim=4) + qtp.ket('3',dim=4)).unit()
    
    #3 = (0,0,1,-1)
    v3 = (qtp.ket('2',dim=4) - qtp.ket('3',dim=4)).unit()
    
    #4 = (1,0,1,0)
    v4 = (qtp.ket('0',dim=4) + qtp.ket('2',dim=4)).unit()
    
    #5 = (1,0,-1,0)
    v5 = (qtp.ket('0',dim=4) - qtp.ket('2',dim=4)).unit()
    
    #6 = (0,1,0,-1)
    v6 = (qtp.ket('1',dim=4) - qtp.ket('3',dim=4)).unit()
    
    #7 = (1,0,0,1)
    v7 = (qtp.ket('0',dim=4) + qtp.ket('3',dim=4)).unit()
    
    #8 = (0,1,1,0)
    v8 = (qtp.ket('1',dim=4) + qtp.ket('2',dim=4)).unit()
    
    #9 = (0,1,-1,0)
    v9 = (qtp.ket('1',dim=4) - qtp.ket('2',dim=4)).unit()
    
    #A = (1,0,0,0)
    v10 = (qtp.ket('0',dim=4)).unit()
    
    #B = (0,1,0,0)
    v11 = (qtp.ket('1',dim=4)).unit()
    
    #C = (0,0,0,1)
    v12 = (qtp.ket('3',dim=4)).unit()
    
    #D = (1,1,1,1)
    v13 = (qtp.ket('0',dim=4) + qtp.ket('1',dim=4) + qtp.ket('2',dim=4) + qtp.ket('3',dim=4)).unit()
    
    #E = (1,-1,1,-1)
    v14 = (qtp.ket('0',dim=4) - qtp.ket('1',dim=4) + qtp.ket('2',dim=4) - qtp.ket('3',dim=4)).unit()
    
    #F = (1,1,-1,-1)
    v15 = (qtp.ket('0',dim=4) + qtp.ket('1',dim=4) - qtp.ket('2',dim=4) - qtp.ket('3',dim=4)).unit()
    
    #G = (1,1,1,-1)
    v16 = (qtp.ket('0',dim=4) + qtp.ket('1',dim=4) + qtp.ket('2',dim=4) - qtp.ket('3',dim=4)).unit()
    
    #H = (1,1,-1,1)
    v17 = (qtp.ket('0',dim=4) + qtp.ket('1',dim=4) - qtp.ket('2',dim=4) + qtp.ket('3',dim=4)).unit()
    
    #I = (-1,1,1,1)
    v18 = (-qtp.ket('0',dim=4) + qtp.ket('1',dim=4) + qtp.ket('2',dim=4) + qtp.ket('3',dim=4)).unit()
    
    allvectors=[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18]
    A_meas=[]
    B_meas=[]
    
    for v in allvectors:
        A_meas += [[v.proj() , id - v.proj()]]
        B_meas += [[(v.proj()).trans() , id - ((v.proj()).trans())]]
    
    datitaJoint = np.zeros((18+2,18))
    
    #Generating the joint distribution
    for x in range(18):
        for y in range(18):
            #print(a,b,x,y)
            datitaJoint[x,y] = (eta**2)*((rho*qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real)
    
    #Generating the marginal distribution of Alice        
    for x in range(18):
        datitaJoint[18,x] = eta*((rho*qtp.tensor(A_meas[x][0], id)).tr().real)
    
    #Generating the marginal distribution of Bob    
    for y in range(18):
        datitaJoint[18+1,y] = eta*((rho*qtp.tensor(id, B_meas[y][0])).tr().real)
    
    return datitaJoint


##############################################################################
####################### Vectorize back and forth #############################
##############################################################################
def p2vec(P):
    
    return P.reshape(-1)

def vec2p(r):
    
    return np.reshape(r,(m+2,m))

def Ds2vec(DA,DB):

    return np.concatenate((np.kron(DA,DB),DA,DB))

##############################################################################
####################### Local Bound calculation ################################
##############################################################################

def nextDet(initD):
    """
    This function returns the next deterministic strategy/vertex.
    It makes easier the iterations.
    """
    localD = np.array(initD)
    localD[-1] = localD[-1]+1
    sizeD = len(initD)
    for i in range(sizeD):
        if localD[sizeD-1-i] >= 2:
            localD[sizeD-1-i] = 0
            if sizeD-1-i >= 1:
                localD[sizeD-1-i-1] = localD[sizeD-1-i-1] + 1
    return np.array(localD)

def gen_perms(v0):
    k = []
    N = len(v0)
    vk = np.array(v0)
    for i in range(2**N):
        k += [vk]
        vk = nextDet(vk)
    return np.array(k)


def findLocalBound(joint_coe,A_coe,B_coe):
    """
    This function follows the one implemented in matlab by QETLAB https://qetlab.com/BellInequalityMax.
    However it does not consider inequalities in the correlator form, since, everything is in CG notation.
    
        joint_coe: Matrix of coefficients c_{i,j} that correspond to the probability p(1,1|i,j). We refer to
                    the output 1 (dropping 0) just for convenience, but this choice is arbitrarily. 
        A_coe:    Vector of coefficients c_{i} that correspond to the marginal probability of Alice p_A(1|i).
        B_coe:    Vector of coefficients c_{j} that correspond to the marginal probability of Bob p_B(1|j).
    """

    bellvec = np.reshape(joint_coe,-1)
    localB = -1000000
    initA = np.zeros(m)
    initB = np.zeros(m)

    for i in range(2**m):
        for j in range(2**m):
            value = np.dot(bellvec,np.kron(initA,initB)) + np.dot(initA,A_coe) + np.dot(initB,B_coe)
            if value>localB:
                localB = value
            initB = nextDet(initB)
        initA = nextDet(initA)
    #print(f'The local bound is {localB}')
    return localB


##############################################################################
####################### random seeds #############################
##############################################################################
def randomD(m):

    return np.random.randint(2, size=m)

def randomPvec(m):
    DA = randomD(m)
    DB = randomD(m)
    P = np.kron(DA,DB)
    return np.concatenate((P,DA,DB))


def getRandomSeed(n=10):
    vecs=[(1/n)*randomPvec(m) for i in range(n)]
    vecseed = np.zeros(len(vecs[0]))
    for v in vecs:
        vecseed = vecseed+v
    return vecseed

##############################################################################
####################### Functions for Gilbert routine ########################
##############################################################################

def symmetrizeD(rvec):
    # This is a possible way to impose the Alice-Bob symmetry in the inequality.
    # Note: This was not used in the paper.
    pvec= vec2p(rvec)
    da= pvec[m]
    db= pvec[m+1]
    return 0.5*(Ds2vec(da,db) + Ds2vec(db,da))

def getNextS(rtarg,sk,skp):
    # This function returns s_k+1 from sk  and s_k' 
    def f0(x):
        y = rtarg - (x*sk+(1-x)*skp)
        return np.sqrt(y.dot(y))

    res = minimize(f0, [0.5], bounds = [[0,1]])
    p=res.x.tolist()[0]

    return p*sk + (1-p)*skp


def heuristicOracle(r):
    """
    This is an heuristic implementation of the oracle that maximize the overlap over the convex set.
    Details on the steps of this function are given in Appendix C of the paper.
    
    """
    maxIt=10**2 #This is just to ensure that the loop ends. In practice it is not needed
    PAB = vec2p(r)
    cAB = PAB[0:m]
    cA = PAB[m]
    cB = PAB[m+1]
    DA = randomD(m)
    DB = randomD(m)

    bestOver = 0
    bestVec = np.zeros(len(r))
    # Here the loop for a given seed
    for i in range(maxIt):

        # First fix DA[x]
        for y in range(m):
            vals= sum(DA[x]*cAB[x,y] for x in range(m))
            if (cB[y] + vals) > 0:
                DB[y]=1
            else:
                DB[y]=0
        #Now fix DB[y]
        for x in range(m):
            vals= sum(DB[y]*cAB[x,y] for y in range(m))
            if (cA[x] + vals)>0:
                DA[x]=1
            else:
                DA[x]=0

        rNew = np.concatenate((np.kron(DA,DB),DA,DB))
        if np.dot(rNew,r)> bestOver:
            bestOver = np.dot(rNew,r)
            bestVec = rNew
        else:
            break # when it stops improving we break the loop
    return bestOver, bestVec


def heuristicOracle_loop(r,Nmax):
    # Function to calculate the heuristic oracle over many differente seeds.
    # This is necessary to avoid getting stuck in a local maximum.
    rstart = p2vec(r)
    bestOver = 0
    bestVec = np.zeros(len(rstart))
    
    for i in range(Nmax):
        over,vec = heuristicOracle(r)
        if over>bestOver:
            bestOver = over
            bestVec=vec
    return bestVec


def heuristic_gilbertAlg(rtarg, s0, Nmax, Noracle=25, Nprint=100,saveC=0,\
                         isCluster=0,parallelData=["NodeNumber","W"]):
    print(f'***Starting heuristic_gilbertAlg***\nNumber of measurements: {m}\n')
    #tolerance = 10**(-3)
    rtarget = p2vec(rtarg)
    sk=s0
    cOld=np.zeros(len(sk))
    cNew=np.ones(len(sk))

    for i in range(Nmax):
        skp = heuristicOracle_loop(rtarget-sk, Noracle) # Calculating sk' heuristically
        sk = getNextS(rtarget,sk,skp) # Computing s_k+1
        cOld = cNew
        cNew = rtarget-sk
        
        if not (i % Nprint): # To take a look after Nprint iterations
            print(f'\nIteration number:{i}')
            now = datetime.now()
            print("time =", now)
            print(f'||sk+1 - sk||: {sqrt(np.dot(cNew-cOld,cNew-cOld))}')
            print(f'||r-sk||: {sqrt(np.dot(cNew,cNew))}')
            pnew = vec2p(cNew)
            print(F'Cxy:\n{pnew[0:6,0:6]}')
            print(F'marg CAx:\n{pnew[m][0:6]}')
            print(F'marg CBx:\n{pnew[m+1][0:6]}')
            
            # In practice it was helpful to save the results after Nprint iterations
            # and analyze the outcome ||cNew||
            np.savetxt("GA_iteration_"+str(i)+".txt",np.transpose([rtarg,sk,cNew]))

    return cNew,sk

##############################################################################
####################### GLOBAL PARAMETERS ####################################
##############################################################################
m = 18 # number of inputs for Alice and Bob m_A = m_B = m

MAXITERATION=1*10**3
PRINTAFTER = 10**2

#Any combination of parameters can be tested
W = 0.9
eta = 1
##############################################################################
##############################################################################


print(f'\n**************Running for a paramter W-eta = {W}-{eta}')

# Define the correlations that we want to test
behav = correlationsKS18(W,eta)
rtarget = p2vec(behav)

# Generate the seed inside the convex set
s0 = randomPvec(m)

# Run the implementation of the Gilbert Algorithm
# the MAXITERATION+2 is a quick fix to keep the last two iterations, instead of only the last one
bell,sk = heuristic_gilbertAlg(rtarget, s0, MAXITERATION+2, Nprint=PRINTAFTER)




