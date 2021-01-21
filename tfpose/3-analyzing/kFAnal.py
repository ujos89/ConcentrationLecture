import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


dataPath = '../0-data/data_prediction/4R16R1S_es_100/90_for_kf.pkl'

dfRaw = pd.read_pickle(dataPath)


# print(dfRaw)


tFrame =  50.
tFps = 20.
tWin = tFrame * (1/tFps)
tMax = tWin * len(dfRaw)


n4error = tFrame

error = 12 / np.sqrt(n4error)



print(error)

dfMeasure = dfRaw['prediction']

print(dfMeasure)


dfMeasureCut = pd.DataFrame() 

nCut = 10

for i in range(0, len(dfRaw), nCut) :
    dfMeasureOne =  pd.DataFrame([dfMeasure.loc[i]])
    dfMeasureCut = pd.concat([dfMeasureCut, dfMeasureOne])

print(dfMeasureCut)



tWinCut = nCut * tWin

tNp = np.arange(0, tMax, step=tWinCut)

u = np.array([[dfMeasureCut.iloc[1]], [dfMeasureCut.iloc[1]]])

print(dfMeasureCut.iloc[1])

# print(u)

# print(tNp)

# plt.scatter(tNp, dfMeasureCut)

# plt.show()


# Covariance for EKF simulation
Q = np.diag([
    0.9,  # variance of location on x-axis
    0.9,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    0.9  # variance of velocity
]) ** 2  # predict state covariance

R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([1.0, 1.0]) ** 2

#DT = 2.0  # time tick [s]
DT = 1  # time tick [s]
#DT = 5.0  # time tick [s]
SIM_TIME = 60.0 * 10# simulation time [s]

show_animation = True
# show_animation = False


def calc_input():
    v = 0.01  # [m/s]
    # yawrate = 0.1  # [rad/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u, uTrue):
    xTrue = motion_model(xTrue, uTrue)

    # add noise to gps x-y
    # z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
    z = observation_model(xTrue)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)
    # ud = u

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):
  
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])


    F = np.array([[1.0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    # B = np.array([[DT * math.cos(x[2, 0]), 0],
    #                [DT * math.sin(x[2, 0]), 0],
    #                [0.0, DT],
    #                [1.0, 0.0]])


    # B = np.array([[DT * x[2, 0], 0],
    #               [DT * x[2, 0], 0],
    #               [0.0, DT],
    #               [1.0, 0.0]])
    # print('======================= shape =======================')
    # print(np.shape(u))
    # print('======================= type =======================')
    # print(type(u))
    # print(u)
    # mU = u[0,0]
    # print(mU)

    # B = np.array([[ DT , 0],
    #                [mU ,  0],
    #                [0.0, DT],
    #                [1.0, 0.0]])

    B = np.array([[ DT , 0],
                   [DT ,  0],
                   [0.0, DT],
                   [1.0, 0.0]])


    # print(B)
    # print(u)

    x = F @ x + B @ u
    # print(x)
    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        #[1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [1.0, 0.0, DT , DT],
        #[0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 1.0, DT , DT ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
    print(__file__ + " start!!")

    time = 0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    while SIM_TIME >= time:
        time += DT
        print(time)

        # u = dfMeasureCut[time].values()
        
        
        uP = calc_input()

        # print(np.shape(u))

        # print('----------------')

       
        # ux = dfMeasureCut.iloc[time]
        # ux = ux.iloc[0]
        # print(ux)
       
        ux = dfRaw.iloc[time]
        ux = ux['prediction']

       
        u = np.array([[ux], [ux]])
        uDT = np.array([[time], [ux]])
        # u = calc_input()

        # print(np.shape(u))
        # print(u)

        # xTrue, z, xDR, ud = observation(xTrue, xDR, u)
        xTrue, z, xDR, ud = observation(xTrue, xDR, uP, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        # hz = np.hstack((hz, z))
        hz = np.hstack((hz, uDT))
  
        if show_animation:
            print(np.shape(hxEst))
            print(hxEst)
            print(np.shape(uDT))
            print(uDT)
            # print(hz[0, :], hz[1, :])
            # print('----------------')
            # print(hxEst[0, :], hxEst[1, :])
            # # plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            # plt.plot(hxTrue[0, :].flatten(),
            #          hxTrue[1, :].flatten(), "-b")
            # plt.plot(hxDR[0, :].flatten(),
            #          hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            # plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(1)
            plt.ylim(0,1)
    # print(hxEst)

if __name__ == '__main__':
    main()