import pandas as pd
from pathlib import Path
import gmplot
import matplotlib.pyplot as plt
import numpy as np

zoom_level = 9

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "flight_data.csv"

def convertSI(df):
    #ft/s to m/s
    df['VELOCITY BODY X'] = df['VELOCITY BODY X'] * 0.3048
    df['VELOCITY BODY Y'] = df['VELOCITY BODY Y'] * 0.3048
    df['VELOCITY BODY Z'] = df['VELOCITY BODY Z'] * 0.3048

    df['VELOCITY WORLD X'] = df['VELOCITY WORLD X'] * 0.3048
    df['VELOCITY WORLD Y'] = df['VELOCITY WORLD Y'] * 0.3048
    df['VELOCITY WORLD Z'] = df['VELOCITY WORLD Z'] * 0.3048

    #ft/s/s to m/s/s
    df['ACCELERATION BODY X'] = df['ACCELERATION BODY X'] * 0.3048
    df['ACCELERATION BODY Y'] = df['ACCELERATION BODY Y'] * 0.3048
    df['ACCELERATION BODY Z'] = df['ACCELERATION BODY Z'] * 0.3048

    #ft to m
    df['PLANE ALTITUDE'] = df['PLANE ALTITUDE'] * 0.3048

    #kts to m/s
    df['AIRSPEED TRUE'] = df['AIRSPEED TRUE'] * 0.514444
    df['AIRSPEED INDICATED'] = df['AIRSPEED INDICATED'] * 0.514444

    #ft/min to m/s
    df['VERTICAL SPEED'] = df['VERTICAL SPEED'] * 0.3048 / 60

    #Delete ground velocity -> not needed
    df = df.drop('GROUND VELOCITY', axis=1)
    return df

def generateMap(latitude_list, longitude_list, latitude_list2 = None, longitude_list2 = None):
    global zoom_level
    gmap = gmplot.GoogleMapPlotter(latitude_list[0], longitude_list[0], zoom_level)
    gmap.scatter(latitude_list, longitude_list, 'blue', size=50, marker=False)

    if((latitude_list2 is not None) and (longitude_list2 is not None)):
        gmap.scatter(latitude_list2, longitude_list2, 'yellow', size=50, marker=False)
        gmap.plot(latitude_list2, longitude_list2, 'yellow', edge_width=2.5)
    gmap.plot(latitude_list, longitude_list, 'blue', edge_width=2.5)

    latTEA, lonTEA = 41.296667, 13.970556
    latVIE, lonVIE = 41.913333, 16.051111
    latLAT, lonLAT = 41.541111, 12.918056

    gmap.marker(latTEA, lonTEA, color="red", title="TEA VOR")
    gmap.marker(latVIE, lonVIE, color="red", title="VIE VOR")
    gmap.marker(latLAT, lonLAT, color="red", title="LAT VOR")

    gmap.draw("map.html")

def plotAltitude(alt, time):
    plt.plot(time, alt, label="Altitude")

    plt.xlabel("Time")
    plt.ylabel("Altitude in meters")
    plt.title("Altitude at time")
    plt.legend()
    plt.show()

def plotVelocities(TAS, IAS, time):
    plt.plot(time, TAS, label="True Airspeed")
    plt.plot(time, IAS, label="Indicated Airspeed")
    
    plt.xlabel("Time")
    plt.ylabel("Airspeed in m/s")
    plt.title("Airspeeds at time")
    plt.legend()
    plt.show()

def manualIntegrate(x, y, z, times, startConditions = None):
    XX = [0.0]
    YY = [0.0]
    ZZ = [0.0]

    if(startConditions is not None):
        XX = [startConditions[0]]
        YY = [startConditions[1]]
        ZZ = [startConditions[2]]

    for i in range(1, len(x)):
        dT = times[i]-times[i-1]
        XX.append(x[i]*dT+XX[i-1])
        YY.append(y[i]*dT+YY[i-1])
        ZZ.append(z[i]*dT+ZZ[i-1])
    return [XX, YY, ZZ]

def BFF_to_NED(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cth,  sth  = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    return np.array([
        [ cth*cy,                cth*sy,               -sth     ],
        [ sr*sth*cy - cr*sy, sr*sth*sy + cr*cy, sr*cth ],
        [ cr*sth*cy + sr*sy, cr*sth*sy - sr*cy, cr*cth ],
    ]).T

def getAngularRates(roll, pitch, yaw, time):
    angular_rates = [np.array([0, 0, 0])]
    yaw = np.unwrap(yaw)
    for i in range(1, len(roll)):
        r = roll[i]
        p = pitch[i]
        y = yaw[i]
        rotationMatrix = np.array([[1, 0, -np.sin(p)], [0, np.cos(r), np.cos(p)*np.sin(r)], [0, -np.sin(r), np.cos(p)*np.cos(r)]])

        dT = time[i]-time[i-1]
        dR = (r-roll[i-1])/dT
        dP = (p-pitch[i-1])/dT
        dY = (y-yaw[i-1])/dT

        angular_rate = np.array([dR, dP, dY])
        angular_rates.append(rotationMatrix @ angular_rate)
    return angular_rates

def trajectoryBFF_NED(df, time, initialLat, initialLon, initialAlt):
    aX = list(df['ACCELERATION BODY Z'])
    aY = list(df['ACCELERATION BODY X'])
    aZ = list(df['ACCELERATION BODY Y'])

    velocityBodyForward = df["VELOCITY BODY Z"].to_numpy() 
    velocityBodyRight = df["VELOCITY BODY X"].to_numpy() 
    velocityBodyDown = df["VELOCITY BODY Y"].to_numpy() 

    roll = df["PLANE BANK DEGREES"].to_numpy()
    pitch = df["PLANE PITCH DEGREES"].to_numpy()
    yaw = df["PLANE HEADING DEGREES TRUE"].to_numpy()

    angular_rates = getAngularRates(roll, pitch, yaw, time)

    aX_NED = []
    aY_NED = []
    aZ_NED = []
    for i in range(len(aX)):
        R = BFF_to_NED(roll[i], pitch[i], yaw[i])
        aBody = np.array([
            aX[i],
            aY[i],
            aZ[i],
        ])

        aInertial = aBody + np.cross(angular_rates[i], np.array([velocityBodyForward[i], velocityBodyRight[i], velocityBodyDown[i]]))

        a_world = R @ aInertial
        #a_world[2] += 9.80665
        aX_NED.append(a_world[0])
        aY_NED.append(a_world[1])
        aZ_NED.append(a_world[2])
    v0 = [
        float(df['VELOCITY WORLD Z'].iloc[0]),   #North
        float(df['VELOCITY WORLD X'].iloc[0]),   #East
        float(-df['VELOCITY WORLD Y'].iloc[0]),  #Down
    ]
    velocities = manualIntegrate(aX_NED, aY_NED, aZ_NED, time, startConditions=v0)
    positions = manualIntegrate(velocities[0], velocities[1], velocities[2], time)
    north = positions[0]
    east = positions[1]
    down = positions[2]

    latitudes = [initialLat]
    longitudes = [initialLon]
    altitudes = [initialAlt]
    
    for i in range(1, len(north)):
        altitudes.append(altitudes[0] - down[i])
        latitudes.append(initialLat + north[i] * 0.000009044)
        longitudes.append(initialLon + east[i] * 0.00000898)

    plt.plot(time, altitudes, label="Altitude")

    #plt.xlabel("Absolute time")
    #plt.ylabel("Altitude in meters")
    #plt.title("Altitude at time")
    #plt.legend()
    #plt.show()

    return[longitudes, latitudes, altitudes]


df_raw = pd.read_csv(file_path, parse_dates=['timestamp'])
df = convertSI(df_raw)

latitude_list = list(df['PLANE LATITUDE'])
longitude_list = list(df['PLANE LONGITUDE'])

TAS = list(df['AIRSPEED TRUE'])
IAS = list(df['AIRSPEED INDICATED'])
time = list(df['ABSOLUTE TIME'])

alt = list(df['PLANE ALTITUDE'])

#plotAltitude(alt, time)
#plotVelocities(TAS, IAS, time)
#generateMap(latitude_list, longitude_list)

trajectory_NED = trajectoryBFF_NED(df, time, latitude_list[0], longitude_list[0], alt[0])
generateMap(latitude_list, longitude_list, longitude_list2=trajectory_NED[0], latitude_list2=trajectory_NED[1])