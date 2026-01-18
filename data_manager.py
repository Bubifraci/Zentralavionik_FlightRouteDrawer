import pandas as pd
from pathlib import Path
import gmplot
import matplotlib.pyplot as plt

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

def generateMap(latitude_list, longitude_list):
    global zoom_level
    gmap = gmplot.GoogleMapPlotter(latitude_list[0], longitude_list[0], zoom_level)
    gmap.scatter(latitude_list, longitude_list, 'blue', size=50, marker=False)
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
    plt.ylabel("Airspeed in kts")
    plt.title("Airspeeds at time")
    plt.legend()
    plt.show()

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
generateMap(latitude_list, longitude_list)