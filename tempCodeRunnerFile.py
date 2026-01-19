def flat_earth_trajectory(lat, lon, time, multiplier=1):
    a = 6378137
    b = 6356752.3142
    f = (a-b)/a
    e2 = f*(2-f)

    initialLat = lat[0]
    initialLon = lon[0]

    vN = []
    vE = []
    for i in range(len(lon)):
        phi = lat[i]
        delta = lon[i]
        vN.append(multiplier * getMeridian(phi, a, e2)*(phi-initialLat))
        vE.append(multiplier * getPrimeVertical(phi, a, e2)*np.cos(initialLat)*(delta - initialLon))
    positions = manualIntegrate(vN, vE, None, time)
    #Since psi is zero, the North/East coordinates are also the flat earth coordinates
    return positions