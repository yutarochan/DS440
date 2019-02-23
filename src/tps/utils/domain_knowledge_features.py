# effectiveTemperature is from the new dataset
# i_ror, i_dor, i_depth are given in the original dataset. 

pi = 3.14

surfaceTemperature = effectiveTemperature
stellarRadius = (surfaceTemperature ^ 4) * 5.67 * 10 ^ -8
planetRadius = i_ror * stellarRadius
planetCrossArea = pi * planetRadius * planetRadius
planetSemiMajorAxis = i_dor * stellarRadius

starMass = (4 * pi * pi * pow(planetSemiMajorAxis, 3)) / (pow(i_period * 60 * 60 * 24, 2) * 6.67408 * 10 ^ -11)
starCrossArea = planetCrossArea * i_depth
starRadius = pow(starCrossArea / pi, 0.5)