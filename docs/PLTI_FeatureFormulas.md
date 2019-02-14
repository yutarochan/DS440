# PLTI Feature Formulas
The following notes document some of the formulas we can derive from the feature set.
By taking a algebraic route, we can potentially unlock new features which we can utilize for our work.

### Period
`i_period`: Period of the 

![period](https://wikimedia.org/api/rest_v1/media/math/render/svg/84a962a4e5da8183f6a38ef23d0a5a9452667148)
![semi-major axis](https://wikimedia.org/api/rest_v1/media/math/render/svg/acc0794d19344d83b82ba93518c68a0fccd0b31b)

* `a`: Orbit's Semi-Major Axis
* `mu`: GM - The Standard Gravitational Parameter
  * G: Universal Gravitational Constant (6.67408 × 10-11 m3 kg-1 s-2)
  * M: The larger mass of the orbiting object (greater than m).
 
![mu](https://wikimedia.org/api/rest_v1/media/math/render/svg/15eee163b440513ad6928bb584fc8ebb0fdb2f3c)
![elliptic orbits](https://wikimedia.org/api/rest_v1/media/math/render/svg/5afd19a7e5f75bc3c9c618e1d54b71d50b2392d4)

Given that we have the mass of the larger orbiting body, we can derive the following:

![](https://latex.codecogs.com/gif.latex?M%20%3D%20V%5Crho%20%3D%20%5Cfrac%7B4%7D%7B3%7D%20%5Cpi%20a%5E3%5Crho)

Given that we have the mass of two of the orbiting bodies, we can find the orbital period:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/3e3b818e77117af84080cc6130389c9908f2fde8)

### Transit Duration
<img src="https://www.paulanthonywilson.com/wp-content/uploads/2014/08/complicado-700x330.png" width="400px">
<img src="https://s0.wp.com/latex.php?latex=++T_%7Bdur%7D+%3D+P%5Cfrac%7B%5Calpha%7D%7B2%5Cpi%7D+%3D+%5Cfrac%7BP%7D%7B%5Cpi%7D%5Csin%5E%7B-1%7D+%5Cleft%28+%5Cfrac%7Bl%7D%7Ba%7D+%5Cright%29+%3D+%5Cfrac%7BP%7D%7B%5Cpi%7D%5Csin%5E%7B-1%7D+%5Cleft%28+%5Cfrac%7B%5Csqrt%7B%28R_%2A+%2B+R_P%29%5E2+-+%28bR_%2A%29%5E2%7D%7D%7Ba%7D+%5Cright%29++%5Clabel%7Beq%3Atransit-duration%7D++&bg=T&fg=000000&s=4" width="400px">

### Impact Parameter
![](https://s0.wp.com/latex.php?latex=++b+%3D+%5Cfrac%7Ba%5Ccos+i%7D%7BR_%2A%7D++&bg=T&fg=000000&s=4)
<img src="https://www.paulanthonywilson.com/wp-content/uploads/2014/08/impact-parameter1.png" width="300px">

### Stellar Radius
![](https://latex.codecogs.com/gif.latex?r_s%20%3D%20%5Csqrt%7B%5Cdfrac%7BL%7D%7B4%5Cpi%20%5Csigma%20T%5E4%7D%7D)

### Luminosity
![](https://latex.codecogs.com/gif.latex?L%20%3D%204%20%5Cpi%20R%5E2%20%5Csigma%20T%5E4)

---
### Resources:
https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/
