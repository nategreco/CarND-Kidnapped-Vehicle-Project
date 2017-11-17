# **Kidnapped Vehicle Project**

### Vehicle localization using particle filters

---

**Kidnapped Vehicle Project**

The goals / steps of this project are the following:

* Complete starter code to succesfully implement a particle filter for vehicle localization


[//]: # (Image References)
[image1]: ./Dataset1.png "Results"

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

My project includes the following files:
* [main.cpp](../src/main.cpp) main program that runs the IO server (given from starter code)
* [particle_filter.cpp](../src/particle_filter.cpp) contains the particle filter implementation
* [particle_filter.h](../src/particle_filter.h) header file for [particle_filter.cpp](../src/particle_filter.cpp)
* [helper_functions.cpp](../src/helper_functions.cpp) contains various helper functions for particle filter implementation
* [helper_functions.h](../src/helper_functions.h) header file for [helper_functions.cpp](../src/helper_functions.cpp)
* [Dataset1.png](./Dataset1.png) result image from simulation


### Discussion

#### 1. Implementation

Implementation of the particle filter mainly followed the methodology out lined in the courses.  The implementations that were left to us were the data assocation function and piecing it all together.  Additionally we had to implement a multivariate normal distribution to handle the weighting of multiple sensor measurements.  Additionally we had the ability to tweak the number of particles.  More particles produces higher accuracy, however, the trade off is performance.  This must be closely monitored in a real time application.

#### 2. Results

Resulting error was 0.115, 0.111, and 0.004, in the x, y, and yaw respectively.

Dataset 1:

![Dataset 1][image1]
