/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Create random number generator
	default_random_engine gen;
	
	// Create normal distributions
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	// Loop through and get poitnts
	particles.clear(); 					// Start with empty vector
	double weight{1.0 / num_particles};	// Default uniform weight
	for (int i =0; i < num_particles; ++i) {
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		particles.push_back(Particle{i,
							dist_x(gen),
							dist_y(gen),
							dist_theta(gen),
							weight,
							associations,
							sense_x,
							sense_y});
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Create random number generator
	default_random_engine gen;
	
	// Loop through all particles and predict
	for (Particle &p : particles) {	
		// Create normal distributions
		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);
		
		// Update with predicted positions
		p.x = dist_x(gen) + (velocity / yaw_rate) * 
			(sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
		p.y = dist_y(gen) + (velocity / yaw_rate) * 
			(cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
		p.theta = dist_theta(gen) + delta_t * yaw_rate;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for (LandmarkObs &pre : predicted) {
		double distance{std::numeric_limits<double>::max()};
		for (const LandmarkObs &obs : observations) {
			if (dist(pre.x, pre.y, obs.x, obs.y) < distance) {
				// Associate prediction with observation
				pre.id = obs.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// Create random number generator and discrete distributions
	default_random_engine gen;
	discrete_distribution<int> intDist(0, num_particles);
	uniform_real_distribution<double> realDist(0, 1);
	
	// Create new particle vector
	vector<Particle> newParticles;
	
	// Get max weight of all particles
	auto itr = max_element(particles.begin(),particles.end(),
		[](const Particle &p1, const Particle &p2){return p1.weight < p2.weight;});
	double mw{itr->weight};
	
	// Resample
	double beta{0.0};
	int index{intDist(gen)};
	for (int i = 0; i < num_particles; ++i) {
		beta += realDist(gen) * 2.0 * mw;
		while (beta > particles[i].weight) {
			beta -= particles[i].weight;
			index = (index + 1) % num_particles;
		}
		newParticles.push_back(particles[index]);
	}
	
	// Copy new particles to particles
	particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
