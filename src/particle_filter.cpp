#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle_filter.h"
using namespace std;
void ParticleFilter::init(double x, double y, double theta, double std[]){
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//  x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	int i;
	for (i = 0; i < num_particles; i++) {
		Particle init_p;
		init_p.id = i;
		init_p.x = dist_x(gen);
		init_p.y = dist_y(gen);
		init_p.theta = dist_theta(gen);
		init_p.weight = 1.0;
		particles.push_back(init_p);
		weights.push_back(init_p.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
   //TODO: Add measurements to each particle and add random Gaussian noise.
	//NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//http://www.cplusplus.com/reference/random/default_random_engine/
	// Add measurements to random Gaussian noise.
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	static default_random_engine gen;
	for (int i = 0; i<num_particles; i++) {
		// two scenarios,yaw_rate is "0"and ">0"
		if (fabs(yaw_rate)<0.0001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		// Add noise 
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	// observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	// implement this method and use it as a helper during the updateWeights phase.
   int i,j;
	int closest_id = -1;
	for (i = 0; i < observations.size(); i++) {
      double min_dist = numeric_limits<double>::max();
		 double obs_x = observations[i].x;
		 double obs_y = observations[i].y;
		 for (j = 0; j < predicted.size(); j++) {
			double pred_x = predicted[j].x;
			double pred_y = predicted[j].y;
			int  pred_id = predicted[j].id;
			double current_dist = dist(obs_x, obs_y, pred_x, pred_y);
			// look for minium distance
			if (current_dist < min_dist) {
				min_dist = current_dist;
				closest_id = pred_id;
			 }
		 }
		// Minimum distance ID
		observations[i].id = closest_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	// more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	// according to the MAP'S coordinate system. You will need to transform between the two systems.
	// Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	// The following is a good resource for the theory:
	// https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	// and the following is a good resource for the actual equation to implement (look at equation 
	// 3.33
	// http://planning.cs.uiuc.edu/node99.html
  int i,j,h;
  double total_weight = 0.0;
  for (i = 0; i < num_particles; i++) {
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;
    // step1 :Transform  vehicle cordinates to map cordinates
     vector<LandmarkObs> transformed_observations;
     for (j = 0; j < observations.size(); j++) {
         LandmarkObs transformed_obs;
        transformed_obs.id = j;
        transformed_obs.x = particle_x + (cos(particle_theta) * observations[j].x) - (sin(particle_theta) * observations[j].y);
        transformed_obs.y = particle_y + (sin(particle_theta) * observations[j].x) + (cos(particle_theta) * observations[j].y);
        transformed_observations.push_back(transformed_obs);
     }
    //Step 2: Look for landmarks within the sensor's measuring range
		vector <LandmarkObs> predicted_landmarks;
		for (h = 0; h< map_landmarks.landmark_list.size(); h++) {
			LandmarkObs landmark;
			landmark.x = map_landmarks.landmark_list[h].x_f;
			landmark.y = map_landmarks.landmark_list[h].y_f;
			landmark.id = map_landmarks.landmark_list[h].id_i;
			double dx = particle_x - landmark.x;
			double dy = particle_y - landmark.y;
			if ((fabs(dx) <=sensor_range) && (fabs(dy) <= sensor_range)){
				predicted_landmarks.push_back(landmark);
			 }
		}
    //step3 : Associate observations with predicted landmarks
    dataAssociation(predicted_landmarks, transformed_observations);
    //Step4: Calculate the weight
    particles[i].weight = 1.0;
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double sigma_x_2 = pow(sigma_x, 2);
    double sigma_y_2 = pow(sigma_y, 2);
    double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
    int k, l;
    //Calculate the weight of particle based on the multivariate Gaussian probability 
    for (k = 0; k < transformed_observations.size(); k++) {
      double trans_obs_x = transformed_observations[k].x;
      double trans_obs_y = transformed_observations[k].y;
      double trans_obs_id = transformed_observations[k].id;
      double multi_prob = 1.0;
      for (l = 0; l < predicted_landmarks.size(); l++) {
        double pred_landmark_x = predicted_landmarks[l].x;
        double pred_landmark_y = predicted_landmarks[l].y;
        double pred_landmark_id = predicted_landmarks[l].id;
        if (trans_obs_id == pred_landmark_id) {
            double exponent = (pow((trans_obs_x - pred_landmark_x), 2) / (2.0 * sigma_x_2)) +
                            (pow((trans_obs_y - pred_landmark_y), 2) / (2.0 * sigma_y_2));
            double multi_prob = normalizer * exp(-exponent);
            particles[i].weight *= multi_prob;
           }
         }
       }
    total_weight+= particles[i].weight;
  }
  ///Step 5: Normalize the weights of all particles 
  for (int i = 0; i < particles.size(); i++) {
    particles[i].weight /= total_weight;
    weights[i] = particles[i].weight;
    }
}
void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resample_particles;
	default_random_engine gen;
	//Generate random particle index
	uniform_int_distribution<int> particle_index(0, num_particles - 1);
	int index = particle_index(gen);
	double beta = 0.0;
	double max_weight = 2.0 * *max_element(weights.begin(), weights.end());
	for (int i = 0; i < particles.size(); i++) {
		uniform_real_distribution<double> random_weight(0.0, max_weight);
		beta += random_weight(gen);
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resample_particles.push_back(particles[index]);
	}
	particles = resample_particles;
}

Particle SetAssociations(Particle& particle, const std::vector<int>& associations,
	const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates
   particle.associations= associations;
   particle.sense_x = sense_x;
   particle.sense_y = sense_y;
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