// Path Integral Monte Carlo program for the 1-D harmonic oscillator

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

double ran_uniform() {

    return rand()/(double)RAND_MAX;
}

double V(double x)          // potential energy function
{
    // use units such that m = 1 and omega_0 = 1
    return 0.5 * pow(x, 2.0);
}

double dVdx(double x)       // derivative dV(x)/dx used in virial theorem
{
    return x;
}

double tau;                 // imaginary time period
int M;                      // number of time slices
double Delta_tau;           // imaginary time step
vector<double> x;           // displacements from equilibrium of M "atoms"

int n_bins;                 // number of bins for psi histogram
double x_min;               // bottom of first bin
double x_max;               // top of last bin
double dx;                  // bin width
vector<double> P;           // histogram for |psi|^2

double delta;               // Metropolis step size in x
int MC_steps;               // number of Monte Carlo steps in simulation

void initialize()
{
    Delta_tau = tau / M;
    x.resize(M);
    x_min = -x_max;
    dx = (x_max - x_min) / n_bins;
    P.resize(n_bins);
    cout << " Initializing atom positions using gsl::ran_uniform()" << endl;
    for (int j = 0; j < M; ++j) {
        x[j] = (2 * ran_uniform() - 1) * x_max;
    }
}

bool Metropolis_step_accepted(double& x_new)
{
    // choose a time slice at random
    int j = int(ran_uniform() * M);
    // indexes of neighbors periodic in tau
    int j_minus = j - 1, j_plus = j + 1;
    if (j_minus < 0) j_minus = M - 1;
    if (j_plus > M - 1) j_plus = 0;
    // choose a random trial displacement
    double x_trial = x[j] + (2 * ran_uniform() - 1) * delta;
    // compute change in energy
    double Delta_E = V(x_trial) - V(x[j])
        + 0.5 * pow((x[j_plus] - x_trial) / Delta_tau, 2.0)
        + 0.5 * pow((x_trial - x[j_minus]) / Delta_tau, 2.0)
        - 0.5 * pow((x[j_plus] - x[j]) / Delta_tau, 2.0)
        - 0.5 * pow((x[j] - x[j_minus]) / Delta_tau, 2.0);
    if (Delta_E < 0.0 || exp(- Delta_tau * Delta_E) > ran_uniform()) {
        x_new = x[j] = x_trial;
        return true;
    } else {
        x_new = x[j];
        return false;
    }
}

int main()
{
    cout << " Path Integral Monte Carlo for the Harmonic Oscillator\n"
         << " -----------------------------------------------------\n";

    // set simulation parameters
    cout << " Imaginary time period tau = " << (tau = 10.0)
         << "\n Number of time slices M = " << (M = 100)
         << "\n Maximum displacement to bin x_max = " << (x_max = 4.0)
         << "\n Number of histogram bins in x = " << (n_bins = 100)
         << "\n Metropolis step size delta = " << (delta = 1.0)
         << "\n Number of Monte Carlo steps = " << (MC_steps = 100000)
         << endl;

    initialize();
    int therm_steps = MC_steps / 5, acceptances = 0;
    double x_new = 0;
    cout << " Doing " << therm_steps << " thermalization steps ...";
    for (int step = 0; step < therm_steps; ++step)
        for (int j = 0; j < M; ++j)
            if (Metropolis_step_accepted(x_new))
                ++acceptances;
    cout << "\n Percentage of accepted steps = "
         << acceptances / double(M * therm_steps) * 100.0 << endl;

    double E_sum = 0, E_sqd_sum = 0;
    P.clear();
    acceptances = 0;
    cout << " Doing " << MC_steps << " production steps ...";
    for (int step = 0; step < MC_steps; ++step) {
        for (int j = 0; j < M; ++j) {
            if (Metropolis_step_accepted(x_new))
                ++acceptances;
            // add x_new to histogram bin
            int bin = int((x_new - x_min) / (x_max - x_min) * n_bins);
            if (bin >= 0 && bin < M)
                P[bin] += 1;
            // compute Energy using virial theorem formula and accumulate
            double E = V(x_new) + 0.5 * x_new * dVdx(x_new);
            E_sum += E;
            E_sqd_sum += E * E;
        }
    }

    // compute averages
    double values = MC_steps * M;
    double E_ave = E_sum / values;
    double E_var = E_sqd_sum / values - E_ave * E_ave;
    cout << "\n <E> = " << E_ave << " +/- " << sqrt(E_var / values)
         << "\n <E^2> - <E>^2 = " << E_var << endl;
    ofstream ofs("pimc.out");
    E_ave = 0;
    for (int bin = 0; bin < n_bins; ++bin) {
        double x = x_min + dx * (bin + 0.5);
        ofs << " " << x << '\t' << P[bin] / values << '\n';
        E_ave += P[bin] / values * (0.5 * x * dVdx(x) + V(x));
    }
    ofs.close();
    cout << " <E> from P(x) = " << E_ave << endl;
    cout << " Probability histogram written to file pimc.out" << endl;

    return 0;
}