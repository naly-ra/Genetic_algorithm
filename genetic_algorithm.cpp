#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <algorithm>
#include <numeric>

// Make sure Eigen library and dependencies are installed as it is necessary to run the code.


using namespace std;
using namespace Eigen;

// As it uses data from 2019 where the rates were negative or really low we set the risk free rate to 0
double RISK_FREE_RATE = 0.00;

MatrixXd DailyReturnsStocks(const string& filePath) {
    
    // Function to read the file into a vector of vector array and 
    // convert it to a MatrixXd to use linear algebra functions.

    ifstream inputFile(filePath);


    if (!inputFile.is_open()) {
        cerr << "Error opening file: " << filePath << endl;
        exit(1);
    }


    vector<vector<float>> matrix;

    
    string line;
    while (getline(inputFile, line)) {
       
        istringstream iss(line);
        vector<float> row;

        
        float value;
        while (iss >> value) {
            row.push_back(value);
        }

        
        matrix.push_back(row);
    }


    inputFile.close();

    // Convert the vector<vector<float>> to MatrixXd

    MatrixXd resultMatrix(matrix.size(), matrix[0].size());
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            resultMatrix(i, j) = matrix[i][j];
        }
    }

    return resultMatrix;
}

void printMatrix(const MatrixXd& matrix) {
    cout << matrix << endl;
}

class Portfolio {
private:

    //matrix of daily returns 
    MatrixXd Matrix = DailyReturnsStocks("daily_returns.txt"); 
    MatrixXd covMatrix;
    VectorXd weights;
    double portfolio_std;

    mt19937 rng;

public:

    Portfolio() : rng(static_cast<unsigned>(time(nullptr))) {
        initializeDefaultWeights();
    }

    
    Portfolio(const VectorXd& initialWeights) : rng(static_cast<unsigned>(time(nullptr))) {
        setWeights(initialWeights);
    }


    void setWeights(const VectorXd& newWeights) {
        if (newWeights.size() == 10) {
            weights = newWeights;
            // Normalize weights to ensure their sum is 1
            weights /= weights.sum();
        } else {
            cerr << "Error: Invalid size for newWeights. Expected size: 10." << endl;
        }
    }

    void initializeDefaultWeights() {
        
        // Initialize weights randomly between 0 and 1
        weights = VectorXd::Random(10).array().abs();

        // Normalization of weights
        weights /= weights.sum();
    }

    void printWeights() const {
        cout << "Weights: \n" << weights << endl;
    }

    void printMatrix() const {
        cout << "Matrix: \n" << Matrix << endl;
    }

    VectorXd getWeights(){
        return weights;
    }

    void cov(){
        MatrixXd centered = Matrix.rowwise() - Matrix.colwise().mean();
        this->covMatrix = (centered.transpose() * centered) / double(Matrix.rows() - 1);
    }

    // compute the standard deviation of the portfolio
    double std() {
        this->cov();
        // We return the opposite of the standard deviation to turn it into a maximization problem
        return - sqrt(weights.transpose() * covMatrix * weights) * sqrt(254);
}

    void std_dev() {

        // Standard deviation of portfolio returns
        VectorXd daily_r = Matrix * weights; 
        double variance = (daily_r.array() - daily_r.mean()).square().sum() / (daily_r.size() - 1);

        portfolio_std = sqrt(variance);
    }

    double mean_returns() {

        VectorXd daily_r = Matrix * weights;
        return daily_r.mean() * sqrt(254);
    }


    double fitness(){

        // Sharpe Ratio
        double _return = this-> mean_returns();
        this->std_dev();
        return (_return - RISK_FREE_RATE) / portfolio_std;
    }
    void mutation(){
        


        default_random_engine generator;
        uniform_int_distribution<int> discrete_distribution(0,9);
        uniform_real_distribution<double> distribution(0.0,1.0);

        int index_random_1 = discrete_distribution(generator);
        int index_random_2 = discrete_distribution(generator);

        double element_random_1 = distribution(generator);
        double element_random_2 = distribution(generator);

        weights(index_random_1) = element_random_1;
        weights(index_random_2) = element_random_2;

        weights /= weights.sum();
    }

};


Portfolio blendCrossover(Portfolio parent1, Portfolio parent2, double alpha = 0.7) {
 
    if (parent1.getWeights().size() != parent2.getWeights().size()) {
        cerr << "Error: Parents have different number of weights." << endl;
        exit(1);
    }

 
    Portfolio child;

    // If parent1 has better performance than parent2 the child inherits more weights from it.
    if (parent1.fitness() >= parent2.fitness()) {

    VectorXd child_weights = alpha * parent1.getWeights() + (1.0 - alpha) * parent2.getWeights(); 
    child.setWeights(child_weights);
    }

    VectorXd child_weights = alpha * parent2.getWeights() + (1.0 - alpha) * parent1.getWeights();
    child.setWeights(child_weights);

    return child;
}

Portfolio NaiveCrossover(Portfolio parent1, Portfolio parent2){

    Portfolio child;
    int size = parent1.getWeights().size();
    int half_size = size / 2; 

    // Initialize an empty vector of size 10
    VectorXd child_weights(size);
    
    // Child receives one half from parent1 and the other half from parent2
    child_weights.head(half_size) = parent1.getWeights().head(half_size);
    child_weights.tail(half_size) = parent2.getWeights().tail(half_size);

    child.setWeights(child_weights);

    return child;

}

Portfolio TwoPointCrossover(Portfolio parent1, Portfolio parent2) {
    Portfolio child;

    int size = parent1.getWeights().size();

    VectorXd child_weights(size);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> distribution(0, size - 1);

    int crossover_point1 = distribution(gen);
    int crossover_point2 = distribution(gen);

    // Ensure crossover_point1 is less than crossover_point2
    if (crossover_point1 > crossover_point2) {
        std::swap(crossover_point1, crossover_point2);
    }

    // Copy genetic material from parents to child
    child_weights.segment(0, crossover_point1) = parent1.getWeights().segment(0, crossover_point1);
    child_weights.segment(crossover_point1, crossover_point2 - crossover_point1 + 1) = parent2.getWeights().segment(crossover_point1, crossover_point2 - crossover_point1 + 1);
    child_weights.segment(crossover_point2 + 1, size - crossover_point2 - 1) = parent1.getWeights().segment(crossover_point2 + 1, size - crossover_point2 - 1);

    child.setWeights(child_weights);

    return child;
}




int main() {
    const int POPULATION_SIZE = 100;
    const int NUM_GENERATIONS = 100;
    const double MUTATION_RATE = 0.01;
    //Percentage of the best portfolios to keep
    const double ELITISM_PERCENTAGE = 0.1;  

    vector<Portfolio> population;

    // Initialize the population
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        Portfolio portfolio;
        population.push_back(portfolio);
    }

    // Main genetic algorithm loop
    for (int generation = 1; generation <= NUM_GENERATIONS; ++generation) {
        
        // Evaluate fitness for each portfolio in the population
        vector<double> fitnessScores;
        for (auto& portfolio : population) {
            fitnessScores.push_back(portfolio.fitness());
        }

        // Find the index of the best portfolios
        vector<int> sortedIndices(POPULATION_SIZE);
        iota(sortedIndices.begin(), sortedIndices.end(), 0);
        sort(sortedIndices.begin(), sortedIndices.end(), [&](int i, int j) {
            return fitnessScores[i] > fitnessScores[j];
        });

        // Calculate the number of portfolios to keep based on elitism percentage
        int numElites = static_cast<int>(ELITISM_PERCENTAGE * POPULATION_SIZE);

        // Print information about the best portfolio and the average portfolio
        if (generation % 10 == 0 || generation == 1) {
            cout << "Generation " << generation << " - Best Portfolio:" << endl;
                population[sortedIndices[0]].printWeights();
                cout << "Sharpe Ratio of the best portfolio: " << abs(fitnessScores[sortedIndices[0]]) << endl;
            
            double averageSharpe = accumulate(fitnessScores.begin(), fitnessScores.end(), 0.0) / fitnessScores.size();
            cout << "Average Sharpe Ratio: " << abs(averageSharpe) << endl << endl;
        }

        // Initialize a new generation 
        vector<Portfolio> newPopulation;
        for (int i = 0; i < numElites; ++i) {
            
            // We keep the best portfolios from generation n-1
            newPopulation.push_back(population[sortedIndices[i]]);
        }

        // Rest of the population are made using crossover
        for (int i = numElites; i < POPULATION_SIZE; ++i) {
            
            // Parents are chosen randomly to reproduce themselves
            Portfolio parent1 = population[rand() % POPULATION_SIZE];
            Portfolio parent2 = population[rand() % POPULATION_SIZE];

            // Crossover, here we select the crossover method wanted
            Portfolio child = TwoPointCrossover(parent1, parent2);

            // Mutation happens at child level 
            if (static_cast<double>(rand()) / RAND_MAX < MUTATION_RATE) {
                child.mutation();
            }

            newPopulation.push_back(child);
        }

        // Go to the next generation
        population = newPopulation;
    }

    return 0;
}
