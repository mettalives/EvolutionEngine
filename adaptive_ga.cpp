#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

// ============================================================
// 1. Individual
// ============================================================

struct Individual {
    std::vector<double> genes;
    double fitness = std::numeric_limits<double>::lowest();

    explicit Individual(size_t n = 0) : genes(n) {}

    void evaluate(const std::function<double(const std::vector<double>&)>& func) {
        fitness = func(genes);
    }

    Individual clone() const {
        Individual copy;
        copy.genes   = genes;
        copy.fitness = fitness;
        return copy;
    }
};

// ============================================================
// 2. Strategies (unchanged)
// ============================================================

class SelectionStrategy {
public:
    virtual ~SelectionStrategy() = default;
    virtual Individual select(const std::vector<Individual>& pop,
                              std::mt19937& rng) const = 0;
};

class CrossoverStrategy {
public:
    virtual ~CrossoverStrategy() = default;
    virtual std::pair<Individual, Individual> crossover(
        const Individual& p1, const Individual& p2,
        double min, double max, std::mt19937& rng) const = 0;
};

class MutationStrategy {
public:
    virtual ~MutationStrategy() = default;
    virtual void mutate(Individual& ind, double rate,
                        double min, double max, std::mt19937& rng) const = 0;
};

class TournamentSelection final : public SelectionStrategy {
    const int k;
public:
    explicit TournamentSelection(int size = 5) : k(size) {}
    Individual select(const std::vector<Individual>& pop,
                      std::mt19937& rng) const override {
        if (pop.empty()) throw std::runtime_error("Empty population");
        std::uniform_int_distribution<size_t> dist(0, pop.size() - 1);
        size_t best = dist(rng);
        for (int i = 1; i < k; ++i) {
            size_t idx = dist(rng);
            if (pop[idx].fitness > pop[best].fitness) best = idx;
        }
        return pop[best].clone();
    }
};

class SBXCrossover final : public CrossoverStrategy {
    const double eta, prob;
public:
    SBXCrossover(double di = 20.0, double p = 0.9) : eta(di), prob(p) {}
    std::pair<Individual, Individual> crossover(
        const Individual& p1, const Individual& p2,
        double min, double max, std::mt19937& rng) const override {
        Individual c1 = p1.clone(), c2 = p2.clone();
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        auto clamp = [min,max](double v){ return std::max(min, std::min(max, v)); };

        if (u01(rng) <= prob) {
            for (size_t i = 0; i < p1.genes.size(); ++i) {
                double r = u01(rng);
                double beta = (r <= 0.5)
                    ? std::pow(2.0 * r, 1.0/(eta+1.0))
                    : std::pow(1.0/(2.0*(1.0-r)), 1.0/(eta+1.0));
                double x1 = p1.genes[i], x2 = p2.genes[i];
                c1.genes[i] = clamp(0.5*((1+beta)*x1 + (1-beta)*x2));
                c2.genes[i] = clamp(0.5*((1-beta)*x1 + (1+beta)*x2));
            }
        }
        return {std::move(c1), std::move(c2)};
    }
};

class PolynomialMutation final : public MutationStrategy {
    const double eta;
public:
    explicit PolynomialMutation(double di = 25.0) : eta(di) {}
    void mutate(Individual& ind, double rate,
                double min, double max, std::mt19937& rng) const override {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        double range = max - min;
        auto clamp = [min,max](double v){ return std::max(min, std::min(max, v)); };

        for (double& g : ind.genes) {
            if (u01(rng) > rate) continue;
            double r = u01(rng);
            double delta = (r <= 0.5)
                ? (std::pow(2.0*r, 1.0/(eta+1.0)) - 1.0)
                : (1.0 - std::pow(2.0*(1.0-r), 1.0/(eta+1.0)));
            g = clamp(g + delta * range);
        }
    }
};

// ============================================================
// 4. GeneticAlgorithm – CRASH-FREE
// ============================================================

class GeneticAlgorithm {
public:
    enum class Goal { Maximize, Minimize };

private:
    const size_t pop_size, num_genes, elite_count;
    const int    max_generations;
    const double mutation_rate, gene_min, gene_max;
    const Goal   goal;

    std::unique_ptr<SelectionStrategy>  selection;
    std::unique_ptr<CrossoverStrategy>  crossover;
    std::unique_ptr<MutationStrategy>   mutation;

    std::vector<Individual> population;
    std::mt19937 rng;

    // Statistics (filled **after** evaluation)
    double current_avg      = 0.0;
    double current_diversity = 0.0;

public:
    GeneticAlgorithm(
        size_t ps, size_t ng, int gens, double mr,
        double gmin, double gmax, size_t elites,
        std::unique_ptr<SelectionStrategy>  sel,
        std::unique_ptr<CrossoverStrategy>  cross,
        std::unique_ptr<MutationStrategy>   mut,
        Goal g = Goal::Maximize,
        std::optional<unsigned int> seed = std::nullopt)
        : pop_size(ps), num_genes(ng), elite_count(elites),
          max_generations(gens), mutation_rate(mr),
          gene_min(gmin), gene_max(gmax), goal(g),
          selection(std::move(sel)), crossover(std::move(cross)), mutation(std::move(mut)),
          rng(seed ? std::mt19937(*seed) : std::mt19937(std::random_device{}()))
    {
        if (elite_count > pop_size) throw std::invalid_argument("elite > pop");
        std::uniform_real_distribution<double> dist(gene_min, gene_max);
        population.reserve(pop_size);
        for (size_t i = 0; i < pop_size; ++i) {
            Individual ind(num_genes);
            for (double& g : ind.genes) g = dist(rng);
            population.emplace_back(std::move(ind));
        }
    }

    void run(const std::function<double(const std::vector<double>&)>& fitness) {
        // Initial evaluation
        for (auto& ind : population) ind.evaluate(fitness);
        update_statistics();

        for (int gen = 0; gen < max_generations; ++gen) {
            sort_population();
            print_stats(gen);

            std::vector<Individual> offspring;
            offspring.reserve(pop_size);

            // Elitism
            for (size_t i = 0; i < elite_count; ++i)
                offspring.push_back(population[i].clone());

            // Create new individuals
            while (offspring.size() < pop_size) {
                Individual p1 = selection->select(population, rng);
                Individual p2 = selection->select(population, rng);

                auto [c1, c2] = crossover->crossover(p1, p2, gene_min, gene_max, rng);
                mutation->mutate(c1, mutation_rate, gene_min, gene_max, rng);
                mutation->mutate(c2, mutation_rate, gene_min, gene_max, rng);

                c1.evaluate(fitness);
                c2.evaluate(fitness);

                offspring.push_back(std::move(c1));
                if (offspring.size() < pop_size)
                    offspring.push_back(std::move(c2));
            }

            population = std::move(offspring);
            update_statistics();
        }

        sort_population();
        print_stats(max_generations);
        std::cout << "\n=== Finished ===\n";
    }

private:
    void sort_population() {
        std::sort(population.begin(), population.end(),
                  [this](const Individual& a, const Individual& b) {
                      return (goal == Goal::Maximize) ? a.fitness > b.fitness
                                                      : a.fitness < b.fitness;
                  });
    }

    void update_statistics() {
        double sum = 0.0;
        for (const auto& ind : population) sum += ind.fitness;
        current_avg = sum / pop_size;

        std::vector<double> means(num_genes, 0.0);
        for (const auto& ind : population)
            for (size_t i = 0; i < num_genes; ++i)
                means[i] += ind.genes[i];
        for (double& m : means) m /= pop_size;

        double var = 0.0;
        for (const auto& ind : population)
            for (size_t i = 0; i < num_genes; ++i) {
                double d = ind.genes[i] - means[i];
                var += d * d;
            }
        current_diversity = std::sqrt(var / (pop_size * num_genes));
    }

    void print_stats(int gen) const {
        const auto& best = population[0];
        std::cout << "Gen " << std::setw(4) << gen
                  << " | Best: " << std::fixed << std::setprecision(8) << best.fitness
                  << " | Avg: "   << std::setprecision(6) << current_avg
                  << " | Div: "   << current_diversity
                  << "\n";
    }
};

// ============================================================
// 5. Main – Now crash-free!
// ============================================================

int main() {
    try {
        auto fitness = [](const std::vector<double>& g) {
            double s = 0.0;
            for (double x : g) s += x * x;
            return s;
        };

        GeneticAlgorithm ga(
            80, 10, 150,
            0.15, -10.0, 10.0, 4,
            std::make_unique<TournamentSelection>(7),
            std::make_unique<SBXCrossover>(20.0, 0.9),
            std::make_unique<PolynomialMutation>(25.0),
            GeneticAlgorithm::Goal::Maximize,
            12345u
        );

        std::cout << "=== Enhanced Real-Coded GA (Crash-Free) ===\n\n";
        ga.run(fitness);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
