
#ifndef LIMBO_BAYES_OPT_BOPTIMIZER_ASYNC_HPP
#define LIMBO_BAYES_OPT_BOPTIMIZER_ASYNC_HPP

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <boost/parameter/aux_/void.hpp>

#include <Eigen/Core>

#include <limbo/bayes_opt/bo_base.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>
#ifdef USE_NLOPT
#include <limbo/opt/nlopt_no_grad.hpp>
#elif defined USE_LIBCMAES
#include <limbo/opt/cmaes.hpp>
#else
#include <limbo/opt/grid_search.hpp>
#endif

#include <src/core/statistics.h>

namespace limbo
{
    namespace defaults
    {
        struct bayes_opt_boptimizer
        {
            BO_PARAM(int, hp_period, -1);
        };
    } // namespace defaults

    BOOST_PARAMETER_TEMPLATE_KEYWORD(acquiopt)

    namespace bayes_opt
    {

        using boptimizer_signature = boost::parameter::parameters<boost::parameter::optional<tag::acquiopt>,
                                                                  boost::parameter::optional<tag::statsfun>,
                                                                  boost::parameter::optional<tag::initfun>,
                                                                  boost::parameter::optional<tag::acquifun>,
                                                                  boost::parameter::optional<tag::stopcrit>,
                                                                  boost::parameter::optional<tag::modelfun>>;

        // clang-format off
        /**
        The classic Bayesian optimization algorithm.

        \rst
        References: :cite:`brochu2010tutorial,Mockus2013`
        \endrst

        This class takes the same template parameters as BoBase. It adds:
        \rst
        +---------------------+------------+----------+---------------+
        |type                 |typedef     | argument | default       |
        +=====================+============+==========+===============+
        |acqui. optimizer     |acquiopt_t  | acquiopt | see below     |
        +---------------------+------------+----------+---------------+
        \endrst

        The default value of acqui_opt_t is:
        - ``opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>`` if NLOpt was found in `waf configure`
        - ``opt::Cmaes<Params>`` if libcmaes was found but NLOpt was not found
        - ``opt::GridSearch<Params>`` otherwise (please do not use this: the algorithm will not work as expected!)
        */
        template <class Params,
          class A1 = boost::parameter::void_,
          class A2 = boost::parameter::void_,
          class A3 = boost::parameter::void_,
          class A4 = boost::parameter::void_,
          class A5 = boost::parameter::void_,
          class A6 = boost::parameter::void_>
        // clang-format on
        class BOptimizerAsync : public BoBase<Params, A1, A2, A3, A4, A5, A6>
        {
        public:
            // defaults
            struct defaults
            {
#ifdef USE_NLOPT
                using acquiopt_t = opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>;
#elif defined(USE_LIBCMAES)
                using acquiopt_t = opt::Cmaes<Params>;
#else
#warning NO NLOpt, and NO Libcmaes: the acquisition function will be optimized by a grid search algorithm (which is usually bad). Please install at least NLOpt or libcmaes to use limbo!.
                using acquiopt_t = opt::GridSearch<Params>;
#endif
            };
            /// link to the corresponding BoBase (useful for typedefs)
            using base_t = BoBase<Params, A1, A2, A3, A4, A5, A6>;
            using model_t = typename base_t::model_t;
            using acquisition_function_t = typename base_t::acquisition_function_t;
            // extract the types
            using args = typename boptimizer_signature::bind<A1, A2, A3, A4, A5, A6>::type;
            using acqui_optimizer_t = typename boost::parameter::binding<args, tag::acquiopt, typename defaults::acquiopt_t>::type;

            Eigen::VectorXd NULL_VEC;
            Eigen::VectorXd NULL_FLOAT;
            std::vector<Eigen::VectorXd> worker_samples;
            std::vector<std::vector<Eigen::VectorXd>> delayed_worker_samples;
            //bool alltrialsfinished = false;
            /// The initialisation step of the main function (run the Bayesian optimization algorithm)
            template <typename Controller, typename StateFunction, typename AggregatorFunction = FirstElem>
            void optimize_init(size_t num_ID_features, size_t num_robots, const StateFunction &sfun, const AggregatorFunction &afun = AggregatorFunction(), bool variable_noise = true)
            {
                NUM_ID_FEATURES = num_ID_features;
                _num_workers = num_robots;
                //this->_init(sfun, afun, reset);
                _variable_noise = variable_noise;
                if (!this->_observations.empty())
                    _model.compute(this->_samples, this->_observations);
                else
                    _model = model_t(StateFunction::dim_in(), StateFunction::dim_out());
                NULL_VEC = Eigen::VectorXd::Constant(BEHAV_DIM + NUM_ID_FEATURES, -1);
                NULL_FLOAT = Eigen::VectorXd::Constant(1, -1);
                for (int i = 0; i < _num_workers; ++i)
                {

                    worker_samples.push_back(NULL_VEC);
                    _worker_observations.push_back(NULL_FLOAT);
                    _worker_stats.push_back(RunningStat());
                    _individual_observations.push_back(std::vector<Eigen::VectorXd>());
                    _individual_samples.push_back(std::vector<Eigen::VectorXd>());
                    // delayed samples, observations, and stats
                    _delay_setting.push_back(false);
                    _pending.push_back({});
                    delayed_worker_samples.push_back(std::vector<Eigen::VectorXd>());
                    _delayed_worker_observations.push_back(std::vector<Eigen::VectorXd>());
                    _delayed_worker_stats.push_back(std::vector<RunningStat>());
                    _delayed_samples.push_back(std::vector<Eigen::VectorXd>());
                    _delayed_observations.push_back(std::vector<Eigen::VectorXd>());
                    _delayed_observation_noises.push_back(std::vector<double>());
                    for (int j = 0; j < _num_workers; ++j)
                    {
                        delayed_worker_samples[i].push_back(NULL_VEC);
                        _delayed_worker_observations[i].push_back(NULL_FLOAT);
                        _delayed_worker_stats[i].push_back(RunningStat());
                    }
                }
            }
            /// The initialisation step of the main function (run the Bayesian optimization algorithm)
            template <typename Controller, typename StateFunction, typename AggregatorFunction = FirstElem>
            void optimize_init_joint(size_t num_unique_faults, size_t num_ID_features, const StateFunction &sfun, const AggregatorFunction &afun = AggregatorFunction(), bool variable_noise = true)
            {
                NUM_ID_FEATURES = num_ID_features;
                _num_workers = 1;
                //this->_init(sfun, afun, reset);
                _variable_noise = variable_noise;
                if (!this->_observations.empty())
                    _model.compute(this->_samples, this->_observations);
                else
                    _model = model_t(num_unique_faults * StateFunction::dim_in(), StateFunction::dim_out());

                NULL_VEC = Eigen::VectorXd::Constant(num_unique_faults * (BEHAV_DIM + NUM_ID_FEATURES), -1);
                NULL_FLOAT = Eigen::VectorXd::Constant(1, -1);
                for (int i = 0; i < _num_workers; ++i)
                {
                    worker_samples.push_back(NULL_VEC);
                    _worker_observations.push_back(NULL_FLOAT);
                    _worker_stats.push_back(RunningStat());

                    _individual_observations.push_back(std::vector<Eigen::VectorXd>());
                    _individual_samples.push_back(std::vector<Eigen::VectorXd>());
                    // delayed samples, observations, and stats
                    _delay_setting.push_back(false);
                    _pending.push_back({});
                    delayed_worker_samples.push_back(std::vector<Eigen::VectorXd>());
                    _delayed_worker_observations.push_back(std::vector<Eigen::VectorXd>());
                    _delayed_worker_stats.push_back(std::vector<RunningStat>());
                    _delayed_samples.push_back(std::vector<Eigen::VectorXd>());
                    _delayed_observations.push_back(std::vector<Eigen::VectorXd>());
                    _delayed_observation_noises.push_back(std::vector<double>());
                    for (int j = 0; j < _num_workers; ++j)
                    {
                        delayed_worker_samples[i].push_back(NULL_VEC);
                        _delayed_worker_observations[i].push_back(NULL_FLOAT);
                        _delayed_worker_stats[i].push_back(RunningStat());
                    }
                }
            }

            void clear_fitness(size_t worker_index)
            {
                _worker_stats[worker_index] = RunningStat();
            }
            void push_fitness(size_t worker_index, double f)
            {
                int r = std::rand() % 100;
                if (r < Params::DELAY_PROB)
                {
                    _worker_stats[worker_index].delayed = true;
                }
                else
                {
                    _worker_stats[worker_index].delayed = false;
                }
                _worker_stats[worker_index].push(f);
            }
            double get_mean_fitness(size_t worker_index)
            {
                return _worker_stats[worker_index].mean();
            }
            void push_time(double time, bool alltrialsfinished)
            {
                if (alltrialsfinished)
                {
                    _times.push_back(time);
                }
            }
            size_t current_worker_index() const
            {
                return _current_index;
            }
            size_t current_stat_index() const
            {
                return _stat_index;
            }
            /// A single step of the main function (run the Bayesian optimization algorithm)
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void update_model(const Eigen::VectorXd &x, const size_t worker_index, size_t stat_index, const StateFunction &sfun, bool all_trials_finished = false, const AggregatorFunction &afun = AggregatorFunction())
            {
                std::cout << "update_model" << std::endl;
                Eigen::VectorXd f = Eigen::VectorXd::Constant(1, _worker_stats[worker_index].mean());
                this->add_new_sample(worker_index, x, f);

                //_model.add_sample(this->_samples.back(), this->_observations.back());
                //_model.compute_inv_kernel();
                if (Params::bayes_opt_boptimizer::hp_period() > 0 && (this->_current_iteration + 1) % Params::bayes_opt_boptimizer::hp_period() == 0)
                    _model.optimize_hyperparams();

                // synchronise the worker_samples, first the]
                sync_self(worker_index);
                bool delay = _worker_stats[worker_index].delayed;
                if (delay)
                {
                    _delay_setting[worker_index] = true;
                }
                else
                {
                    _delay_setting[worker_index] = false; //will not yet be added to busy samples
                    sync_workers(worker_index);
                    // send all messages from 
                    clear_pending(worker_index);
                }
                // select new controller if needed
                if (all_trials_finished)
                {
                    std::cout << "alltrialsfinished" << std::endl;
                    this->_current_iteration++;
                    this->_total_iterations++;
                    if (_variable_noise)
                    {
                        this->_observation_noises.push_back(_worker_stats[worker_index].standard_error());
                    }
                    else
                    {
                        this->_observation_noises.push_back(0.0f);
                    }
                    this->_observations.push_back(_worker_observations[worker_index]);
                    this->_samples.push_back(worker_samples[worker_index]);
                    this->_individual_observations[worker_index].push_back(_worker_observations[worker_index]);
                    this->_individual_samples[worker_index].push_back(worker_samples[worker_index]);
                    
                    for (int j = 0; j < _num_workers; ++j)
                    {

                        if (j == worker_index)
                        {
                            _delayed_observation_noises[j].push_back(this->_observation_noises.back());
                            _delayed_observations[j].push_back(this->_observations.back());
                            _delayed_samples[j].push_back(this->_samples.back());
                        }
                        else
                        {
                            if (delay)
                            {
                               _pending[j].push_back(this->_samples.size() - 1);
                            }
                            else
                            {
                                _delayed_observation_noises[j].push_back(this->_observation_noises.back());
                                _delayed_observations[j].push_back(this->_observations.back());
                                _delayed_samples[j].push_back(this->_samples.back());
                            }
                        }
                    }
                    _current_index = worker_index;
                    _stat_index = stat_index;
                    this->_update_stats(*this, afun);
                    worker_samples[worker_index] = NULL_VEC;
                    _worker_observations[worker_index] = NULL_FLOAT;
                    _worker_stats[worker_index] = RunningStat(); // reset to zero sample size
                    sync_self(worker_index);
                    if (!delay)
                    {
                        sync_workers(worker_index);
                    }
                }
            }
            void sync_self(int worker_index)
            {
                delayed_worker_samples[worker_index][worker_index] = worker_samples[worker_index]; // will be added to delayed busy samples
                _delayed_worker_stats[worker_index][worker_index] = _worker_stats[worker_index];
                _delayed_worker_observations[worker_index][worker_index] = _worker_observations[worker_index];
            }
            void sync_workers(int worker_index)
            {
                // this means all other workers will receive the current stats from the current worker
                for (int i = 0; i < _num_workers; ++i)
                {
                    if (i != worker_index)
                    {
                        delayed_worker_samples[i][worker_index] = worker_samples[worker_index]; // will be added to delayed busy samples
                        _delayed_worker_stats[i][worker_index] = _worker_stats[worker_index];
                        _delayed_worker_observations[i][worker_index] = _worker_observations[worker_index];
                        // and also any pending samples and observations from the current worker
                    }
                }
            }

            void clear_pending(int worker_index)
            {
                // all delayed samples, observations, and noises come in
                for (int i = 0; i < _num_workers; ++i)
                {
                    if(i == worker_index)
                    {
                        continue;
                    }
                    for (size_t idx : _pending[i])
                    {
                        _delayed_observation_noises[i].push_back(this->_observation_noises[idx]);
                        _delayed_observations[i].push_back(this->_observations[idx]);
                        _delayed_samples[i].push_back(this->_samples[idx]);
                    }
                     _pending[i].clear();
                }
                //finally, clear all pending messages from the worker_index
               
            }
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            Eigen::VectorXd select_sample(int worker_index, const Eigen::VectorXd &id_vec, const AggregatorFunction &afun = AggregatorFunction())
            {

                combine_observation_noise(worker_index);
                auto samples = combined_samples(worker_index);
                Params::samples = samples;
                auto observations = combined_observations(worker_index);
                if (!samples.empty())
                {
                    _model.compute(samples, observations); //non-incremental computation of model
                }
                acquisition_function_t acqui(_model, this->_current_iteration);
                Params::busy_samples.clear();

                for (size_t i = 0; i < _num_workers; ++i)
                {
                    if(i == worker_index)
                    {
                        continue; // do not add the worker's own sample as busy sample
                    }
                    Eigen::VectorXd samp;
                    if (worker_index == -1)
                    {
                        samp = worker_samples[i];
                    }
                    else
                    {
                        samp = delayed_worker_samples[worker_index][i];
                    }
                    if (samp != NULL_FLOAT)
                    {
                        Params::busy_samples.push_back(samp);
                    }
                }

                // std::ofstream log("samples_log.txt", std::ios::app);
                
                // std::string delay;
                // if (worker_index == -1)
                // {
                //     delay = "delay";
                // }
                // else{
                //     delay = _delay_setting[worker_index] ? "delay" : "no delay";
                // }
                // log << worker_index << ": " << delay << std::endl;
                // log << "true" << std::endl;
                // for (int i = 0; i < this->_samples.size(); ++i)
                // {
                //     log << this->_samples[i].transpose() << " " << this->_observations[i].transpose() << std::endl;
                // }
                // log << "busy" << std::endl;
                // for (int j = 0; j < Params::busy_samples.size(); ++j)
                // {
                //     log << " " << Params::busy_samples[j].transpose() << std::endl;
                // }
                // log << "robot" << std::endl;
                // for (int i = 0; i < samples.size(); ++i)
                // {
                //     log << samples[i].transpose() << " " << observations[i].transpose() << std::endl;
                // }
                // log << "_________________________________________________" << std::endl;
                // check if any
                Eigen::VectorXd starting_point = tools::random_vector(StateFunction::dim_in(), Params::bayes_opt_bobase::bounded());
#if BO_ACQUISITION >= 2
                _model.compute_inv_kernel();
                return acqui_optimizer(acqui, starting_point, Params::bayes_opt_bobase::bounded(), id_vec, afun);
#else
                auto acqui_optimization =
                    [&](const Eigen::VectorXd &x, bool g) { return acqui(x, afun, g); };
                return acqui_optimizer(acqui_optimization, starting_point, Params::bayes_opt_bobase::bounded(), id_vec);
#endif
            }

            // eval-and-add but allow replacing prior observations with more accurate ones
            /// Add a new sample / observation pair
            /// - does not update the model!
            /// - we don't add NaN and inf observations
            virtual void
            add_new_sample(size_t index, const Eigen::VectorXd &s, const Eigen::VectorXd &v)
            {
                if (tools::is_nan_or_inf(v))
                    throw EvaluationError();
                worker_samples[index] = s;
                _worker_observations[index] = v;
            }

            /// return the best observation so far (i.e. max(f(x)))
            template <typename AggregatorFunction = FirstElem>
            const Eigen::VectorXd &best_observation(size_t index, const AggregatorFunction &afun = AggregatorFunction()) const
            {
                auto rewards = std::vector<double>(this->_individual_observations[index].size());
                std::transform(this->_individual_observations[index].begin(), this->_individual_observations[index].end(), rewards.begin(), afun);
                auto max_e = std::max_element(rewards.begin(), rewards.end());
                return this->_individual_observations[index][std::distance(rewards.begin(), max_e)];
            }

            /// return the best sample so far (i.e. the argmax(f(x)))
            template <typename AggregatorFunction = FirstElem>
            const Eigen::VectorXd &best_sample(size_t index, const AggregatorFunction &afun = AggregatorFunction()) const
            {
                auto rewards = std::vector<double>(this->_individual_observations[index].size());
                std::transform(this->_individual_observations[index].begin(), this->_individual_observations[index].end(), rewards.begin(), afun);
                auto max_e = std::max_element(rewards.begin(), rewards.end());
                return this->_individual_samples[index][std::distance(rewards.begin(), max_e)];
            }

            const model_t &model() const { return _model; }

            std::vector<Eigen::VectorXd> combined_observations(int self_index)
            {
                std::vector<Eigen::VectorXd> comb;
                if (self_index == -1)
                {
                    comb = this->_observations;
                    for (int i = 0; i < _num_workers; ++i)
                    {
                        if (_worker_observations[i] != NULL_FLOAT)
                        {
                            comb.push_back(_worker_observations[i]);
                        }
                    }
                }
                else
                {
                    comb = _delayed_observations[self_index];
                    for (int i = 0; i < _num_workers; ++i)
                    {
                        if (i == self_index) // is already included in the above
                        {
                            continue;
                        }
                        if (_delayed_worker_observations[self_index][i] != NULL_FLOAT)
                        {
                            comb.push_back(_delayed_worker_observations[self_index][i]);
                        }
                    }
                }
                return comb;
            }
            /* call this BEFORE combined_observations() */
            void combine_observation_noise(int self_index)
            {
                std::vector<double> comb;
                if (self_index == -1)
                {
                    comb = _observation_noises;
                    for (int i = 0; i < _num_workers; ++i)
                    {
                        if (_worker_observations[i] != NULL_FLOAT)
                        {
                            if (_variable_noise)
                            {
                                comb.push_back(_worker_stats[i].standard_error());
                            }
                            else
                            {
                                comb.push_back(0.0f);
                            }
                        }
                    }
                }
                else
                {
                    comb = _delayed_observation_noises[self_index];
                    for (int i = 0; i < _num_workers; ++i)
                    {
                        if (i == self_index) // is already included in the above
                        {
                            continue;
                        }
                        if (_delayed_worker_observations[self_index][i] != NULL_FLOAT)
                        {
                            if (_variable_noise)
                            {
                                comb.push_back(_delayed_worker_stats[self_index][i].standard_error());
                            }
                            else
                            {
                                comb.push_back(0.0f);
                            }
                        }
                    }
                }
                _model.kernel_function().set_noise(comb);
            }

            std::vector<Eigen::VectorXd> combined_samples(int self_index)
            {
                std::vector<Eigen::VectorXd> comb;
                if (self_index == -1)
                {
                    comb = this->_samples;
                    for (int i = 0; i < _num_workers; ++i)
                    {
                        if (_worker_observations[i] != NULL_FLOAT)
                        {
                            comb.push_back(worker_samples[i]);
                        }
                    }
                }
                else
                {
                    comb = _delayed_samples[self_index];
                    std::cout <<"length of delayed_samples " << _delayed_samples[self_index].size() << std::endl;
                    for (int i = 0; i < _num_workers; ++i)
                    {
                        if (i == self_index) // is already included in the above
                        {
                            continue;
                        }
                        std::cout <<"length of delayed_worker_samples " << delayed_worker_samples.size() << std::endl;
                        //std::cout << "worker sample " << i << " :" << worker_samples[i] << std::endl;
                        if (_delayed_worker_observations[self_index][i] != NULL_FLOAT) // because worker_samples.size() may be greater
                        {
                            comb.push_back(delayed_worker_samples[self_index][i]);
                            
                            //std::cout << "comb " << comb.back() << std::endl;
                        }
                    }
                }
                return comb;
            }

            std::vector<double> times() const
            {
                return _times;
            }

            // double get_performance(const Eigen::VectorXd &v)
            // {
            //     Eigen::VectorXd mu;
            //     double sigma;
            //     std::tie(mu, sigma) = _model.query(v);
            //     return mu[0];
            // }

            // double get_gradnorm(const Eigen::VectorXd &v)
            // {
            //     // get the D
            //     Eigen::VectorXd dkn = dKn(v);                     // Dxt
            //     Eigen::MatrixXd Kn = _model.get_inverse_kernel(); // t x t
            //     Eigen::VectorXd obs = observation;                //tx1
            //     Eigen::VectorXd grad = dKn * Kn * obs;
            //     return grad.norm();
            // }
            // Eigen::VectorXd gradient_of_twonorm(const Eigen::VectorXd &v, const Eigen::VectorXd &w)
            // {
            //     // https://math.stackexchange.com/questions/291318/derivative-of-the-2-norm-of-a-multivariate-function
            //     // sum(f_i(X) \grad f_i(X)) / norm(f_i(X)), here f_i(X) = (Xc_i - X_i) --> \grad f_i(X) = -1 for component that changed. 0 for component that did not
            //     // but we want inidividual components instead
            //     size_t inputs = _model.dim_in();
            //     Eigen::VectorXd grad = Eigen::VectorXd(inputs);
            //     double norm = (v - w).norm();
            //     for (size_t i = 0; i < inputs; ++i)
            //     {
            //         grad(i) = (v[i] - w[i]) * (-1) / norm;
            //     }
            //     return grad;
            // }
            // Eigen::VectorXd gradient_of_squared_twonorm(const Eigen::VectorXd &v, const Eigen::VectorXd &w)
            // {
            //     // sum(2f_i(X)\grad f_i(X))
            //     // but we want inidividual components instead and \grad = -1 for component that change, 0 otherwise
            //     size_t inputs = _model.dim_in();
            //     Eigen::VectorXd grad = Eigen::VectorXd(inputs);
            //     for (size_t i = 0; i < inputs; ++i)
            //     {
            //         grad(i) = 2 * (v[i] - w[i]) * (-1);
            //     }
            //     return grad;
            // }
            // Eigen::VectorXd gradient_matern52(const Eigen::VectorXd &v, const Eigen::VectorXd &w)
            // {
            //     // sum(2f_i(X)\grad f_i(X))
            //     // but we want inidividual components instead and \grad = -1 for component that change, 0 otherwise
            //     //  _sf2 * (1 + term1 + term2) * std::exp(-term1); where term1 = sqrt(5)d/l , term2 = 5d^2/(3l^2)
            //     // -->
            //     size_t inputs = _model.dim_in();
            //     Eigen::VectorXd grad = Eigen::VectorXd(inputs);
            //     double s = _model.kernel_function().sf();
            //     double l = _model.kernel_function().l();
            //     Eigen::VectorXd grad_norm = std::sqrt(5) * gradient_of_twonorm(v, w) / l;
            //     Eigen::VectorXd grad_sq_norm = 5 * gradient_of_squared_twonorm(v, w) / (3 * l * l);
            //     double d = (v - w).norm();
            //     double term1 = std::sqrt(5) * d / l;
            //     double term2 = 5 * (d * d) / (3 * l * l);
            //     double r = std::exp(-term1);
            //     for (size_t i = 0; i < inputs; ++i)
            //     {
            //         double left = s * (grad_norm(i) + grad_sq_norm(i)) * (r); // d x exp(-x) = dx exp(-x) + x*dexp(-x)
            //         double right - s *(1 + term1 + term2) * (-grad_sq_norm(i) * std::exp(-term2));
            //         grad(i) = left + right;
            //     }
            //     return grad;
            // }
            // Eigen::MatrixXd dKn(const Eigen::VectorXd &v)
            // {
            //     size_t inputs = _model.dim_in();
            //     std::vector<VectorXd> obs = _model.observations();
            //     size_t t = obs.size();
            //     Eigen::MatrixXd g = Eigen::MatrixXd(inputs, t);

            //     for (size_t j = 0; j < t; ++i)
            //     {
            //         Eigen::VectorXd grad = gradient_matern52(v, obs[j]);
            //         for (size_t j = 0; j < inputs; ++j)
            //         {
            //             g(j, i) = grad[j];
            //         }
            //     }
            //     return g;
            // }
            // after choosing the new sample
            void set_worker_samples(size_t index, const Eigen::VectorXd &result)
            {
                worker_samples[index] = result;
                
                if (_delay_setting[index])
                {
                    // only update self
                    delayed_worker_samples[index][index] = result; 
                }
                else{  // update all
                    for (int i = 0; i < _num_workers; ++i)
                    {
                        delayed_worker_samples[i][index] = result;
                    }
                }
            }

        protected:
            model_t _model;
            int _num_workers = 6; // number of workers
            //size_t _num_trials = 8;   //number of trials that has to be completed for an observation
            size_t NUM_ID_FEATURES;
            size_t _current_index, _stat_index;
            std::vector<Eigen::VectorXd> _worker_observations;
            std::vector<std::vector<Eigen::VectorXd>> _delayed_worker_observations;
            std::vector<std::vector<Eigen::VectorXd>> _individual_observations, _individual_samples;
            std::vector<std::vector<Eigen::VectorXd>> _delayed_samples, _delayed_observations;
            std::vector<std::vector<double>> _delayed_observation_noises;
            std::vector<double> _observation_noises;
            std::vector<double> _times;
            std::vector<RunningStat> _worker_stats;
            std::vector<std::vector<RunningStat>> _delayed_worker_stats;
            acqui_optimizer_t acqui_optimizer;
            std::vector<bool> _delay_setting;
            std::vector<std::vector<size_t>> _pending;
            bool _variable_noise;
        }; // namespace bayes_opt

        namespace _default_hp
        {
            template <typename Params>
            using model_t = model::GPOpt<Params>;
            template <typename Params>
            using acqui_t = acqui::UCB<Params, model_t<Params>>;
        } // namespace _default_hp

        /// A shortcut for a BOptimizer with UCB + GPOpt
        /// The acquisition function and the model CANNOT be tuned (use BOptimizer for this)
        template <class Params,
                  class A1 = boost::parameter::void_,
                  class A2 = boost::parameter::void_,
                  class A3 = boost::parameter::void_,
                  class A4 = boost::parameter::void_>
        using BOptimizerHPOpt = BOptimizerAsync<Params, modelfun<_default_hp::model_t<Params>>, acquifun<_default_hp::acqui_t<Params>>, A1, A2, A3, A4>;
    } // namespace bayes_opt
} // namespace limbo
#endif
