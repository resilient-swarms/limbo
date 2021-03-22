
#ifndef LIMBO_ACQUI_UCB_LOCALPEN_HPP
#define LIMBO_ACQUI_UCB_LOCALPEN_HPP

#include <Eigen/Core>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <src/ite/global.hpp>

namespace limbo
{
    namespace acqui
    {
        /** @ingroup acqui
        \rst

        UCB with local penalisation according to 
        González, J., Dai, Z., Hennig, P., & Lawrence, N. (2016). 
        Batch bayesian optimization via local penalization. 
        Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, 
        AISTATS 2016, 648–657.
        Implemented following:
        https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/acquisitions/LP.py

        Parameters:
          - ``double alpha``
        \endrst
        */
        template <typename Params, typename Model>
        class UCB_LocalPenalisation
        {
        public:
            double _L;// lipschitz constant
            size_t num_penalisations;
            UCB_LocalPenalisation(const Model &model, int iteration = 0) : _model(model) { std::cout << "construct UCB with Local Penalisation" << std::endl; }

            size_t dim_in() const { return _model.dim_in(); }

            size_t dim_out() const { return _model.dim_out(); }

            double get_performance(const Eigen::VectorXd &v) const
            {
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = _model.query(v);
                return mu[0];
            }
            double get_M() const
            {
                return get_max_observation();
            }

            double _hammer_function_precompute_r(const Eigen::VectorXd &x0, double Max, double L) const
            {
                Eigen::VectorXd m;
                double sigma;
                std::tie(m, sigma) = _model.query(x0);
                double r_x0 = std::max(1e-4, (Max - m[0]) / L);
                // std::cout <<"r = " << r_x0*_L<< std::endl;
                // std::cout <<"r/L= " << r_x0 << std::endl;
                return r_x0;
            }

            double _hammer_function_precompute_s(const Eigen::VectorXd &x0, double L) const
            {
                Eigen::VectorXd m;
                double s;
                std::tie(m, s) = _model.query(x0);
                s = std::max(s, 1e-4); //avoid numerical issues with 0
                s = std::sqrt(s) / L;
                // std::cout <<"s = " << s*_L<< std::endl;
                // std::cout <<"s/L= " << s << std::endl;
                return s;
            }

            double _hammer_function(const Eigen::VectorXd &x, const Eigen::VectorXd &x0, double r_x0, double s_x0) const
            {
                //Creates the function to define the exclusion zones

                double dist = (x - x0).norm();
                double z = (dist - r_x0) / (std::sqrt(2) * s_x0);
                // std::cout << "dist= " << dist << std::endl;
                // std::cout <<"z = " << z << std::endl;
                // logcdf = log(1/sqrt(2*pi) * integral(exp(-t**2 / 2), t=-inf..x))  --> log(std::erf)
                //return norm.logcdf((np.sqrt((np.square(np.atleast_2d(x) [:, None, :] - np.atleast_2d(x0)[None, :, :])).sum(-1)) - r_x0) / s_x0);
                return 0.5 * std::erfc(-z);
            }

            double finite_diff_grad(const Eigen::VectorXd &v) const
            {
                double original_performance = get_performance(v);
                std::vector<Eigen::VectorXd> neighbours = Params::get_closest_neighbours(v);
                if (neighbours.size() < BEHAV_DIM)
                {
                    return -INFINITY; // we do not consider this one; since it is just to compute the maximal gradient norm across the archive
                }
                Eigen::VectorXd grad = Eigen::VectorXd::Zero(BEHAV_DIM);
                
                for (size_t i = 0; i < neighbours.size(); ++i)
                {
                    double performance_diff = get_performance(neighbours[i]) - original_performance;
                    // std::cout << "performance diff "<< i <<": "<< performance_diff<<std::endl;
                    // std::cout << "neighbours "<< i <<": "<<neighbours[i].transpose()<<std::endl;
                    // std::cout << "v : "<< v << std::endl;
                    Eigen::VectorXd x_diff = neighbours[i] - v;
                    for (size_t j = 0; j < BEHAV_DIM; ++j)
                    {

                        if (x_diff[j] != 0)
                        {
                            grad(j) = grad(j) + performance_diff / x_diff[j];
                        }
                    }
                }
                std::cout << "grad "<< grad.transpose() << std::endl;
#ifdef HETEROGENEOUS
                float ratio = std::sqrt(global::behav_dim + global::num_ID_features) / std::sqrt(global::behav_dim); //account for the ignored dimensions, if any
#else
                float ratio = 1.0;
#endif

                return ratio * grad.norm();
            }
            double get_gradnorm(const Eigen::VectorXd &v) const
            {
                if (_model.observations().empty())
                {
                    return 0.0;
                }
                // // get the D
                // Eigen::MatrixXd dkn = dKn(v);                       // Dxt
                // Eigen::MatrixXd Kn = _model.get_inv_kernel();       // t x t
                // Eigen::VectorXd obs = _model.observations_matrix(); //tx1
                // Eigen::VectorXd grad = dkn * Kn * obs;

                double gradnorm = finite_diff_grad(v);
                //std::cout << "gradnorm" << gradnorm << std::endl;
                // double gradnorm = grad.norm();
                // std::cout << "dKn " << dkn << std::endl;
                // std::cout << "Kn " << Kn << std::endl;
                // std::cout << "obs-matrix " << obs.transpose() << std::endl;
                // std::cout << "grad " << grad.transpose() << std::endl;
                // std::cout << "gradnorm " << gradnorm << std::endl;

                // Eigen::JacobiSVD<Eigen::MatrixXd> svd(Kn);
                // double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
                // std::cout << "condition number " << cond;
                // now check finite difference method

                // Eigen::VectorXd grad_fd  = Eigen::VectorXd(dim_in);
                // double h = 1e-8;
                // double performance = get_performance(v);
                // for (size_t i=0; i < dim_in(); ++i)
                // {
                //     Eigen::VectorXd v_h = v;
                //     v_h(i) = h;
                //     grad_fd(i) = (get_performance(v_h) - performance)/h;
                // }

                // double gradnorm_finite_diff = grad_fd.norm();
                // std::cout << "gradnorm  "<< gradnorm << std::endl;
                // std::cout << "gradnorm fd "<< gradnorm_finite_diff << std::endl;
                return gradnorm;
            }
            Eigen::VectorXd gradient_of_twonorm(const Eigen::VectorXd &s, const Eigen::VectorXd &w) const
            {
                // https://math.stackexchange.com/questions/291318/derivative-of-the-2-norm-of-a-multivariate-function
                // sum(f_i(X) \grad f_i(X)) / norm(f_i(X)), here f_i(X) = (Xc_i - X_i) --> \grad f_i(X) = -1 for component that changed. 0 for component that did not
                // but we want inidividual components instead
                size_t inputs = dim_in();
                Eigen::VectorXd grad = Eigen::VectorXd(inputs);
                double norm = (s - w).norm();
                for (size_t i = 0; i < inputs; ++i)
                {
                    grad(i) = (s[i] - w[i]) * (-1) / norm;
                }
                return grad;
            }
            Eigen::VectorXd gradient_of_squared_twonorm(const Eigen::VectorXd &s, const Eigen::VectorXd &w) const
            {
                // sum(2f_i(X)\grad f_i(X))
                // but we want inidividual components instead and \grad = -1 for component that change, 0 otherwise
                size_t inputs = dim_in();
                Eigen::VectorXd grad = Eigen::VectorXd(inputs);
                for (size_t i = 0; i < inputs; ++i)
                {
                    grad(i) = 2 * (s[i] - w[i]) * (-1);
                }
                return grad;
            }
            Eigen::VectorXd gradient_matern52(const Eigen::VectorXd &s, const Eigen::VectorXd &w) const
            {
                // sum(2f_i(X)\grad f_i(X))
                // but we want inidividual components instead and \grad = -1 for component that change, 0 otherwise
                //  _sf2 * (1 + term1 + term2) * std::exp(-term1); where term1 = sqrt(5)d/l , term2 = 5d^2/(3l^2)
                // -->
                size_t inputs = dim_in();
                Eigen::VectorXd grad = Eigen::VectorXd(inputs);
                double sf = _model.kernel_function().sf();
                double l = _model.kernel_function().l();
                Eigen::VectorXd grad_norm = std::sqrt(5) * gradient_of_twonorm(s, w) / l;
                Eigen::VectorXd grad_sq_norm = 5 * gradient_of_squared_twonorm(s, w) / (3 * l * l);
                double d = (s - w).norm();
                double term1 = std::sqrt(5) * d / l;
                double term2 = 5 * (d * d) / (3 * l * l);
                double r = std::exp(-term1);
                for (size_t i = 0; i < inputs; ++i)
                {
                    double left = sf * (grad_norm(i) + grad_sq_norm(i)) * (r); // d x exp(-x) = dx exp(-x) + x*dexp(-x)
                    double right = sf * (1 + term1 + term2) * (-grad_norm(i) * std::exp(-term1));
                    grad(i) = left + right;
                }
                return grad;
            }
            Eigen::MatrixXd dKn(const Eigen::VectorXd &v) const
            {
                size_t inputs = dim_in();
                std::vector<Eigen::VectorXd> samples = _model.samples();
                size_t t = samples.size();
                Eigen::MatrixXd g = Eigen::MatrixXd(inputs, t);

                for (size_t i = 0; i < t; ++i)
                {
                    Eigen::VectorXd grad = gradient_matern52(samples[i],v);
                    for (size_t j = 0; j < inputs; ++j)
                    {
                        g(j, i) = grad[j];
                    }
                }
                return g;
            }
            double get_max_observation() const
            {
                std::vector<Eigen::VectorXd> observations = _model.observations();
                double max = 0.0;
                for (size_t i = 0; i < observations.size(); ++i)
                {
                    if (observations[i][0] > max)
                    {
                        max = observations[i][0];
                    }
                }
                return max;
            }
            opt::eval_t operator()(const Eigen::VectorXd &v, bool gradient) const
            {
                //std::cout << "perform UCB" << std::endl;
                assert(!gradient);
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = _model.query(v);
                Params::M = get_max_observation();
                double penalty = this->local_penalisation(v);
                double UCB = mu[0] + Params::acqui_ucb::alpha() * sqrt(sigma);
                if (UCB < 0.0)
                {
                    if (penalty == 0)
                    {
                        return opt::no_grad(-1000000.0 + UCB);
                    }
                    return opt::no_grad(std::max(-1000000.0 + UCB, UCB / penalty));
                }
                else
                {
                    return opt::no_grad(UCB * penalty);
                }
            }

            virtual double local_penalisation(const Eigen::VectorXd &x) const
            {
                if (Params::busy_samples.empty())
                {
                    return 1.0;
                }
                double penalty = 1.0;
                Eigen::VectorXd mu = Eigen::VectorXd(1);
                double sigma;
                //std::ofstream log("penalisation_log2.txt",std::ios::app);
                // std::cout << "checking penalty for sample " << x.transpose() << std::endl;
                // std::cout << "local penalty:" << std::endl;
                for (size_t i = 0; i < Params::busy_samples.size(); ++i)
                {
                    Eigen::VectorXd x0 = Params::busy_samples[i];

                    double r_x0 = _hammer_function_precompute_r(x0, Params::M,Params::L);
                    double s_x0 = _hammer_function_precompute_s(x0,Params::L);
                    double phi = _hammer_function(x, x0, r_x0, s_x0);
                    //log << x.transpose() << " " << x0.transpose() << " " << r_x0 << " " << s_x0 << " " << phi << std::endl;
                    if (phi < .001)
                    {
                        ++Params::count;
                        
                        std::cout << "number of strong penalisations " << Params::count << std::endl;
                        std::cout <<"busy sample 1: "<< x0.transpose() << std::endl;
                        std::cout <<"queried sample 1: "<< x.transpose() << std::endl;
                    }
                    penalty *= phi;
                }
                return penalty;
            }
        
        protected:
            const Model &_model;
            
        };
    } // namespace acqui
} // namespace limbo


#endif
