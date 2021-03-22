
#ifndef LIMBO_ACQUI_UCB_LOCALPEN3_HPP
#define LIMBO_ACQUI_UCB_LOCALPEN3_HPP

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

        Alvi, A. S., Ru, B., Callies, J., Roberts, S. J., & Osborne, M. A. (2019). 
        Asynchronous batch Bayesian optimisation with improved local penalisation. 
        36th International Conference on Machine Learning, ICML 2019, 
        2019-June, 373–387.


        Contrary to version 2, this version 3 uses local lipschitz constants

        Parameters:
          - ``double alpha``
        \endrst
        */
        template <typename Params, typename Model>
        class UCB_LocalPenalisation3 : public UCB_LocalPenalisation<Params,Model>
        {
        public:
            size_t num_penalisations;
            const double gamma = 1.5;
            UCB_LocalPenalisation3(const Model &model, int iteration = 0) : UCB_LocalPenalisation<Params,Model>(model,iteration) {Params::LOCAL_L=true;};

            virtual double local_penalisation(const Eigen::VectorXd &x) const
            {
                if (Params::busy_samples.empty())
                {
                    return 1.0;
                }
                double penalty = 1.0;
                Eigen::VectorXd mu = Eigen::VectorXd(1);
                double sigma;
                //std::ofstream log("ppenalisation_log2.txt",std::ios::app);
                // std::cout << "checking penalty for sample " << x.transpose() << std::endl;
                // std::cout << "local penalty:" << std::endl;
                for (size_t i = 0; i < Params::busy_samples.size(); ++i)
                {
                    Eigen::VectorXd x0 = Params::busy_samples[i];

                    double E_r = this->_hammer_function_precompute_r(x0, Params::M,Params::LL[i]);
                    double s_x0 = this->_hammer_function_precompute_s(x0,Params::LL[i]);
                    double norm = (x - x0).norm();
                    double max_r = E_r + gamma * s_x0;
                    double phi = std::min(norm/max_r,1.0);
                    //log << x.transpose() << " " << x0.transpose() << " " << E_r << " " << s_x0 << " " << max_r << " " << phi << std::endl;
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
            
        };
    } // namespace acqui
} // namespace limbo

#endif
