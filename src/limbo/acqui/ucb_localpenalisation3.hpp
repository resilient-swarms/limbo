//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
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
