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
#ifndef LIMBO_KERNEL_KERNEL_VARIABLENOISE_HPP
#define LIMBO_KERNEL_KERNEL_VARIABLENOISE_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo
{
    namespace kernel
    {
        /**
          @ingroup kernel
          \rst
          Base struct for kernel definition. It handles the noise and its optimization (only if the kernel allows hyper-parameters optimization).
          \endrst

          Parameters:
             - ``double noise`` (initial signal noise squared)
             - ``bool optimize_noise`` (whether we are optimizing for the noise or not)
        */
        template <typename Params, typename Kernel>
        struct BaseKernelVariableNoise
        {
        public:
            BaseKernelVariableNoise(size_t dim = 1)
            {

            }

            double operator()(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, int i = -1, int j = -2) const
            {
                return static_cast<const Kernel *>(this)->kernel(v1, v2) + ((i == j) ? _noise[i] + 1e-8 : 0.0);
            }

            // Get signal noise
            double noise() const { return 0.0; } // don't add any additional noise in addition to the above

            void set_noise(std::vector<double> value) { _noise = value; }

        protected:
            std::vector<double> _noise;
            double _noise_p;

            // Functions for compilation issues
            // They should never be called like this
            size_t params_size() const { return 0; }

            Eigen::VectorXd params() const { return Eigen::VectorXd(); }

            void set_params(const Eigen::VectorXd &p) {}

            Eigen::VectorXd gradient(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
            {
                // This should never be called!
                assert(false);
            }
        };
    } // namespace kernel
} // namespace limbo

#endif
