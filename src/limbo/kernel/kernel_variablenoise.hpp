
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
