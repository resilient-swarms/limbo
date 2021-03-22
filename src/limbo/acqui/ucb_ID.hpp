
#ifndef LIMBO_ACQUI_UCB_ID_HPP
#define LIMBO_ACQUI_UCB_ID_HPP

#include <Eigen/Core>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace acqui {
        /** @ingroup acqui
        \rst
        Classic UCB (Upper Confidence Bound). See :cite:`brochu2010tutorial`, p. 14

          .. math::
            UCB(x) = \mu(x) + \alpha \sigma(x).

        Parameters:
          - ``double alpha``
        \endrst
        */
        template <typename Params, typename Model>
        class UCB_ID {
        public:
            UCB_ID(const Model& model, int iteration = 0) : _model(model) {std::cout << "construct UCB" << std::endl;}

            size_t dim_in() const { return _model.dim_in(); }

            size_t dim_out() const { return _model.dim_out(); }

            template <typename AggregatorFunction>
            opt::eval_t operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun, bool gradient) const
            {
                //std::cout << "perform UCB" << std::endl;
                assert(!gradient);
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = _model.query_ID(v);
                //std::cout << "alpha = " << Params::acqui_ucb::alpha() << std::endl;
                return opt::no_grad(afun(mu) + Params::acqui_ucb::alpha() * sqrt(sigma));
            }

        protected:
            const Model& _model;
        };
    }
}

#endif
