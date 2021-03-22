
#ifndef LIMBO_STAT_ASYNCSTAT_BEST_HPP
#define LIMBO_STAT_ASYNCSTATS_BEST_HPP

#include <limbo/stat/async_statbase.hpp>

namespace limbo
{
    namespace stat
    {
        ///@ingroup stat
        ///filename: `samples.dat`
        template <typename Params>
        struct AsyncStatsBest : public AsyncStatBase<Params>
        {
            bool first_call = false;
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO &bo, const AggregatorFunction &afun)
            {
                if (!bo.stats_enabled() || bo.samples().empty())
                    return;
                size_t worker_index = bo.current_worker_index();
                size_t stat_index = bo.current_stat_index();
                this->_create_log_file(bo, "async_stats_best" + std::to_string(stat_index) + ".dat", stat_index);

                (*this->_log_file[stat_index]) << bo.times().back() << " " << bo.best_sample(worker_index, afun).transpose() << " " << bo.best_observation(worker_index, afun).transpose() << std::endl;
            }
        };
    } // namespace stat
} // namespace limbo

#endif
