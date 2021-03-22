
#ifndef LIMBO_STAT_ASYNCSTATS_HPP
#define LIMBO_STAT_ASYNCSTATS_HPP

#include <limbo/stat/async_statbase.hpp>

namespace limbo {
    namespace stat {
        ///@ingroup stat
        ///filename: `samples.dat`
        template <typename Params>
        struct AsyncStats : public AsyncStatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun)
            {
                if (!bo.stats_enabled() || bo.samples().empty())
                    return;
                size_t worker_index = bo.current_worker_index();
                size_t stat_index = bo.current_stat_index();
                std::cout << "stat worker index " << worker_index << std::endl;
                this->_create_log_file(bo, "async_stats"+std::to_string(stat_index)+".dat",stat_index);

                (*this->_log_file[stat_index]) << bo.times().back() << " " << bo.samples().back().transpose() << " " << bo.observations().back().transpose() << std::endl;
            }
        };
    }
}

#endif
