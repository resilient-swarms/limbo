
#ifndef LIMBO_STAT_ASYNC_STAT_BASE_HPP
#define LIMBO_STAT_ASYNC_STAT_BASE_HPP

#include <fstream>
#include <string>
#include <map>
#include <memory>

namespace limbo {
    namespace stat {
        /**
          Base class for statistics

          The only method provided is protected :

          \rst
          .. code-block:: cpp

            template <typename BO>
            void _create_log_file(const BO& bo, const std::string& name)


          This method allocates an attribute `_log_file` (type: `std::shared_ptr<std::ofstream>`) if it has not been created yet, and does nothing otherwise. This method is designed so that you can safely call it in operator() while being 'guaranteed' that the file exists. Using this method is not mandatory for a statistics class.
          \endrst
        */
        template <typename Params>
        struct AsyncStatBase {
            
            AsyncStatBase() {

            }

            /// main method (to be written in derived classes)
            template <typename BO>
            void operator()(const BO& bo)
            {
                assert(false);
            }

        protected:
            std::map<size_t,std::shared_ptr<std::ofstream>> _log_file;

            template <typename BO>
            void _create_log_file(const BO& bo, const std::string& name, size_t index)
            {
                auto found = _log_file.find(index); 
                if ( found == _log_file.end() && bo.stats_enabled()) {
                    std::string log = bo.res_dir() + "/" + name;
                    _log_file[index]=std::make_shared<std::ofstream>(log.c_str());
                    assert(_log_file[index]->good());
                }
            }

            
        };
    }
}

#endif
