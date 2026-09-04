#include "spark_transport/tp4_fused_prefill_c_api.h"
#include "spark_transport/tp4_fused_prefill_session.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace {
void error_copy(const char* value, char* error, std::size_t bytes) {
  if (!error || !bytes) return;
  const auto length = std::min(std::strlen(value), bytes - 1U);
  std::memcpy(error, value, length); error[length] = 0;
}
spark_transport::Tp4BidirectionalPrefillOptions translate(
    const spark_tp4_bidirectional_prefill_config_v1& c) {
  if (c.struct_size < sizeof(c) || c.primary.struct_size < sizeof(c.primary) ||
      c.primary.base.rank >= 4 || c.timeout_seconds == 0 ||
      c.rail_count != 2 || c.query_rows != 8192 ||
      c.primary.elements_per_row != 4096 || c.primary.bytes_per_row != 8192 ||
      c.primary.base.payload_bytes != 8192ULL * 4096ULL * 2ULL ||
      !c.primary.base.peer0 || !c.primary.base.peer1 ||
      !c.primary.base.device0 || !c.primary.base.device1 ||
      !c.secondary_peer0 || !c.secondary_peer1 ||
      !c.secondary_device0 || !c.secondary_device1 ||
      c.primary.base.graph_submit_cpu_plus_one != 0 ||
      c.primary.base.graph_progress_cpu_plus_one != 0 ||
      c.primary.base.control_port0 == 0 || c.primary.base.control_port1 == 0 ||
      c.secondary_control_port0 == 0 || c.secondary_control_port1 == 0 ||
      c.primary.base.control_port0 == c.primary.base.control_port1 ||
      c.primary.base.control_port0 == c.secondary_control_port0 ||
      c.primary.base.control_port0 == c.secondary_control_port1 ||
      c.primary.base.control_port1 == c.secondary_control_port0 ||
      c.primary.base.control_port1 == c.secondary_control_port1 ||
      c.secondary_control_port0 == c.secondary_control_port1 ||
      std::strcmp(c.primary.base.peer0, c.secondary_peer0) == 0 ||
      std::strcmp(c.primary.base.peer1, c.secondary_peer1) == 0 ||
      std::strcmp(c.primary.base.peer0, c.primary.base.peer1) == 0 ||
      std::strcmp(c.primary.base.peer0, c.secondary_peer1) == 0 ||
      std::strcmp(c.primary.base.peer1, c.secondary_peer0) == 0 ||
      std::strcmp(c.secondary_peer0, c.secondary_peer1) == 0 ||
      std::strcmp(c.primary.base.device0, c.secondary_device0) == 0 ||
      std::strcmp(c.primary.base.device1, c.secondary_device1) == 0 ||
      std::strcmp(c.primary.base.device0, c.primary.base.device1) == 0 ||
      std::strcmp(c.primary.base.device0, c.secondary_device1) == 0 ||
      std::strcmp(c.primary.base.device1, c.secondary_device0) == 0 ||
      std::strcmp(c.secondary_device0, c.secondary_device1) == 0)
    throw std::invalid_argument("fused prefill requires exact Q8192 rail2 config");
  spark_transport::Tp4BidirectionalPrefillOptions o;
  o.rank=c.primary.base.rank; o.peer0=c.primary.base.peer0; o.peer1=c.primary.base.peer1;
  o.device0=c.primary.base.device0; o.device1=c.primary.base.device1;
  o.gid0=c.primary.base.gid0; o.gid1=c.primary.base.gid1;
  o.control_port0=c.primary.base.control_port0; o.control_port1=c.primary.base.control_port1;
  o.rail_count=2; o.query_rows=8192; o.elements_per_row=4096;
  o.secondary_peer0=c.secondary_peer0; o.secondary_peer1=c.secondary_peer1;
  o.secondary_device0=c.secondary_device0; o.secondary_device1=c.secondary_device1;
  o.secondary_gid0=c.secondary_gid0; o.secondary_gid1=c.secondary_gid1;
  o.secondary_control_port0=c.secondary_control_port0;
  o.secondary_control_port1=c.secondary_control_port1;
  o.timeout_seconds=c.timeout_seconds;
  return o;
}
struct Handle { explicit Handle(spark_transport::Tp4BidirectionalPrefillOptions o):session(std::move(o)){} spark_transport::Tp4FusedPrefillSession session; };
}
extern "C" spark_tp4_fused_prefill_handle spark_tp4_fused_prefill_create(
    const spark_tp4_bidirectional_prefill_config_v1* config, char* error,
    std::size_t bytes) {
  try { if(!config) throw std::invalid_argument("fused config is null"); auto h=std::make_unique<Handle>(translate(*config)); return h.release(); }
  catch(const std::exception& e){error_copy(e.what(),error,bytes);return nullptr;}
}
extern "C" int spark_tp4_fused_prefill_all_reduce(
    spark_tp4_fused_prefill_handle handle,const void* input,void* output,
    void* stream,char* error,std::size_t bytes){
  try { if(!handle) throw std::invalid_argument("fused handle is null"); static_cast<Handle*>(handle)->session.all_reduce_fused(input,output,stream,8192); return 0; }
  catch(const std::exception& e){error_copy(e.what(),error,bytes);return -1;}
}
extern "C" int spark_tp4_fused_prefill_all_reduce_rows(
    spark_tp4_fused_prefill_handle handle,const void* input,void* output,
    std::uint32_t query_rows,void* stream,char* error,std::size_t bytes){
  try { if(!handle) throw std::invalid_argument("fused handle is null"); static_cast<Handle*>(handle)->session.all_reduce_fused(input,output,stream,query_rows); return 0; }
  catch(const std::exception& e){error_copy(e.what(),error,bytes);return -1;}
}
extern "C" int spark_tp4_fused_prefill_get_health_status(
    spark_tp4_fused_prefill_handle handle, spark_tp4_health_status* status,
    std::size_t status_bytes, char* error, std::size_t error_bytes) {
  try {
    if (!handle) throw std::invalid_argument("fused handle is null");
    if (!status || status_bytes < sizeof(*status))
      throw std::invalid_argument("fused health status buffer is invalid");
    const auto snapshot = static_cast<Handle*>(handle)->session.health_status();
    spark_tp4_health_status result{};
    result.struct_size = sizeof(result);
    if (snapshot.healthy) result.flags |= SPARK_TP4_HEALTHY;
    if (snapshot.poisoned) result.flags |= SPARK_TP4_HEALTH_POISONED;
    if (snapshot.proxy_thread_running)
      result.flags |= SPARK_TP4_HEALTH_PROGRESS_THREAD_RUNNING;
    result.submitted_sequence = snapshot.submitted_sequence;
    result.completed_sequence = snapshot.completed_sequence;
    result.failing_sequence = snapshot.failing_sequence;
    result.error_code = snapshot.error_code;
    result.failing_stage = snapshot.failing_stage;
    result.failing_rail = -1;
    result.failing_peer = -1;
    *status = result;
    return 0;
  } catch (const std::exception& e) {
    error_copy(e.what(), error, error_bytes);
    return -1;
  }
}
extern "C" void spark_tp4_fused_prefill_destroy(spark_tp4_fused_prefill_handle h){delete static_cast<Handle*>(h);}
