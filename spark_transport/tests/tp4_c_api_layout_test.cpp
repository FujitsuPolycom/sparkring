#include "spark_transport/tp4_c_api.h"

#include <cstddef>
#include <type_traits>

static_assert(std::is_standard_layout_v<spark_tp4_config>);
static_assert(std::is_trivially_copyable_v<spark_tp4_config>);
static_assert(sizeof(void*) != 8 || sizeof(spark_tp4_config) == 64);
static_assert(sizeof(void*) != 8 || offsetof(spark_tp4_config, rank) == 0);
static_assert(sizeof(void*) != 8 || offsetof(spark_tp4_config, peer0) == 8);
static_assert(sizeof(void*) != 8 || offsetof(spark_tp4_config, peer1) == 16);
static_assert(sizeof(void*) != 8 || offsetof(spark_tp4_config, device0) == 24);
static_assert(sizeof(void*) != 8 || offsetof(spark_tp4_config, device1) == 32);
static_assert(sizeof(void*) != 8 || offsetof(spark_tp4_config, gid0) == 40);
static_assert(sizeof(void*) != 8 || offsetof(spark_tp4_config, gid1) == 41);
static_assert(
    sizeof(void*) != 8 || offsetof(spark_tp4_config, control_port0) == 42);
static_assert(
    sizeof(void*) != 8 || offsetof(spark_tp4_config, control_port1) == 44);
static_assert(
    sizeof(void*) != 8 || offsetof(spark_tp4_config, payload_bytes) == 48);
static_assert(
    sizeof(void*) != 8 ||
    offsetof(spark_tp4_config, graph_submit_cpu_plus_one) == 56);
static_assert(
    sizeof(void*) != 8 ||
    offsetof(spark_tp4_config, graph_progress_cpu_plus_one) == 60);

static_assert(std::is_standard_layout_v<spark_tp4_graph_status>);
static_assert(std::is_trivially_copyable_v<spark_tp4_graph_status>);
static_assert(sizeof(spark_tp4_graph_status) == 56);
static_assert(offsetof(spark_tp4_graph_status, struct_size) == 0);
static_assert(offsetof(spark_tp4_graph_status, flags) == 4);
static_assert(offsetof(spark_tp4_graph_status, captured_nodes) == 8);
static_assert(offsetof(spark_tp4_graph_status, published_sequence) == 16);
static_assert(offsetof(spark_tp4_graph_status, consumed_sequence) == 24);
static_assert(offsetof(spark_tp4_graph_status, completed_sequence) == 32);
static_assert(offsetof(spark_tp4_graph_status, overflow_sequence) == 40);
static_assert(
    offsetof(spark_tp4_graph_status, graph_submit_cpu_plus_one) == 48);
static_assert(
    offsetof(spark_tp4_graph_status, graph_progress_cpu_plus_one) == 52);
static_assert(SPARK_TP4_ALLREDUCE_PROTOCOL_SERIAL_ACK == 0);
static_assert(SPARK_TP4_ALLREDUCE_PROTOCOL_TWO_SLOT_DEFERRED_ACK == 1);
static_assert(SPARK_TP4_GRAPH_KERNEL_FUSED == 0);
static_assert(SPARK_TP4_GRAPH_KERNEL_SPLIT_64K == 1);
static_assert(SPARK_TP4_GRAPH_KERNEL_TIERED_64K == 2);
static_assert(SPARK_TP4_GRAPH_STATUS_TWO_SLOT_DEFERRED_ACK == (1U << 7));
static_assert(SPARK_TP4_GRAPH_STATUS_SPLIT_64K == (1U << 8));
static_assert(SPARK_TP4_GRAPH_STATUS_TIERED_64K == (1U << 9));
static_assert(SPARK_TP4_GRAPH_STATUS_DUAL_PORT_STRIPED == (1U << 10));
static_assert(SPARK_TP4_WIRE_SCHEDULE_SEQUENTIAL == 0);
static_assert(SPARK_TP4_WIRE_SCHEDULE_DUAL_PORT_STRIPED == 1);
static_assert(
    (SPARK_TP4_GRAPH_STATUS_SPLIT_64K &
     SPARK_TP4_GRAPH_STATUS_TIERED_64K) == 0);

int main() { return 0; }
