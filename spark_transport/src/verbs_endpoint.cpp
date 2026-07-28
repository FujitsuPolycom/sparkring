#include "spark_transport/verbs_endpoint.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>

namespace spark_transport {
namespace {

void check_verbs(int result, const char* operation) {
  if (result != 0) {
    throw std::system_error(result, std::generic_category(), operation);
  }
}

ibv_device* find_device(ibv_device** devices, const std::string& name) {
  for (std::size_t index = 0; devices[index] != nullptr; ++index) {
    if (name == ibv_get_device_name(devices[index])) {
      return devices[index];
    }
  }
  throw std::invalid_argument("RDMA device not found: " + name);
}

}  // namespace

VerbsEndpoint::VerbsEndpoint(const std::string& device_name, std::uint8_t port,
                             std::uint8_t gid_index, MemoryBuffer& buffer)
    : port_(port), gid_index_(gid_index), buffer_(buffer) {
  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr || device_count == 0) {
    throw std::runtime_error("no RDMA devices are available");
  }

  try {
    context_ = ibv_open_device(find_device(devices, device_name));
    if (context_ == nullptr) {
      throw std::runtime_error("ibv_open_device failed");
    }
    ibv_free_device_list(devices);
    devices = nullptr;

    protection_domain_ = ibv_alloc_pd(context_);
    if (protection_domain_ == nullptr) {
      throw std::runtime_error("ibv_alloc_pd failed");
    }

    memory_region_ = ibv_reg_mr(
        protection_domain_, buffer_.host_data(), buffer_.size(),
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ);
    if (memory_region_ == nullptr) {
      throw std::runtime_error("ibv_reg_mr failed for memory kind " +
                               std::string(memory_kind_name(buffer_.kind())));
    }

    completion_queue_ = ibv_create_cq(context_, 128, nullptr, nullptr, 0);
    if (completion_queue_ == nullptr) {
      throw std::runtime_error("ibv_create_cq failed");
    }

    ibv_qp_init_attr initialization{};
    initialization.send_cq = completion_queue_;
    initialization.recv_cq = completion_queue_;
    initialization.qp_type = IBV_QPT_RC;
    initialization.cap.max_send_wr = 128;
    initialization.cap.max_recv_wr = 16;
    initialization.cap.max_send_sge = 1;
    initialization.cap.max_recv_sge = 1;
    initialization.cap.max_inline_data = 64;
    initialization.sq_sig_all = 0;
    queue_pair_ = ibv_create_qp(protection_domain_, &initialization);
    if (queue_pair_ == nullptr) {
      throw std::runtime_error("ibv_create_qp failed");
    }
    max_inline_data_ = initialization.cap.max_inline_data;

    ibv_qp_attr attributes{};
    attributes.qp_state = IBV_QPS_INIT;
    attributes.pkey_index = 0;
    attributes.port_num = port_;
    attributes.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                                 IBV_ACCESS_REMOTE_WRITE |
                                 IBV_ACCESS_REMOTE_READ;
    check_verbs(ibv_modify_qp(queue_pair_, &attributes,
                              IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                  IBV_QP_ACCESS_FLAGS),
                "ibv_modify_qp(INIT)");
  } catch (...) {
    if (devices != nullptr) {
      ibv_free_device_list(devices);
    }
    cleanup();
    throw;
  }
}

VerbsEndpoint::~VerbsEndpoint() { cleanup(); }

void VerbsEndpoint::cleanup() noexcept {
  if (queue_pair_ != nullptr) {
    ibv_destroy_qp(queue_pair_);
    queue_pair_ = nullptr;
  }
  if (completion_queue_ != nullptr) {
    ibv_destroy_cq(completion_queue_);
    completion_queue_ = nullptr;
  }
  if (memory_region_ != nullptr) {
    ibv_dereg_mr(memory_region_);
    memory_region_ = nullptr;
  }
  if (protection_domain_ != nullptr) {
    ibv_dealloc_pd(protection_domain_);
    protection_domain_ = nullptr;
  }
  if (context_ != nullptr) {
    ibv_close_device(context_);
    context_ = nullptr;
  }
}

EndpointInfo VerbsEndpoint::local_info() const {
  ibv_port_attr port_attributes{};
  check_verbs(ibv_query_port(context_, port_, &port_attributes),
              "ibv_query_port");
  ibv_gid gid{};
  check_verbs(ibv_query_gid(context_, port_, gid_index_, &gid),
              "ibv_query_gid");

  EndpointInfo info{};
  info.qp_number = queue_pair_->qp_num;
  info.rkey = memory_region_->rkey;
  info.address =
      reinterpret_cast<std::uint64_t>(memory_region_->addr);
  info.buffer_bytes = memory_region_->length;
  info.lid = port_attributes.lid;
  std::memcpy(info.gid, gid.raw, sizeof(info.gid));
  return info;
}

void VerbsEndpoint::connect(const EndpointInfo& remote) {
  if (remote.magic != kEndpointMagic ||
      remote.version != kEndpointVersion) {
    throw std::runtime_error("incompatible remote endpoint metadata");
  }
  if (remote.qp_number == 0 || remote.rkey == 0 || remote.address == 0 ||
      remote.buffer_bytes == 0) {
    throw std::runtime_error("remote endpoint metadata is incomplete");
  }
  remote_ = remote;

  ibv_qp_attr attributes{};
  attributes.qp_state = IBV_QPS_RTR;
  attributes.path_mtu = IBV_MTU_4096;
  attributes.dest_qp_num = remote_.qp_number;
  attributes.rq_psn = 0;
  attributes.max_dest_rd_atomic = 1;
  attributes.min_rnr_timer = 12;
  attributes.ah_attr.is_global = 1;
  attributes.ah_attr.dlid = remote_.lid;
  attributes.ah_attr.sl = 0;
  attributes.ah_attr.src_path_bits = 0;
  attributes.ah_attr.port_num = port_;
  std::memcpy(attributes.ah_attr.grh.dgid.raw, remote_.gid,
              sizeof(remote_.gid));
  attributes.ah_attr.grh.sgid_index = gid_index_;
  attributes.ah_attr.grh.hop_limit = 1;

  check_verbs(
      ibv_modify_qp(queue_pair_, &attributes,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
      "ibv_modify_qp(RTR)");

  std::memset(&attributes, 0, sizeof(attributes));
  attributes.qp_state = IBV_QPS_RTS;
  attributes.sq_psn = 0;
  attributes.timeout = 14;
  attributes.retry_cnt = 7;
  attributes.rnr_retry = 7;
  attributes.max_rd_atomic = 1;
  check_verbs(
      ibv_modify_qp(queue_pair_, &attributes,
                    IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                        IBV_QP_MAX_QP_RD_ATOMIC),
      "ibv_modify_qp(RTS)");
}

void VerbsEndpoint::write(std::size_t local_offset,
                          std::size_t remote_offset, std::size_t bytes,
                          std::uint64_t work_id, bool signaled) {
  if (local_offset + bytes > buffer_.size()) {
    throw std::out_of_range("local RDMA write exceeds its buffer");
  }
  if (remote_offset + bytes > remote_.buffer_bytes) {
    throw std::out_of_range("remote RDMA write exceeds its buffer");
  }
  if (bytes > UINT32_MAX) {
    throw std::out_of_range("single RDMA write exceeds verbs SGE length");
  }

  ibv_sge scatter_gather{};
  scatter_gather.addr =
      reinterpret_cast<std::uint64_t>(buffer_.host_data()) + local_offset;
  scatter_gather.length = static_cast<std::uint32_t>(bytes);
  scatter_gather.lkey = memory_region_->lkey;

  ibv_send_wr request{};
  request.wr_id = work_id;
  request.sg_list = &scatter_gather;
  request.num_sge = 1;
  request.opcode = IBV_WR_RDMA_WRITE;
  request.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  if (bytes <= max_inline_data_) {
    request.send_flags |= IBV_SEND_INLINE;
  }
  request.wr.rdma.remote_addr = remote_.address + remote_offset;
  request.wr.rdma.rkey = remote_.rkey;

  ibv_send_wr* bad_request = nullptr;
  check_verbs(ibv_post_send(queue_pair_, &request, &bad_request),
              "ibv_post_send");
}

void VerbsEndpoint::wait_for_send(std::uint64_t expected_work_id) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    ibv_wc completion{};
    const int count = ibv_poll_cq(completion_queue_, 1, &completion);
    if (count < 0) {
      throw std::runtime_error("ibv_poll_cq failed");
    }
    if (count == 0) {
      continue;
    }
    if (completion.status != IBV_WC_SUCCESS) {
      throw std::runtime_error(
          std::string("RDMA completion failed: ") +
          ibv_wc_status_str(completion.status));
    }
    if (completion.wr_id != expected_work_id) {
      throw std::runtime_error("unexpected RDMA completion work ID");
    }
    return;
  }
  throw std::runtime_error("timed out waiting for RDMA completion");
}

}  // namespace spark_transport
