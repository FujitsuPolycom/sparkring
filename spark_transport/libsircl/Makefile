CC ?= cc
.DEFAULT_GOAL := all
PYTHON ?= python3
NVCC ?= nvcc
BUILD ?= build
# The library's version, from the one version source (VERSION).
VERSION := $(shell cat VERSION)
CPPFLAGS += -Iinclude -Isrc -DLIBSIRCL_CPU_ONLY -DNCCL_OS_LINUX -DLIBSIRCL_VERSION='"$(VERSION)"'
CFLAGS ?= -O2 -g
# Diagnostic messages are bounded copies that truncate by design.
CFLAGS += -std=c11 -fPIC -Wall -Wextra -Werror -Wno-format-truncation -pthread -MMD -MP

# The library: API layer, bootstrap, fabric arithmetic, session engine, CUDA
# driver loader and the embedded kernel packs, plus SIRCL's native proxy and
# SIRCL's point-to-point native library, each twice: over the real libibverbs
# (loaded at run time) and over the shared-memory verbs stand-in (emulation).
SOURCES = src/api.c src/bootstrap.c src/stubs.c src/aliases.c src/device_api.c src/fabric.c src/engine.c src/cuda_api.c \
          src/kernelpack.c
TRANSPORT_HW = src/transport/proxy_hw.c src/transport/p2p_hw.c src/transport/verbs_dl.c
TRANSPORT_EMU = src/transport/proxy_emu.c src/transport/p2p_emu.c src/transport/shm_verbs.c
OBJECTS = $(SOURCES:src/%.c=$(BUILD)/%.o) $(TRANSPORT_HW:src/transport/%.c=$(BUILD)/transport/%.o) \
          $(TRANSPORT_EMU:src/transport/%.c=$(BUILD)/transport/%.o) $(BUILD)/kernels_fatbin.o $(BUILD)/fold_fatbin.o \
          $(BUILD)/links_fatbin.o $(BUILD)/p2p_fatbin.o
-include $(OBJECTS:.o=.d)

# Four kernel packs, CUDA C++ compiled ahead of time for GB10 (sm_121) and
# the workstation's RTX 5090 (sm_120): the transport pack (all-reduce,
# all-gather, reduce-scatter and all-to-all over the ring session), the fold
# pack (the local rank-ordered reduction of every other datatype and op), the
# link pack (the chain all-reduce and the link collectives) and the
# point-to-point pack (the channel send and receive kernels). Each fatbin is
# prebuilt in kernels/prebuilt/ with its SHA-256, so building the library needs
# no CUDA toolkit; `make kernels` regenerates them with nvcc and `make
# kernels-check` proves the prebuilt files are what the sources and pinned
# compiler produce.
KERNEL_SOURCES = kernels/sircl_kernels.cu
FATBIN = kernels/prebuilt/sircl_kernels.fatbin
FOLD_SOURCES = kernels/sircl_fold.cu
FOLD_FATBIN = kernels/prebuilt/sircl_fold.fatbin
LINKS_SOURCES = kernels/sircl_links.cu
LINKS_FATBIN = kernels/prebuilt/sircl_links.fatbin
P2P_SOURCES = kernels/sircl_p2p.cu
P2P_FATBIN = kernels/prebuilt/sircl_p2p.fatbin
# SIRCL's point-to-point native library is vendored byte-identical (src/transport/sircl_p2p_proxy.c, SHA-256
# in its .sha256 file); the build refuses a copy of other contents. P2P_FEATURES=1 (the default) compiles the
# engine's setup check of the library's local feature word (p2p_local_features bit 0: p2p_destroy counts
# failed verbs calls), which needs a copy that defines it (SIRCL change LF); the build refuses a copy without
# it. P2P_FEATURES=0 builds without the check.
P2P_PROXY = src/transport/sircl_p2p_proxy.c
P2P_FEATURES ?= 1
ifeq ($(P2P_FEATURES),1)
CPPFLAGS += -DSCCL_P2P_LOCAL_FEATURES
endif
NVCCFLAGS = -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121

.PHONY: all check clean generate kernels kernels-check emulation-tools mpi-shim
all: $(BUILD)/libsircl.so

$(BUILD) $(BUILD)/transport:
	mkdir -p $@
$(BUILD)/%.o: src/%.c include/nccl.h src/internal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
# SIRCL's native proxy is vendored byte-identical as well (src/transport/sircl_roce_proxy.c, SHA-256 in its
# .sha256 file); the build refuses a copy of other contents. The engine checks its local feature word
# (roce_local_features) at setup. Both vendored copies carry LF line endings, the bytes SIRCL's release
# builds its natives from; a copy with carriage returns is refused even when its .sha256 was recorded.
ROCE_PROXY = src/transport/sircl_roce_proxy.c
$(BUILD)/roce_source.ok: $(ROCE_PROXY) $(ROCE_PROXY).sha256 | $(BUILD)
	test "$$(sha256sum $(ROCE_PROXY) | cut -d' ' -f1)" = "$$(cat $(ROCE_PROXY).sha256)" || \
	  { echo "$(ROCE_PROXY) differs from the vendored SIRCL source recorded in $(ROCE_PROXY).sha256"; exit 1; }
	! grep -q "$$(printf '\r')" $(ROCE_PROXY) || \
	  { echo "$(ROCE_PROXY) has CRLF line endings; vendor the LF bytes of SIRCL's oneshot/_roce_proxy.c"; exit 1; }
	grep -q 'roce_local_features' $(ROCE_PROXY) || \
	  { echo "$(ROCE_PROXY) predates SIRCL change LF (no roce_local_features); vendor SIRCL's oneshot/_roce_proxy.c with LF"; exit 1; }
	touch $@
$(BUILD)/transport/proxy_hw.o: src/transport/proxy_hw.c src/transport/sircl_roce_proxy.c $(BUILD)/roce_source.ok | $(BUILD)/transport
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/transport/verbs_dl.o: src/transport/verbs_dl.c | $(BUILD)/transport
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/transport/proxy_emu.o: src/transport/proxy_emu.c src/transport/sircl_roce_proxy.c $(BUILD)/roce_source.ok | $(BUILD)/transport
	$(CC) -Isrc/transport/fake_verbs $(CFLAGS) -c $< -o $@
# A stamp named for the P2P_FEATURES value, so changing it rebuilds the engine and repeats the source check.
$(BUILD)/p2p_features-%.stamp: | $(BUILD)
	rm -f $(BUILD)/p2p_features-*.stamp
	touch $@
$(BUILD)/engine.o: $(BUILD)/p2p_features-$(P2P_FEATURES).stamp
$(BUILD)/p2p_source.ok: $(P2P_PROXY) $(P2P_PROXY).sha256 $(BUILD)/p2p_features-$(P2P_FEATURES).stamp | $(BUILD)
	test "$$(sha256sum $(P2P_PROXY) | cut -d' ' -f1)" = "$$(cat $(P2P_PROXY).sha256)" || \
	  { echo "$(P2P_PROXY) differs from the vendored SIRCL source recorded in $(P2P_PROXY).sha256"; exit 1; }
	! grep -q "$$(printf '\r')" $(P2P_PROXY) || \
	  { echo "$(P2P_PROXY) has CRLF line endings; vendor the LF bytes of SIRCL's p2p/_p2p_proxy.c"; exit 1; }
	if [ "$(P2P_FEATURES)" = 1 ] && ! grep -q 'p2p_local_features' $(P2P_PROXY); then \
	  echo "P2P_FEATURES=1: $(P2P_PROXY) predates SIRCL change LF (no p2p_local_features); vendor SIRCL's"; \
	  echo "p2p/_p2p_proxy.c with LF, or build with P2P_FEATURES=0"; exit 1; fi
	touch $@
$(BUILD)/transport/p2p_hw.o: src/transport/p2p_hw.c src/transport/p2p_names.h $(BUILD)/p2p_source.ok | $(BUILD)/transport
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/transport/p2p_emu.o: src/transport/p2p_emu.c src/transport/p2p_names.h $(BUILD)/p2p_source.ok | $(BUILD)/transport
	$(CC) -Isrc/transport/fake_verbs $(CFLAGS) -c $< -o $@
$(BUILD)/transport/shm_verbs.o: src/transport/shm_verbs.c | $(BUILD)/transport
	$(CC) -Isrc/transport/fake_verbs -Isrc/transport $(CFLAGS) -c $< -o $@
$(BUILD)/kernels_fatbin.c: $(FATBIN) $(FATBIN).sha256 tools/embed_fatbin.py | $(BUILD)
	$(PYTHON) tools/embed_fatbin.py $(FATBIN) $@ --check-sha256 $(FATBIN).sha256
$(BUILD)/fold_fatbin.c: $(FOLD_FATBIN) $(FOLD_FATBIN).sha256 tools/embed_fatbin.py | $(BUILD)
	$(PYTHON) tools/embed_fatbin.py $(FOLD_FATBIN) $@ --check-sha256 $(FOLD_FATBIN).sha256 --symbol sccl_fold_fatbin
$(BUILD)/links_fatbin.c: $(LINKS_FATBIN) $(LINKS_FATBIN).sha256 tools/embed_fatbin.py | $(BUILD)
	$(PYTHON) tools/embed_fatbin.py $(LINKS_FATBIN) $@ --check-sha256 $(LINKS_FATBIN).sha256 --symbol sccl_links_fatbin
$(BUILD)/p2p_fatbin.c: $(P2P_FATBIN) $(P2P_FATBIN).sha256 tools/embed_fatbin.py | $(BUILD)
	$(PYTHON) tools/embed_fatbin.py $(P2P_FATBIN) $@ --check-sha256 $(P2P_FATBIN).sha256 --symbol sccl_p2p_fatbin
$(BUILD)/kernels_fatbin.o $(BUILD)/fold_fatbin.o $(BUILD)/links_fatbin.o $(BUILD)/p2p_fatbin.o: $(BUILD)/%.o: $(BUILD)/%.c
	$(CC) -fPIC -O0 -c $< -o $@
# The file is libsircl.so.<version>; its SONAME is the interface name libnccl.so.2, which the dynamic linker
# and ctypes.CDLL("libnccl.so.2") resolve. libsircl.so is a development link to it.
$(BUILD)/libsircl.so: $(BUILD)/libsircl.so.$(VERSION)
	ln -sf libsircl.so.$(VERSION) $@
$(BUILD)/libsircl.so.$(VERSION): $(OBJECTS) exports.map $(BUILD)/entries.ok
	$(CC) -shared -pthread $(LDFLAGS) -Wl,-soname,libnccl.so.2 -Wl,-Bsymbolic-functions -Wl,--version-script=exports.map -o $@ $(OBJECTS) -ldl

# The kernel-entry check (tests/check_entries.c): the library's pack loader runs on the embedded fatbins
# through a stand-in CUDA driver (tests/fake_cuda.c) that reads their cubins offline, so every entry the
# loader names must be defined in every architecture's cubin. The library links only after it passes.
ARCHITECTURES = $(words $(filter -gencode,$(NVCCFLAGS)))
$(BUILD)/fake-cuda/libcuda.so.1: tests/fake_cuda.c | $(BUILD)
	mkdir -p $(BUILD)/fake-cuda
	$(CC) -std=c11 -O2 -fPIC -Wall -Wextra -Werror -fvisibility=hidden -shared -Wl,-soname,libcuda.so.1 -o $@ $<
$(BUILD)/check_entries: tests/check_entries.c $(BUILD)/kernelpack.o $(BUILD)/cuda_api.o $(BUILD)/kernels_fatbin.o \
                        $(BUILD)/fold_fatbin.o $(BUILD)/links_fatbin.o $(BUILD)/p2p_fatbin.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ -ldl
$(BUILD)/entries.ok: $(BUILD)/check_entries $(BUILD)/fake-cuda/libcuda.so.1
	LD_LIBRARY_PATH=$(abspath $(BUILD)/fake-cuda) $(BUILD)/check_entries $(ARCHITECTURES)
	touch $@

kernels:
	$(NVCC) $(NVCCFLAGS) -fatbin -o $(FATBIN) $(KERNEL_SOURCES)
	sha256sum $(FATBIN) | cut -d' ' -f1 > $(FATBIN).sha256
	$(NVCC) $(NVCCFLAGS) -fatbin -o $(FOLD_FATBIN) $(FOLD_SOURCES)
	sha256sum $(FOLD_FATBIN) | cut -d' ' -f1 > $(FOLD_FATBIN).sha256
	$(NVCC) $(NVCCFLAGS) -fatbin -o $(LINKS_FATBIN) $(LINKS_SOURCES)
	sha256sum $(LINKS_FATBIN) | cut -d' ' -f1 > $(LINKS_FATBIN).sha256
	$(NVCC) $(NVCCFLAGS) -fatbin -o $(P2P_FATBIN) $(P2P_SOURCES)
	sha256sum $(P2P_FATBIN) | cut -d' ' -f1 > $(P2P_FATBIN).sha256
kernels-check: | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -fatbin -o $(BUILD)/check.fatbin $(KERNEL_SOURCES)
	test "$$(sha256sum $(BUILD)/check.fatbin | cut -d' ' -f1)" = "$$(cat $(FATBIN).sha256)"
	@echo "prebuilt transport pack matches the sources: $$(cat $(FATBIN).sha256)"
	$(NVCC) $(NVCCFLAGS) -fatbin -o $(BUILD)/check-fold.fatbin $(FOLD_SOURCES)
	test "$$(sha256sum $(BUILD)/check-fold.fatbin | cut -d' ' -f1)" = "$$(cat $(FOLD_FATBIN).sha256)"
	@echo "prebuilt fold pack matches the sources: $$(cat $(FOLD_FATBIN).sha256)"
	$(NVCC) $(NVCCFLAGS) -fatbin -o $(BUILD)/check-links.fatbin $(LINKS_SOURCES)
	test "$$(sha256sum $(BUILD)/check-links.fatbin | cut -d' ' -f1)" = "$$(cat $(LINKS_FATBIN).sha256)"
	@echo "prebuilt link pack matches the sources: $$(cat $(LINKS_FATBIN).sha256)"
	$(NVCC) $(NVCCFLAGS) -fatbin -o $(BUILD)/check-p2p.fatbin $(P2P_SOURCES)
	test "$$(sha256sum $(BUILD)/check-p2p.fatbin | cut -d' ' -f1)" = "$$(cat $(P2P_FATBIN).sha256)"
	@echo "prebuilt point-to-point pack matches the sources: $$(cat $(P2P_FATBIN).sha256)"

# GPU-emulation tools: the kernel pack's test entry points for the mixed-group check.
emulation-tools: $(BUILD)/libsirclkp_test.so
$(BUILD)/libsirclkp_test.so: tests/emulation/kp_shim.c src/kernelpack.c src/cuda_api.c $(BUILD)/kernels_fatbin.o \
                            $(BUILD)/fold_fatbin.o $(BUILD)/links_fatbin.o $(BUILD)/p2p_fatbin.o
	$(CC) $(CFLAGS) -Isrc -fvisibility=hidden -shared -o $@ tests/emulation/kp_shim.c src/kernelpack.c \
	  src/cuda_api.c $(BUILD)/kernels_fatbin.o $(BUILD)/fold_fatbin.o $(BUILD)/links_fatbin.o $(BUILD)/p2p_fatbin.o -ldl

# A launcher-free MPI subset for nccl-tests (MPI=1 MPI_HOME=$(BUILD)/mpi-shim); see RUNBOOK.md.
mpi-shim: $(BUILD)/mpi-shim/lib/libmpi.so
$(BUILD)/mpi-shim/lib/libmpi.so: tools/mpi-shim/mpi_shim.c tools/mpi-shim/include/mpi.h
	mkdir -p $(BUILD)/mpi-shim/lib $(BUILD)/mpi-shim/include
	cp tools/mpi-shim/include/mpi.h $(BUILD)/mpi-shim/include/mpi.h
	$(CC) -std=c11 -O2 -fPIC -Wall -Wextra -Werror -shared -pthread -Itools/mpi-shim/include -o $@ tools/mpi-shim/mpi_shim.c

$(BUILD)/test_fabric: tests/test_fabric.c $(BUILD)/fabric.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@
check: all $(BUILD)/test_fabric
	$(BUILD)/test_fabric
	$(PYTHON) tests/test_abi.py --library $(abspath $(BUILD)/libsircl.so)
	$(PYTHON) tests/test_api.py --library $(abspath $(BUILD)/libsircl.so)
	$(PYTHON) tests/test_bootstrap.py
	$(PYTHON) tests/test_lifecycle.py --library $(abspath $(BUILD)/libsircl.so)
	$(PYTHON) tests/test_engine_cpu.py --library $(abspath $(BUILD)/libsircl.so)
	$(PYTHON) tests/test_shm_verbs.py
	$(PYTHON) tests/test_vendored_sources.py
	$(PYTHON) tests/test_p2p_native.py
	$(PYTHON) tests/test_site_routes.py
	$(PYTHON) tests/test_mpi_shim.py
	$(PYTHON) tests/test_check_nccl_tests.py
	$(PYTHON) tests/test_kernel_entries.py --build $(BUILD)
generate:
	$(PYTHON) tools/generate_api.py
clean:
	rm -rf $(BUILD)
