#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

/*
 * Probe the mlx5 RDMA transmit flow domain without sending traffic.
 *
 * The program opens one named RDMA device, creates an RDMA-TX matcher for one
 * reserved UDP source port, and creates either a UDP-destination rewrite or a
 * fixed outer-Ethernet encapsulation action. The encapsulation mode preserves
 * every byte of the inner RoCE frame, including its invariant CRC.
 * Passing --attach installs the action until the bounded runtime expires or,
 * with --managed, until SIGINT/SIGTERM. A supervisor must stop dependent serving
 * processes before stopping a managed attachment. --attach is intended only
 * for an isolated device because the exact
 * source-port matcher selects every RDMA transmit packet with that reserved
 * UDP source port on the named device.
 */

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <infiniband/mlx5_api.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MATCH_PARAMETER_BYTES 0x180U
#define MLX5_MODIFICATION_TYPE_SET 0x1U
#define MLX5_MODIFICATION_FIELD_OUTER_ETHERTYPE 0x03U
#define MLX5_MODIFICATION_FIELD_OUTER_UDP_DESTINATION_PORT 0x0cU
#define MLX5_MATCH_CRITERIA_OUTER_HEADERS 0x1U
#define MLX5_OUTER_UDP_SOURCE_PORT_OFFSET 28U
#define OUTER_ETHERNET_HEADER_BYTES 14U
#define MAX_RUN_SECONDS 7200U
#define STOP_POLL_NANOSECONDS 100000000L

static volatile sig_atomic_t stop_requested;

static void request_stop(int signal_number)
{
    (void)signal_number;
    stop_requested = 1;
}

static int install_stop_handlers(void)
{
    struct sigaction action;

    memset(&action, 0, sizeof(action));
    action.sa_handler = request_stop;
    if (sigemptyset(&action.sa_mask) != 0 ||
        sigaction(SIGINT, &action, NULL) != 0 ||
        sigaction(SIGTERM, &action, NULL) != 0) {
        return -1;
    }
    return 0;
}

static int compare_timespec(const struct timespec *left,
                            const struct timespec *right)
{
    if (left->tv_sec != right->tv_sec) {
        return left->tv_sec < right->tv_sec ? -1 : 1;
    }
    if (left->tv_nsec != right->tv_nsec) {
        return left->tv_nsec < right->tv_nsec ? -1 : 1;
    }
    return 0;
}

static int wait_until_timeout_or_stop(unsigned int seconds)
{
    struct timespec deadline;

    if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) {
        return -1;
    }
    deadline.tv_sec += (time_t)seconds;
    while (!stop_requested) {
        struct timespec now;
        struct timespec wake;
        int rc;

        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
            return -1;
        }
        if (compare_timespec(&now, &deadline) >= 0) {
            return 0;
        }
        wake = now;
        wake.tv_nsec += STOP_POLL_NANOSECONDS;
        if (wake.tv_nsec >= 1000000000L) {
            wake.tv_sec += 1;
            wake.tv_nsec -= 1000000000L;
        }
        if (compare_timespec(&wake, &deadline) > 0) {
            wake = deadline;
        }
        rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, NULL);
        if (rc != 0 && rc != EINTR) {
            errno = rc;
            return -1;
        }
    }
    return 0;
}

static int wait_until_stop(void)
{
    const struct timespec interval = {0, STOP_POLL_NANOSECONDS};

    /* Polling avoids a lost-signal race between checking the flag and sleep. */
    while (!stop_requested) {
        if (nanosleep(&interval, NULL) != 0 && errno != EINTR) {
            return -1;
        }
    }
    return 0;
}

struct options {
    const char *device_name;
    uint16_t source_port;
    uint16_t replacement_port;
    uint16_t replacement_ethertype;
    uint8_t encapsulation_header[OUTER_ETHERNET_HEADER_BYTES];
    bool have_encapsulation_header;
    unsigned int run_seconds;
    bool attach;
    bool managed;
};

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s --device RDMA_DEVICE --source-port PORT "
            "(--replacement-port PORT | --replacement-ethertype TYPE | "
            "--encap-ethernet HEX28) "
            "[--attach (--run-seconds SECONDS | --managed)]\n",
            program);
}

static int parse_unsigned(const char *text, unsigned long maximum,
                          unsigned long *result)
{
    char *end = NULL;
    unsigned long value;

    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || text[0] == '\0' || end == text || *end != '\0' ||
        value > maximum) {
        return -1;
    }
    *result = value;
    return 0;
}

static int parse_options(int argc, char **argv, struct options *options)
{
    enum {
        OPT_DEVICE = 1000,
        OPT_SOURCE_PORT,
        OPT_REPLACEMENT_PORT,
        OPT_REPLACEMENT_ETHERTYPE,
        OPT_ENCAP_ETHERNET,
        OPT_ATTACH,
        OPT_RUN_SECONDS,
        OPT_MANAGED
    };
    static const struct option long_options[] = {
        {"device", required_argument, NULL, OPT_DEVICE},
        {"source-port", required_argument, NULL, OPT_SOURCE_PORT},
        {"replacement-port", required_argument, NULL, OPT_REPLACEMENT_PORT},
        {"replacement-ethertype", required_argument, NULL,
         OPT_REPLACEMENT_ETHERTYPE},
        {"encap-ethernet", required_argument, NULL, OPT_ENCAP_ETHERNET},
        {"attach", no_argument, NULL, OPT_ATTACH},
        {"run-seconds", required_argument, NULL, OPT_RUN_SECONDS},
        {"managed", no_argument, NULL, OPT_MANAGED},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option;

    static const char hex[] = "0123456789abcdef";

    memset(options, 0, sizeof(*options));
    while ((option = getopt_long(argc, argv, "h", long_options, NULL)) != -1) {
        unsigned long value;

        switch (option) {
        case OPT_DEVICE:
            options->device_name = optarg;
            break;
        case OPT_SOURCE_PORT:
            if (parse_unsigned(optarg, UINT16_MAX, &value) != 0 || value == 0) {
                fprintf(stderr, "--source-port must be from 1 to 65535\n");
                return -1;
            }
            options->source_port = (uint16_t)value;
            break;
        case OPT_REPLACEMENT_PORT:
            if (parse_unsigned(optarg, UINT16_MAX, &value) != 0 || value == 0) {
                fprintf(stderr, "--replacement-port must be from 1 to 65535\n");
                return -1;
            }
            options->replacement_port = (uint16_t)value;
            break;
        case OPT_REPLACEMENT_ETHERTYPE: {
            char *end = NULL;

            errno = 0;
            value = strtoul(optarg, &end, 0);
            if (errno != 0 || optarg[0] == '\0' || end == optarg ||
                *end != '\0' || value == 0 || value > UINT16_MAX) {
                fprintf(stderr,
                        "--replacement-ethertype must be from 1 to 0xffff\n");
                return -1;
            }
            options->replacement_ethertype = (uint16_t)value;
            break;
        }
        case OPT_ENCAP_ETHERNET:
            if (strlen(optarg) != OUTER_ETHERNET_HEADER_BYTES * 2U) {
                fprintf(stderr, "--encap-ethernet must contain 28 hexadecimal digits\n");
                return -1;
            }
            for (size_t index = 0; index < OUTER_ETHERNET_HEADER_BYTES; ++index) {
                char high = (char)tolower((unsigned char)optarg[index * 2U]);
                char low = (char)tolower((unsigned char)optarg[index * 2U + 1U]);
                const char *high_digit = strchr(hex, high);
                const char *low_digit = strchr(hex, low);

                if (high_digit == NULL || low_digit == NULL) {
                    fprintf(stderr, "--encap-ethernet contains a non-hexadecimal digit\n");
                    return -1;
                }
                options->encapsulation_header[index] = (uint8_t)(
                    ((unsigned int)(high_digit - hex) << 4U) |
                    (unsigned int)(low_digit - hex));
            }
            options->have_encapsulation_header = true;
            break;
        case OPT_ATTACH:
            options->attach = true;
            break;
        case OPT_MANAGED:
            options->managed = true;
            break;
        case OPT_RUN_SECONDS:
            if (parse_unsigned(optarg, MAX_RUN_SECONDS, &value) != 0 || value == 0) {
                fprintf(stderr, "--run-seconds must be from 1 to %u\n",
                        MAX_RUN_SECONDS);
                return -1;
            }
            options->run_seconds = (unsigned int)value;
            break;
        case 'h':
            usage(argv[0]);
            return 1;
        default:
            return -1;
        }
    }
    if (options->device_name == NULL || options->source_port == 0 ||
        ((options->replacement_port != 0) +
         (options->replacement_ethertype != 0) +
         options->have_encapsulation_header) != 1 ||
        optind != argc) {
        usage(argv[0]);
        return -1;
    }
    if (options->managed && options->run_seconds != 0) {
        fprintf(stderr, "--managed and --run-seconds are mutually exclusive\n");
        return -1;
    }
    if (options->managed && !options->attach) {
        fprintf(stderr, "--managed is valid only with --attach\n");
        return -1;
    }
    if (options->attach && options->run_seconds == 0 && !options->managed) {
        fprintf(stderr, "--attach requires --run-seconds or --managed\n");
        return -1;
    }
    if (!options->attach && options->run_seconds != 0) {
        fprintf(stderr, "--run-seconds is valid only with --attach\n");
        return -1;
    }
    return 0;
}

static struct ibv_context *open_context(const char *name)
{
    struct ibv_device **devices;
    struct ibv_context *context = NULL;
    int count = 0;
    int index;

    devices = ibv_get_device_list(&count);
    if (devices == NULL) {
        fprintf(stderr, "cannot enumerate RDMA devices: %s\n", strerror(errno));
        return NULL;
    }
    for (index = 0; index < count; ++index) {
        if (strcmp(ibv_get_device_name(devices[index]), name) == 0) {
            context = ibv_open_device(devices[index]);
            break;
        }
    }
    ibv_free_device_list(devices);
    if (context == NULL) {
        fprintf(stderr, "cannot open RDMA device %s: %s\n", name,
                strerror(errno));
    }
    return context;
}

static void encode_16bit_set(uint16_t field, uint16_t value,
                             uint32_t words[2])
{
    uint32_t control =
        (MLX5_MODIFICATION_TYPE_SET << 28) |
        ((uint32_t)field << 16) | 16U;

    words[0] = htonl(control);
    words[1] = htonl((uint32_t)value);
}

int main(int argc, char **argv)
{
    struct options options;
    struct mlx5dv_flow_match_parameters *mask = NULL;
    struct mlx5dv_flow_match_parameters *value = NULL;
    struct mlx5dv_flow_matcher_attr matcher_attributes = {0};
    struct mlx5dv_flow_matcher *matcher = NULL;
    struct ibv_flow_action *modify_action = NULL;
    struct ibv_flow_action *encapsulation_action = NULL;
    struct ibv_flow_action *selected_action = NULL;
    struct ibv_flow *flow = NULL;
    struct ibv_context *context = NULL;
    struct mlx5dv_flow_action_attr action = {0};
    uint32_t command_words[2];
    int parse_result;
    int result = EXIT_FAILURE;

    parse_result = parse_options(argc, argv, &options);
    if (parse_result != 0) {
        return parse_result > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (install_stop_handlers() != 0) {
        fprintf(stderr, "cannot install SIGINT/SIGTERM cleanup handlers: %s\n",
                strerror(errno));
        return EXIT_FAILURE;
    }
    context = open_context(options.device_name);
    if (context == NULL) {
        goto cleanup;
    }
    mask = calloc(1, sizeof(*mask) + MATCH_PARAMETER_BYTES);
    value = calloc(1, sizeof(*value) + MATCH_PARAMETER_BYTES);
    if (mask == NULL || value == NULL) {
        fprintf(stderr, "cannot allocate mlx5 flow match parameters\n");
        goto cleanup;
    }
    mask->match_sz = MATCH_PARAMETER_BYTES;
    value->match_sz = MATCH_PARAMETER_BYTES;
    /* fte_match_set_lyr_2_4 places outer UDP source port at byte 28.
     * A reserved source port scopes the marker to logical-diagonal QPs and
     * leaves every other QP on the same RDMA device unchanged.
     */
    ((uint8_t *)mask->match_buf)[MLX5_OUTER_UDP_SOURCE_PORT_OFFSET] = 0xffU;
    ((uint8_t *)mask->match_buf)[MLX5_OUTER_UDP_SOURCE_PORT_OFFSET + 1U] = 0xffU;
    {
        const uint16_t source_port = htons(options.source_port);
        memcpy((uint8_t *)value->match_buf + MLX5_OUTER_UDP_SOURCE_PORT_OFFSET,
               &source_port, sizeof(source_port));
    }
    matcher_attributes.type = IBV_FLOW_ATTR_NORMAL;
    matcher_attributes.priority = 0;
    matcher_attributes.match_criteria_enable =
        MLX5_MATCH_CRITERIA_OUTER_HEADERS;
    matcher_attributes.match_mask = mask;
    matcher_attributes.comp_mask = MLX5DV_FLOW_MATCHER_MASK_FT_TYPE;
    matcher_attributes.ft_type = MLX5DV_FLOW_TABLE_TYPE_RDMA_TX;
    matcher = mlx5dv_create_flow_matcher(context, &matcher_attributes);
    if (matcher == NULL) {
        fprintf(stderr, "RDMA-TX matcher creation failed: %s\n", strerror(errno));
        goto cleanup;
    }
    if (options.have_encapsulation_header) {
        encapsulation_action = mlx5dv_create_flow_action_packet_reformat(
            context, sizeof(options.encapsulation_header),
            options.encapsulation_header,
            MLX5DV_FLOW_ACTION_PACKET_REFORMAT_TYPE_L2_TO_L2_TUNNEL,
            MLX5DV_FLOW_TABLE_TYPE_RDMA_TX);
        if (encapsulation_action == NULL) {
            fprintf(stderr,
                    "RDMA-TX outer-Ethernet encapsulation creation failed: %s\n",
                    strerror(errno));
            goto cleanup;
        }
        selected_action = encapsulation_action;
    } else {
        const uint16_t field = options.replacement_ethertype != 0
                                   ? MLX5_MODIFICATION_FIELD_OUTER_ETHERTYPE
                                   : MLX5_MODIFICATION_FIELD_OUTER_UDP_DESTINATION_PORT;
        const uint16_t value_to_set = options.replacement_ethertype != 0
                                          ? options.replacement_ethertype
                                          : options.replacement_port;

        encode_16bit_set(field, value_to_set, command_words);
        modify_action = mlx5dv_create_flow_action_modify_header(
            context, sizeof(command_words), (uint64_t *)(void *)command_words,
            MLX5DV_FLOW_TABLE_TYPE_RDMA_TX);
        if (modify_action == NULL) {
            fprintf(stderr, "RDMA-TX header rewrite action creation failed: %s\n",
                    strerror(errno));
            goto cleanup;
        }
        selected_action = modify_action;
    }
    if (options.attach && !stop_requested) {
        action.type = MLX5DV_FLOW_ACTION_IBV_FLOW_ACTION;
        action.action = selected_action;
        flow = mlx5dv_create_flow(matcher, value, 1, &action);
        if (flow == NULL) {
            fprintf(stderr, "RDMA-TX UDP rewrite flow attachment failed: %s\n",
                    strerror(errno));
            goto cleanup;
        }
    }
    printf("{\"device\":\"%s\",\"rdma_tx_matcher\":true,"
           "\"action\":\"%s\",\"source_port\":%u,"
           "\"replacement_port\":%u,\"attached\":%s,"
           "\"requested_run_seconds\":%u,\"max_run_seconds\":%u,"
           "\"sigint_sigterm_cleanup\":true%s}\n",
           options.device_name,
           options.have_encapsulation_header ? "outer_ethernet_encapsulation"
               : options.replacement_ethertype != 0 ? "ethertype_rewrite"
                                                     : "udp_destination_rewrite",
           options.source_port, options.replacement_port,
           flow != NULL ? "true" : "false", options.run_seconds,
           MAX_RUN_SECONDS,
           options.managed ? ",\"managed\":true,\"lifetime_seconds\":null" : "");
    fflush(stdout);
    if (flow != NULL && (options.managed ? wait_until_stop()
                                        : wait_until_timeout_or_stop(options.run_seconds)) != 0) {
        fprintf(stderr, "flow wait failed: %s\n", strerror(errno));
        goto cleanup;
    }
    result = EXIT_SUCCESS;

cleanup:
    if (stop_requested) {
        result = EXIT_SUCCESS;
    }
    if (flow != NULL && ibv_destroy_flow(flow) != 0) {
        fprintf(stderr, "cannot destroy attached RDMA-TX flow: %s\n",
                strerror(errno));
        result = EXIT_FAILURE;
    }
    if (modify_action != NULL && ibv_destroy_flow_action(modify_action) != 0) {
        fprintf(stderr, "cannot destroy UDP rewrite action: %s\n", strerror(errno));
        result = EXIT_FAILURE;
    }
    if (encapsulation_action != NULL &&
        ibv_destroy_flow_action(encapsulation_action) != 0) {
        fprintf(stderr, "cannot destroy outer-Ethernet encapsulation action: %s\n",
                strerror(errno));
        result = EXIT_FAILURE;
    }
    if (matcher != NULL && mlx5dv_destroy_flow_matcher(matcher) != 0) {
        fprintf(stderr, "cannot destroy RDMA-TX matcher: %s\n", strerror(errno));
        result = EXIT_FAILURE;
    }
    free(value);
    free(mask);
    if (context != NULL && ibv_close_device(context) != 0) {
        fprintf(stderr, "cannot close RDMA device: %s\n", strerror(errno));
        result = EXIT_FAILURE;
    }
    return result;
}
