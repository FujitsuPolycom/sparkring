#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

/*
 * Tag RoCE v2 packets by destination IPv4 address in the mlx5 RDMA transmit
 * flow domain.
 *
 * One process owns one RDMA device. Each --rule ADDRESS=ETHERTYPE installs an
 * RDMA-TX flow that matches outer EtherType 0x0800 and the exact outer
 * destination IPv4 address and rewrites the outer EtherType to ETHERTYPE.
 * Every RDMA queue pair on the device that sends to ADDRESS is tagged,
 * whatever its flow label or UDP source port, so relays downstream can act
 * on the tag. The rules stay installed until SIGINT/SIGTERM (--managed) or
 * for --run-seconds; process exit removes them.
 *
 * Build: cc -O2 -Wall -Wextra mesh_marker.c -o ring8-mesh-marker -libverbs -lmlx5
 */

#include <arpa/inet.h>
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
#define MLX5_MATCH_CRITERIA_OUTER_HEADERS 0x1U
/* Byte offsets inside fte_match_set_lyr_2_4 (outer headers). */
#define OUTER_ETHERTYPE_OFFSET 6U
#define OUTER_DST_IPV4_OFFSET 60U
#define MAX_RULES 64
#define MAX_RUN_SECONDS 7200U
#define STOP_POLL_NANOSECONDS 100000000L

static volatile sig_atomic_t stop_requested;

static void request_stop(int signal_number)
{
    (void)signal_number;
    stop_requested = 1;
}

struct rule {
    struct in_addr address;
    uint16_t ethertype;
    struct ibv_flow_action *action;
    struct ibv_flow *flow;
};

struct options {
    const char *device_name;
    struct rule rules[MAX_RULES];
    int rule_count;
    unsigned int run_seconds;
    bool managed;
};

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s --device RDMA_DEVICE --rule IPV4=ETHERTYPE [--rule ...] "
            "(--run-seconds SECONDS | --managed)\n",
            program);
}

static int parse_rule(const char *text, struct rule *rule)
{
    char address[INET_ADDRSTRLEN];
    const char *equals = strchr(text, '=');
    char *end = NULL;
    unsigned long value;
    size_t length;

    if (equals == NULL) {
        return -1;
    }
    length = (size_t)(equals - text);
    if (length == 0 || length >= sizeof(address)) {
        return -1;
    }
    memcpy(address, text, length);
    address[length] = '\0';
    if (inet_pton(AF_INET, address, &rule->address) != 1) {
        return -1;
    }
    errno = 0;
    value = strtoul(equals + 1, &end, 0);
    if (errno != 0 || end == equals + 1 || *end != '\0' || value == 0 ||
        value > UINT16_MAX || value == 0x0800U) {
        return -1;
    }
    rule->ethertype = (uint16_t)value;
    return 0;
}

static int parse_options(int argc, char **argv, struct options *options)
{
    enum { OPT_DEVICE = 1000, OPT_RULE, OPT_RUN_SECONDS, OPT_MANAGED };
    static const struct option long_options[] = {
        {"device", required_argument, NULL, OPT_DEVICE},
        {"rule", required_argument, NULL, OPT_RULE},
        {"run-seconds", required_argument, NULL, OPT_RUN_SECONDS},
        {"managed", no_argument, NULL, OPT_MANAGED},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option;

    memset(options, 0, sizeof(*options));
    while ((option = getopt_long(argc, argv, "h", long_options, NULL)) != -1) {
        switch (option) {
        case OPT_DEVICE:
            options->device_name = optarg;
            break;
        case OPT_RULE:
            if (options->rule_count >= MAX_RULES ||
                parse_rule(optarg, &options->rules[options->rule_count]) != 0) {
                fprintf(stderr, "--rule must be IPV4=ETHERTYPE (not 0x0800), at most %d\n",
                        MAX_RULES);
                return -1;
            }
            options->rule_count++;
            break;
        case OPT_RUN_SECONDS: {
            char *end = NULL;
            unsigned long value;

            errno = 0;
            value = strtoul(optarg, &end, 10);
            if (errno != 0 || end == optarg || *end != '\0' || value == 0 ||
                value > MAX_RUN_SECONDS) {
                fprintf(stderr, "--run-seconds must be from 1 to %u\n", MAX_RUN_SECONDS);
                return -1;
            }
            options->run_seconds = (unsigned int)value;
            break;
        }
        case OPT_MANAGED:
            options->managed = true;
            break;
        case 'h':
            usage(argv[0]);
            return 1;
        default:
            return -1;
        }
    }
    if (options->device_name == NULL || options->rule_count == 0 || optind != argc ||
        options->managed == (options->run_seconds != 0)) {
        usage(argv[0]);
        return -1;
    }
    for (int i = 0; i < options->rule_count; ++i) {
        for (int j = 0; j < i; ++j) {
            if (options->rules[i].address.s_addr == options->rules[j].address.s_addr) {
                fprintf(stderr, "duplicate --rule address\n");
                return -1;
            }
        }
    }
    return 0;
}

static struct ibv_context *open_context(const char *name)
{
    struct ibv_device **devices;
    struct ibv_context *context = NULL;
    int count = 0;

    devices = ibv_get_device_list(&count);
    if (devices == NULL) {
        fprintf(stderr, "cannot enumerate RDMA devices: %s\n", strerror(errno));
        return NULL;
    }
    for (int index = 0; index < count; ++index) {
        if (strcmp(ibv_get_device_name(devices[index]), name) == 0) {
            context = ibv_open_device(devices[index]);
            break;
        }
    }
    ibv_free_device_list(devices);
    if (context == NULL) {
        fprintf(stderr, "cannot open RDMA device %s: %s\n", name, strerror(errno));
    }
    return context;
}

static int wait_for_stop(const struct options *options)
{
    struct timespec deadline;

    if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) {
        return -1;
    }
    deadline.tv_sec += (time_t)options->run_seconds;
    while (!stop_requested) {
        const struct timespec interval = {0, STOP_POLL_NANOSECONDS};
        struct timespec now;

        if (!options->managed) {
            if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
                return -1;
            }
            if (now.tv_sec > deadline.tv_sec ||
                (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec)) {
                return 0;
            }
        }
        if (nanosleep(&interval, NULL) != 0 && errno != EINTR) {
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    struct options options;
    struct mlx5dv_flow_match_parameters *mask = NULL;
    struct mlx5dv_flow_match_parameters *value = NULL;
    struct mlx5dv_flow_matcher_attr matcher_attributes = {0};
    struct mlx5dv_flow_matcher *matcher = NULL;
    struct ibv_context *context = NULL;
    struct sigaction stop_action;
    int parse_result;
    int installed = 0;
    int result = EXIT_FAILURE;

    parse_result = parse_options(argc, argv, &options);
    if (parse_result != 0) {
        return parse_result > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    memset(&stop_action, 0, sizeof(stop_action));
    stop_action.sa_handler = request_stop;
    if (sigemptyset(&stop_action.sa_mask) != 0 || sigaction(SIGINT, &stop_action, NULL) != 0 ||
        sigaction(SIGTERM, &stop_action, NULL) != 0) {
        fprintf(stderr, "cannot install SIGINT/SIGTERM handlers: %s\n", strerror(errno));
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
    memset((uint8_t *)mask->match_buf + OUTER_ETHERTYPE_OFFSET, 0xff, 2);
    memset((uint8_t *)mask->match_buf + OUTER_DST_IPV4_OFFSET, 0xff, 4);
    matcher_attributes.type = IBV_FLOW_ATTR_NORMAL;
    matcher_attributes.priority = 0;
    matcher_attributes.match_criteria_enable = MLX5_MATCH_CRITERIA_OUTER_HEADERS;
    matcher_attributes.match_mask = mask;
    matcher_attributes.comp_mask = MLX5DV_FLOW_MATCHER_MASK_FT_TYPE;
    matcher_attributes.ft_type = MLX5DV_FLOW_TABLE_TYPE_RDMA_TX;
    matcher = mlx5dv_create_flow_matcher(context, &matcher_attributes);
    if (matcher == NULL) {
        fprintf(stderr, "RDMA-TX matcher creation failed: %s\n", strerror(errno));
        goto cleanup;
    }
    for (int i = 0; i < options.rule_count && !stop_requested; ++i) {
        struct rule *rule = &options.rules[i];
        struct mlx5dv_flow_action_attr action = {0};
        const uint16_t ipv4_ethertype = htons(0x0800U);
        uint32_t words[2];

        words[0] = htonl((MLX5_MODIFICATION_TYPE_SET << 28) |
                         ((uint32_t)MLX5_MODIFICATION_FIELD_OUTER_ETHERTYPE << 16) | 16U);
        words[1] = htonl((uint32_t)rule->ethertype);
        rule->action = mlx5dv_create_flow_action_modify_header(
            context, sizeof(words), (uint64_t *)(void *)words, MLX5DV_FLOW_TABLE_TYPE_RDMA_TX);
        if (rule->action == NULL) {
            fprintf(stderr, "RDMA-TX EtherType rewrite action creation failed: %s\n",
                    strerror(errno));
            goto cleanup;
        }
        memset(value->match_buf, 0, MATCH_PARAMETER_BYTES);
        memcpy((uint8_t *)value->match_buf + OUTER_ETHERTYPE_OFFSET, &ipv4_ethertype, 2);
        memcpy((uint8_t *)value->match_buf + OUTER_DST_IPV4_OFFSET, &rule->address.s_addr, 4);
        action.type = MLX5DV_FLOW_ACTION_IBV_FLOW_ACTION;
        action.action = rule->action;
        rule->flow = mlx5dv_create_flow(matcher, value, 1, &action);
        if (rule->flow == NULL) {
            fprintf(stderr, "RDMA-TX flow for %s failed: %s\n", inet_ntoa(rule->address),
                    strerror(errno));
            goto cleanup;
        }
        installed++;
    }
    printf("{\"device\":\"%s\",\"rules\":[", options.device_name);
    for (int i = 0; i < options.rule_count; ++i) {
        printf("%s{\"dst\":\"%s\",\"ethertype\":\"0x%04x\"}", i ? "," : "",
               inet_ntoa(options.rules[i].address), options.rules[i].ethertype);
    }
    printf("],\"installed\":%d,\"managed\":%s,\"run_seconds\":%u}\n", installed,
           options.managed ? "true" : "false", options.run_seconds);
    fflush(stdout);
    if (wait_for_stop(&options) != 0) {
        fprintf(stderr, "wait failed: %s\n", strerror(errno));
        goto cleanup;
    }
    result = EXIT_SUCCESS;

cleanup:
    if (stop_requested) {
        result = EXIT_SUCCESS;
    }
    for (int i = 0; i < options.rule_count; ++i) {
        if (options.rules[i].flow != NULL && ibv_destroy_flow(options.rules[i].flow) != 0) {
            result = EXIT_FAILURE;
        }
        if (options.rules[i].action != NULL &&
            ibv_destroy_flow_action(options.rules[i].action) != 0) {
            result = EXIT_FAILURE;
        }
    }
    if (matcher != NULL && mlx5dv_destroy_flow_matcher(matcher) != 0) {
        result = EXIT_FAILURE;
    }
    free(value);
    free(mask);
    if (context != NULL && ibv_close_device(context) != 0) {
        result = EXIT_FAILURE;
    }
    return result;
}
