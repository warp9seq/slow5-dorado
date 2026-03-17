#include "torch_utils/gpu_monitor.h"

#include "utils/sys_utils.h"

#if (defined(_WIN32) || defined(__linux__)) && !DORADO_ROCM_BUILD
#define HAS_NVML 1
#else
#define HAS_NVML 0
#endif

// ROCm SMI support (Linux ROCm builds only)
#if defined(__linux__) && DORADO_ROCM_BUILD
#define HAS_RSMI 1
#else
#define HAS_RSMI 0
#endif

#if HAS_NVML
#include "utils/scoped_trace_log.h"
#include "utils/string_utils.h"

#include <nvml.h>
#if defined(_WIN32)
#include <Windows.h>
#else  // _WIN32
#include <dlfcn.h>
#endif  // _WIN32
#if DORADO_ORIN
#include <torch/torch.h>
#endif  // DORADO_ORIN
#endif  // HAS_NVML

#if HAS_RSMI
#include "utils/scoped_trace_log.h"
#include "utils/string_utils.h"

#include <dlfcn.h>
#endif  // HAS_RSMI

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace dorado::utils::gpu_monitor {

namespace {

#if HAS_NVML
#ifdef NVML_DEVICE_NAME_V2_BUFFER_SIZE
#define ONT_NVML_BUFFER_SIZE NVML_DEVICE_NAME_V2_BUFFER_SIZE
#elif defined(NVML_DEVICE_NAME_BUFFER_SIZE)
#define ONT_NVML_BUFFER_SIZE NVML_DEVICE_NAME_BUFFER_SIZE
#endif
static_assert(ONT_NVML_BUFFER_SIZE, "nvml buffer size must be defined");

// Prefixless versions of symbols we use
// X(name, optional)
#define FOR_EACH_NVML_SYMBOL(X)                     \
    X(DeviceGetCount, false)                        \
    X(DeviceGetCount_v2, true)                      \
    X(DeviceGetCurrentClocksThrottleReasons, false) \
    X(DeviceGetHandleByIndex, false)                \
    X(DeviceGetHandleByIndex_v2, true)              \
    X(DeviceGetHandleByUUID, false)                 \
    X(DeviceGetIndex, false)                        \
    X(DeviceGetName, false)                         \
    X(DeviceGetPerformanceState, false)             \
    X(DeviceGetPowerManagementDefaultLimit, false)  \
    X(DeviceGetPowerUsage, false)                   \
    X(DeviceGetTemperature, false)                  \
    X(DeviceGetTemperatureThreshold, false)         \
    X(DeviceGetUtilizationRates, false)             \
    X(Init, false)                                  \
    X(Init_v2, true)                                \
    X(Shutdown, false)                              \
    X(SystemGetDriverVersion, false)                \
    X(ErrorString, false)                           \
    // line intentionally blank

/**
 * Handle to the NVML API.
 * Also provides a scoped wrapper around NVML API initialisation.
 */
class NvmlApi final {
    // Platform specific library handling.
#ifdef _WIN32
    HINSTANCE m_handle = nullptr;
    bool platform_open() {
        m_handle = LoadLibraryA("nvml.dll");
        if (m_handle != nullptr) {
            return true;
        }

        // Search in other places that the documentation and other resources mentions.
        const char *win64_dir_env = getenv("ProgramW6432");
        const std::string win64_dir = win64_dir_env ? win64_dir_env : "C:";
        const std::string paths[] = {
                win64_dir + "\\NVIDIA Corporation\\NVSMI\\nvml.dll",
                win64_dir + "\\NVIDIA Corporation\\NVSMI\\nvml\\lib\\nvml.dll",
                win64_dir + "\\NVIDIA Corporation\\GDK\\nvml.dll",
                win64_dir + "\\NVIDIA Corporation\\GDK\\nvml\\lib\\nvml.dll",
        };
        for (const auto &path : paths) {
            m_handle = LoadLibraryA(path.c_str());
            if (m_handle != nullptr) {
                return true;
            }
        }
        return false;
    }
    void platform_close() {
        if (m_handle != nullptr) {
            FreeLibrary(m_handle);
            m_handle = nullptr;
        }
    }
    template <typename T>
    bool platform_load_symbol(T *&func_ptr, const char *name, bool optional) {
        func_ptr = reinterpret_cast<T *>(GetProcAddress(m_handle, name));
        if (func_ptr == nullptr && !optional) {
            spdlog::warn("Failed to load NVML symbol {}: {}", name, GetLastError());
            return false;
        }
        return true;
    }

#else   // _WIN32
    void *m_handle = nullptr;
    bool platform_open() {
        // Prioritise loading the versioned lib.
        for (const char *path : {"libnvidia-ml.so.1", "libnvidia-ml.so"}) {
            m_handle = dlopen(path, RTLD_NOW);
            if (m_handle != nullptr) {
                return true;
            }
        }
        return false;
    }
    void platform_close() {
        if (m_handle != nullptr) {
            dlclose(m_handle);
            m_handle = nullptr;
        }
    }
    template <typename T>
    bool platform_load_symbol(T *&func_ptr, const char *name, bool optional) {
        func_ptr = reinterpret_cast<T *>(dlsym(m_handle, name));
        if (func_ptr == nullptr && !optional) {
            spdlog::warn("Failed to load NVML symbol {}: {}", name, dlerror());
            return false;
        }
        return true;
    }
#endif  // _WIN32

    // Add members for each function pointer.
#define GENERATE_MEMBER(name, optional)         \
    using name##_ptr = decltype(&::nvml##name); \
    name##_ptr m_##name = nullptr;
    FOR_EACH_NVML_SYMBOL(GENERATE_MEMBER)
#undef GENERATE_MEMBER

    bool load_symbols() {
#define LOAD_SYMBOL(name, optional)                                \
    if (!platform_load_symbol(m_##name, "nvml" #name, optional)) { \
        return false;                                              \
    }
        FOR_EACH_NVML_SYMBOL(LOAD_SYMBOL)
#undef LOAD_SYMBOL
        return true;
    }

    void clear_symbols() {
#define CLEAR_SYMBOL(name, optional) m_##name = nullptr;
        FOR_EACH_NVML_SYMBOL(CLEAR_SYMBOL)
#undef CLEAR_SYMBOL
    }

    void init() {
        if (!platform_open() || !load_symbols()) {
            spdlog::info("Failed to load NVML");
            clear_symbols();
            platform_close();
            return;
        }

        // Fall back to the original nvmlInit() if _v2 doesn't exist.
        auto *do_init = m_Init_v2 ? m_Init_v2 : m_Init;

        // We retry initialisation for a certain amount of time, to allow the driver to load on system startup
        auto start = std::chrono::system_clock::now();
        auto wait_seconds = std::chrono::seconds(10);
        nvmlReturn_t result;
        do {
            result = do_init();
            if (result == NVML_SUCCESS) {
                break;
            }
            spdlog::warn("Failed to initialize NVML: {}, retrying in 1s...", m_ErrorString(result));
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } while ((std::chrono::system_clock::now() - start) < wait_seconds);

        if (result != NVML_SUCCESS) {
            spdlog::warn("Failed to initialize NVML after {} seconds: {}", wait_seconds.count(),
                         m_ErrorString(result));
            clear_symbols();
            platform_close();
        }
    }

    void shutdown() {
        if (m_Shutdown != nullptr) {
            m_Shutdown();
        }
        clear_symbols();
        platform_close();
    }

    NvmlApi(const NvmlApi &) = delete;
    NvmlApi &operator=(const NvmlApi &) = delete;

    // This slight design flaw is in place of having NvmlApi as a singleton.
    // Instead it is held as a member variable of the DeviceInfoCache singleton.
    // This is preferable to having dependencies between singletons.
    friend class DeviceInfoCache;
    NvmlApi() { init(); }

    ~NvmlApi() { shutdown(); }

public:
    bool is_loaded() { return m_handle != nullptr; }

    std::optional<nvmlDevice_t> get_device_handle(unsigned int device_index) {
        nvmlDevice_t device;
        auto *get_handle = m_DeviceGetHandleByIndex_v2 ? m_DeviceGetHandleByIndex_v2
                                                       : m_DeviceGetHandleByIndex;
        ScopedTraceLog log{__func__};
        nvmlReturn_t result = get_handle(device_index, &device);
        if (result != NVML_SUCCESS) {
            return std::nullopt;
        }
        return device;
    }

    nvmlReturn_t SystemGetDriverVersion(char *version, unsigned int length) {
        ScopedTraceLog log{__func__};
        return m_SystemGetDriverVersion(version, length);
    }

    nvmlReturn_t DeviceGetTemperature(const nvmlDevice_t &device,
                                      nvmlTemperatureSensors_t sensorType,
                                      unsigned int *temp) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetTemperature(device, sensorType, temp);
    }

    nvmlReturn_t DeviceGetTemperatureThreshold(const nvmlDevice_t &device,
                                               nvmlTemperatureThresholds_t thresholdType,
                                               unsigned int *temp) {
        ScopedTraceLog log{__func__};
        log.write("nvmlTemperatureThresholds_t: " + std::to_string(thresholdType));
        return m_DeviceGetTemperatureThreshold(device, thresholdType, temp);
    }

    nvmlReturn_t DeviceGetPerformanceState(const nvmlDevice_t &device, unsigned int *limit) {
        ScopedTraceLog log{__func__};
        nvmlPstates_t state;
        auto result = m_DeviceGetPerformanceState(device, &state);
        *limit = static_cast<unsigned int>(state);
        return result;
    }

    nvmlReturn_t DeviceGetPowerManagementDefaultLimit(const nvmlDevice_t &device,
                                                      unsigned int *limit) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetPowerManagementDefaultLimit(device, limit);
    }

    nvmlReturn_t DeviceGetPowerUsage(const nvmlDevice_t &device, unsigned int *power) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetPowerUsage(device, power);
    }

    nvmlReturn_t DeviceGetUtilizationRates(const nvmlDevice_t &device,
                                           nvmlUtilization_t *utilization) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetUtilizationRates(device, utilization);
    }

    nvmlReturn_t DeviceGetCurrentClocksThrottleReasons(const nvmlDevice_t &device,
                                                       unsigned long long *clocksThrottleReasons) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetCurrentClocksThrottleReasons(device, clocksThrottleReasons);
    }

    nvmlReturn_t DeviceGetName(const nvmlDevice_t &device, char *name, unsigned int length) {
        ScopedTraceLog log{__func__};
        return m_DeviceGetName(device, name, length);
    }

    nvmlReturn_t DeviceGetCount(unsigned int *count) {
        auto *device_count_op = m_DeviceGetCount_v2 ? m_DeviceGetCount_v2 : m_DeviceGetCount;
        ScopedTraceLog log{__func__};
        return device_count_op(count);
    }

    const char *ErrorString(nvmlReturn_t result) { return m_ErrorString(result); }
};

void assign_threshold_temp(NvmlApi *nvml,
                           const nvmlDevice_t &device,
                           nvmlTemperatureThresholds_t thresholdType,
                           std::optional<unsigned int> &temp,
                           std::string &error_reason) {
    unsigned int value{};
    auto result = nvml->DeviceGetTemperatureThreshold(device, thresholdType, &value);
    if (result == NVML_SUCCESS) {
        temp = value;
    } else {
        error_reason = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_threshold_temperatures(NvmlApi *nvml,
                                                const nvmlDevice_t &device,
                                                DeviceStatusInfo &info) {
    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
                          info.gpu_shutdown_temperature, info.gpu_shutdown_temperature_error);
    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
                          info.gpu_slowdown_temperature, info.gpu_slowdown_temperature_error);
    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX,
                          info.gpu_max_operating_temperature,
                          info.gpu_max_operating_temperature_error);
}

void retrieve_and_assign_current_power_usage(NvmlApi *nvml,
                                             nvmlDevice_t &device,
                                             DeviceStatusInfo &info) {
    unsigned int value{};
    auto result = nvml->DeviceGetPowerUsage(device, &value);
    if (result == NVML_SUCCESS) {
        info.current_power_usage = value;
    } else {
        info.current_power_usage_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_power_cap(NvmlApi *nvml,
                                   const nvmlDevice_t &device,
                                   DeviceStatusInfo &info) {
    unsigned int value{};
    auto result = nvml->DeviceGetPowerManagementDefaultLimit(device, &value);
    if (result == NVML_SUCCESS) {
        info.default_power_cap = value;
    } else {
        info.default_power_cap_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_utilization(NvmlApi *nvml,
                                     const nvmlDevice_t &device,
                                     DeviceStatusInfo &info) {
    nvmlUtilization_t utilization{};
    auto result = nvml->DeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        info.percentage_utilization_error = nvml->ErrorString(result);
        return;
    }
    info.percentage_utilization_gpu = utilization.gpu;
    info.percentage_utilization_memory = utilization.memory;
}

void retrieve_and_assign_current_performance(NvmlApi *nvml,
                                             const nvmlDevice_t &device,
                                             DeviceStatusInfo &info) {
    unsigned int value;
    auto result = nvml->DeviceGetPerformanceState(device, &value);
    if (result == NVML_SUCCESS) {
        info.current_performance_state = value;
    } else {
        info.current_performance_state_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_current_throttling_reason(NvmlApi *nvml,
                                                   const nvmlDevice_t &device,
                                                   DeviceStatusInfo &info) {
    unsigned long long reason{};
    auto result = nvml->DeviceGetCurrentClocksThrottleReasons(device, &reason);
    if (result == NVML_SUCCESS) {
        info.current_throttling_reason = reason;
    } else {
        info.current_throttling_reason_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_current_temperature(NvmlApi *nvml,
                                             const nvmlDevice_t &device,
                                             DeviceStatusInfo &info) {
    unsigned int value{};
    auto result = nvml->DeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &value);
    if (result == NVML_SUCCESS) {
        info.current_temperature = value;
    } else {
        info.current_temperature_error = nvml->ErrorString(result);
    }
}

void retrieve_and_assign_device_name(NvmlApi *nvml,
                                     const nvmlDevice_t &device,
                                     DeviceStatusInfo &info) {
    char device_name[ONT_NVML_BUFFER_SIZE];
    auto result = nvml->DeviceGetName(device, device_name, ONT_NVML_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        info.device_name = device_name;
    } else {
        info.device_name_error = nvml->ErrorString(result);
    }
}

class DeviceHandles final {
    NvmlApi &m_nvml;
    std::unordered_map<unsigned int, std::optional<nvmlDevice_t>> m_device_handles{};

public:
    DeviceHandles(NvmlApi &nvml) : m_nvml(nvml) { assert(m_nvml.is_loaded()); }

    std::optional<nvmlDevice_t> get_handle(unsigned int device_index) {
        auto device_handle_lookup = m_device_handles.find(device_index);
        if (device_handle_lookup != m_device_handles.end()) {
            return device_handle_lookup->second;
        }

        auto device = m_nvml.get_device_handle(device_index);
        m_device_handles[device_index] = device;
        return device;
    }
};

class DeviceInfoCache final {
    std::mutex m_mutex{};
    NvmlApi m_nvml{};
    std::unique_ptr<DeviceHandles> m_device_handles;
    std::unordered_map<nvmlDevice_t, std::optional<DeviceStatusInfo>> m_device_info{};
    // Includes devices which cannot be accessed via NVML so need to check return codes
    // on individual device specific NVML function calls.
    unsigned int m_device_count = 0;
    std::vector<unsigned int> m_visible_device_indices;

    DeviceInfoCache() {
        set_device_count();
        if (m_nvml.is_loaded()) {
            m_device_handles = std::make_unique<DeviceHandles>(m_nvml);
        }
    }

    void map_visible_devices(unsigned int device_count) {
        // NVML doesn't respect CUDA_VISIBLE_DEVICES envvar, so check this separately
        const char *cuda_visible_devices_env = std::getenv("CUDA_VISIBLE_DEVICES");
        if (cuda_visible_devices_env != nullptr) {
            spdlog::debug("Found CUDA_VISIBLE_DEVICES={}", cuda_visible_devices_env);
            std::set<int> used_ids;
            auto device_ids = utils::split(cuda_visible_devices_env, ',');
            if (device_ids.size() > device_count) {
                spdlog::error(
                        "CUDA_VISIBLE_DEVICES={} specifies more device ids than the number of GPUs "
                        "present",
                        cuda_visible_devices_env);
                throw std::runtime_error("Invalid device ids");
            }

            if (!device_ids.empty() && (utils::starts_with(device_ids.front(), "GPU-") ||
                                        utils::starts_with(device_ids.front(), "MIG-"))) {
                for (const auto &id : device_ids) {
                    nvmlDevice_t device;
                    if (auto rc = m_nvml.m_DeviceGetHandleByUUID(id.c_str(), &device);
                        rc != NVML_SUCCESS) {
                        spdlog::warn(
                                "Unable to identify GPU device '{}' - skipping further device "
                                "enumeration",
                                id);
                        // stop parsing on error
                        break;
                    }

                    unsigned int index = 0;
                    if (auto rc = m_nvml.m_DeviceGetIndex(device, &index); rc != NVML_SUCCESS) {
                        spdlog::warn(
                                "Unable to retrieve index for GPU device '{}' - skipping further "
                                "device enumeration",
                                id);
                        // stop parsing on error
                        break;
                    };
                    used_ids.insert(index);
                    m_visible_device_indices.push_back(index);
                }
            } else {
                std::string_view last_device_id;
                try {
                    for (const auto &id : device_ids) {
                        last_device_id = id;
                        int index = std::stoi(id);
                        if (index < 0 || index >= static_cast<int>(device_count)) {
                            spdlog::warn(
                                    "Invalid index '{}' for GPU device - skipping further device "
                                    "enumeration",
                                    index);
                            // stop parsing on invalid id
                            break;
                        }

                        used_ids.insert(index);
                        m_visible_device_indices.push_back(index);
                    }
                } catch (const std::exception &) {
                    // stop parsing on error
                    spdlog::warn(
                            "Unable to parse id '{}' for GPU device - skipping further device "
                            "enumeration",
                            last_device_id);
                }
            }
            if (used_ids.size() != m_visible_device_indices.size()) {
                // passing the same id twice should return no devices
                spdlog::warn("Duplicate GPU ids detected - no GPUs identified");
                m_visible_device_indices.clear();
            }
        } else {
            m_visible_device_indices.resize(device_count);
            std::iota(std::begin(m_visible_device_indices), std::end(m_visible_device_indices), 0);
        }
    }

    void set_device_count() {
        unsigned int device_count = 0;
        if (m_nvml.is_loaded()) {
            auto result = m_nvml.DeviceGetCount(&device_count);
            if (result != NVML_SUCCESS) {
                device_count = 0;
                spdlog::warn("Call to DeviceGetCount failed: {}", m_nvml.ErrorString(result));
            }
        }
#if DORADO_ORIN
        if (device_count == 0) {
            // Orin may not have NVML, in which case ask torch how many devices it thinks there are.
            device_count = torch::cuda::device_count();
            spdlog::info("Setting device count to {} as reported from torch", device_count);
        }
#endif
        map_visible_devices(device_count);
        unsigned int cuda_visible_devices_count =
                static_cast<unsigned int>(m_visible_device_indices.size());

        if (cuda_visible_devices_count > device_count) {
            spdlog::warn(
                    "CUDA_VISIBLE_DEVICES contains more device ids ({}) than devices found by NVML "
                    "({}).",
                    cuda_visible_devices_count, device_count);
        }
        m_device_count = std::min(cuda_visible_devices_count, device_count);
    }

    std::optional<DeviceStatusInfo> create_new_device_entry(unsigned int device_index,
                                                            nvmlDevice_t device) {
        auto &info = m_device_info.emplace(device, DeviceStatusInfo{}).first->second;
        info->device_index = device_index;
        retrieve_and_assign_threshold_temperatures(&m_nvml, device, *info);
        retrieve_and_assign_power_cap(&m_nvml, device, *info);
        retrieve_and_assign_device_name(&m_nvml, device, *info);
        return info;
    }

    std::pair<std::optional<DeviceStatusInfo>, nvmlDevice_t> get_cached_device_info(
            unsigned int device_index) {
        std::lock_guard<std::mutex> lock(m_mutex);
        const unsigned int mapped_device_index = m_visible_device_indices[device_index];
        auto device = m_device_handles->get_handle(mapped_device_index);
        if (!device) {
            return {std::nullopt, nullptr};
        }

        auto device_info_lookup = m_device_info.find(*device);
        if (device_info_lookup != m_device_info.end()) {
            return {device_info_lookup->second, *device};
        }

        return {create_new_device_entry(mapped_device_index, *device), *device};
    }

public:
    static DeviceInfoCache &instance() {
        static DeviceInfoCache cache;
        return cache;
    }

    std::optional<nvmlDevice_t> get_device_handle(unsigned int device_index) {
        if (!m_nvml.is_loaded()) {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_device_handles->get_handle(m_visible_device_indices[device_index]);
    }

    NvmlApi &nvml() { return m_nvml; }

    unsigned int get_device_count() { return m_device_count; }

    std::optional<DeviceStatusInfo> get_device_info(unsigned int device_index) {
        if (!m_nvml.is_loaded()) {
            return std::nullopt;
        }
        auto [info, device] = get_cached_device_info(device_index);
        if (!info) {
            return std::nullopt;
        }

        // We have a copy of the cached DeviceStatusInfo so we can update without
        // locking. NVML itelf is thread safe.
        retrieve_and_assign_current_temperature(&m_nvml, device, *info);
        retrieve_and_assign_current_power_usage(&m_nvml, device, *info);
        retrieve_and_assign_utilization(&m_nvml, device, *info);
        retrieve_and_assign_current_performance(&m_nvml, device, *info);
        retrieve_and_assign_current_throttling_reason(&m_nvml, device, *info);

        return info;
    }
};

std::optional<std::string> read_version_from_nvml() {
    auto &nvml_api = DeviceInfoCache::instance().nvml();
    if (!nvml_api.is_loaded()) {
        return std::nullopt;
    }

    // Grab the driver version
    char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE + 1]{};
    nvmlReturn_t result =
            nvml_api.SystemGetDriverVersion(version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        return version;
    } else {
        spdlog::warn("Failed to query driver version: {}", nvml_api.ErrorString(result));
        return std::nullopt;
    }
}

#endif  // HAS_NVML

#if HAS_RSMI
// Minimal ROCm SMI type definitions for dynamic loading.
// These match the stable ABI of librocm_smi64 and avoid a compile-time
// dependency on rocm_smi headers.
using rsmi_status_t = int;
constexpr rsmi_status_t RSMI_STATUS_SUCCESS = 0;

// Temperature metrics (rsmi_temperature_metric_t stable ABI values)
constexpr uint32_t RSMI_TEMP_CURRENT   = 0;  // Current temperature
constexpr uint32_t RSMI_TEMP_MAX       = 6;  // Max operating temperature
constexpr uint32_t RSMI_TEMP_CRIT      = 7;  // Critical (slowdown) threshold
constexpr uint32_t RSMI_TEMP_EMERGENCY = 9;  // Emergency (shutdown) threshold

// Temperature sensor type (rsmi_temperature_type_t stable ABI values)
constexpr uint32_t RSMI_TEMP_TYPE_EDGE = 0;  // Edge/GPU die temperature sensor

// Buffer sizes
constexpr size_t RSMI_NAME_BUFFER_SIZE           = 256;
constexpr size_t RSMI_DRIVER_VERSION_BUFFER_SIZE = 256;

// Function pointer type aliases (matching ROCm SMI stable ABI)
using rsmi_init_fn                        = rsmi_status_t (*)(uint64_t);
using rsmi_shut_down_fn                   = rsmi_status_t (*)();
using rsmi_num_monitor_devs_fn            = rsmi_status_t (*)(uint32_t *);
using rsmi_dev_name_get_fn                = rsmi_status_t (*)(uint32_t, char *, size_t);
using rsmi_dev_temp_metric_get_fn         = rsmi_status_t (*)(uint32_t, uint32_t, uint32_t, int64_t *);
using rsmi_dev_power_ave_get_fn           = rsmi_status_t (*)(uint32_t, uint32_t, uint64_t *);
using rsmi_dev_power_cap_default_get_fn   = rsmi_status_t (*)(uint32_t, uint64_t *);
using rsmi_dev_busy_percent_get_fn        = rsmi_status_t (*)(uint32_t, uint32_t *);
using rsmi_dev_memory_busy_percent_get_fn = rsmi_status_t (*)(uint32_t, uint32_t *);
using rsmi_dev_perf_level_get_fn          = rsmi_status_t (*)(uint32_t, uint32_t *);
using rsmi_status_string_fn               = rsmi_status_t (*)(rsmi_status_t, const char **);
using rsmi_driver_version_str_get_fn      = rsmi_status_t (*)(char *, size_t);

/**
 * Handle to the ROCm SMI API.
 * Dynamically loads librocm_smi64.so and manages library lifetime.
 */
class RsmiApi final {
    void *m_handle = nullptr;

    bool platform_open() {
        for (const char *path : {"librocm_smi64.so.5", "librocm_smi64.so"}) {
            m_handle = dlopen(path, RTLD_NOW);
            if (m_handle != nullptr) {
                return true;
            }
        }
        return false;
    }

    void platform_close() {
        if (m_handle != nullptr) {
            dlclose(m_handle);
            m_handle = nullptr;
        }
    }

    template <typename T>
    bool load_symbol(T *&func_ptr, const char *name, bool optional) {
        func_ptr = reinterpret_cast<T *>(dlsym(m_handle, name));
        if (func_ptr == nullptr && !optional) {
            spdlog::warn("Failed to load ROCm SMI symbol {}: {}", name, dlerror());
            return false;
        }
        return true;
    }

    // Required function pointers
    rsmi_init_fn             m_init             = nullptr;
    rsmi_shut_down_fn        m_shut_down        = nullptr;
    rsmi_num_monitor_devs_fn m_num_monitor_devs = nullptr;
    rsmi_dev_name_get_fn     m_dev_name_get     = nullptr;
    rsmi_status_string_fn    m_status_string    = nullptr;

    // Optional function pointers (availability depends on ROCm version)
    rsmi_dev_temp_metric_get_fn         m_dev_temp_metric_get         = nullptr;
    rsmi_dev_power_ave_get_fn           m_dev_power_ave_get           = nullptr;
    rsmi_dev_power_cap_default_get_fn   m_dev_power_cap_default_get   = nullptr;
    rsmi_dev_busy_percent_get_fn        m_dev_busy_percent_get        = nullptr;
    rsmi_dev_memory_busy_percent_get_fn m_dev_memory_busy_percent_get = nullptr;
    rsmi_dev_perf_level_get_fn          m_dev_perf_level_get          = nullptr;
    rsmi_driver_version_str_get_fn      m_driver_version_str_get      = nullptr;

    bool load_symbols() {
        if (!load_symbol(m_init,             "rsmi_init",             false)) return false;
        if (!load_symbol(m_shut_down,        "rsmi_shut_down",        false)) return false;
        if (!load_symbol(m_dev_name_get,     "rsmi_dev_name_get",     false)) return false;
        if (!load_symbol(m_status_string,    "rsmi_status_string",    false)) return false;
        load_symbol(m_num_monitor_devs, "rsmi_num_monitor_devs", true);
        load_symbol(m_dev_temp_metric_get,         "rsmi_dev_temp_metric_get",         true);
        load_symbol(m_dev_power_ave_get,           "rsmi_dev_power_ave_get",           true);
        load_symbol(m_dev_power_cap_default_get,   "rsmi_dev_power_cap_default_get",   true);
        load_symbol(m_dev_busy_percent_get,        "rsmi_dev_busy_percent_get",        true);
        load_symbol(m_dev_memory_busy_percent_get, "rsmi_dev_memory_busy_percent_get", true);
        load_symbol(m_dev_perf_level_get,          "rsmi_dev_perf_level_get",          true);
        load_symbol(m_driver_version_str_get,      "rsmi_driver_version_str_get",      true);
        return true;
    }

    void clear_symbols() {
        m_init                        = nullptr;
        m_shut_down                   = nullptr;
        m_num_monitor_devs            = nullptr;
        m_dev_name_get                = nullptr;
        m_status_string               = nullptr;
        m_dev_temp_metric_get         = nullptr;
        m_dev_power_ave_get           = nullptr;
        m_dev_power_cap_default_get   = nullptr;
        m_dev_busy_percent_get        = nullptr;
        m_dev_memory_busy_percent_get = nullptr;
        m_dev_perf_level_get          = nullptr;
        m_driver_version_str_get      = nullptr;
    }

    void init() {
        if (!platform_open() || !load_symbols()) {
            spdlog::info("Failed to load ROCm SMI library");
            clear_symbols();
            platform_close();
            return;
        }
        auto result = m_init(0);
        if (result != RSMI_STATUS_SUCCESS) {
            spdlog::warn("Failed to initialize ROCm SMI: error {}", result);
            clear_symbols();
            platform_close();
        }
    }

    void shutdown() {
        if (m_shut_down != nullptr) {
            m_shut_down();
        }
        clear_symbols();
        platform_close();
    }

    RsmiApi(const RsmiApi &) = delete;
    RsmiApi &operator=(const RsmiApi &) = delete;

    friend class RsmiDeviceInfoCache;
    RsmiApi() { init(); }
    ~RsmiApi() { shutdown(); }

public:
    bool is_loaded() const { return m_handle != nullptr; }

    uint32_t DeviceGetCount() {
        uint32_t count = 0;
        if (m_num_monitor_devs) {
            if (m_num_monitor_devs(&count) == RSMI_STATUS_SUCCESS) {
                return count;
            }
            count = 0;
        }
        // Fallback: probe by index until rsmi_dev_name_get fails.
        constexpr uint32_t MAX_DEVICES = 64;
        char name[RSMI_NAME_BUFFER_SIZE];
        while (count < MAX_DEVICES &&
               m_dev_name_get(count, name, RSMI_NAME_BUFFER_SIZE) == RSMI_STATUS_SUCCESS) {
            ++count;
        }
        return count;
    }

    rsmi_status_t DeviceGetName(uint32_t dv_ind, char *name, size_t len) {
        ScopedTraceLog log{__func__};
        return m_dev_name_get(dv_ind, name, len);
    }

    // Returns temperature in degrees Celsius (RSMI reports in millidegrees).
    rsmi_status_t DeviceGetTemperature(uint32_t dv_ind, uint32_t metric, unsigned int *temp_c) {
        ScopedTraceLog log{__func__};
        if (!m_dev_temp_metric_get) {
            return -1;
        }
        int64_t temp_mdeg = 0;
        auto result = m_dev_temp_metric_get(dv_ind, RSMI_TEMP_TYPE_EDGE, metric, &temp_mdeg);
        if (result == RSMI_STATUS_SUCCESS) {
            *temp_c = static_cast<unsigned int>(temp_mdeg / 1000);
        }
        return result;
    }

    // Returns power in milliwatts (RSMI reports in microwatts).
    rsmi_status_t DeviceGetPowerUsage(uint32_t dv_ind, unsigned int *power_mw) {
        ScopedTraceLog log{__func__};
        if (!m_dev_power_ave_get) {
            return -1;
        }
        uint64_t power_uw = 0;
        auto result = m_dev_power_ave_get(dv_ind, 0, &power_uw);
        if (result == RSMI_STATUS_SUCCESS) {
            *power_mw = static_cast<unsigned int>(power_uw / 1000);
        }
        return result;
    }

    // Returns default power cap in milliwatts (RSMI reports in microwatts).
    rsmi_status_t DeviceGetPowerCapDefault(uint32_t dv_ind, unsigned int *cap_mw) {
        ScopedTraceLog log{__func__};
        if (!m_dev_power_cap_default_get) {
            return -1;
        }
        uint64_t cap_uw = 0;
        auto result = m_dev_power_cap_default_get(dv_ind, &cap_uw);
        if (result == RSMI_STATUS_SUCCESS) {
            *cap_mw = static_cast<unsigned int>(cap_uw / 1000);
        }
        return result;
    }

    rsmi_status_t DeviceGetBusyPercent(uint32_t dv_ind, uint32_t *busy_percent) {
        ScopedTraceLog log{__func__};
        if (!m_dev_busy_percent_get) {
            return -1;
        }
        return m_dev_busy_percent_get(dv_ind, busy_percent);
    }

    rsmi_status_t DeviceGetMemoryBusyPercent(uint32_t dv_ind, uint32_t *busy_percent) {
        ScopedTraceLog log{__func__};
        if (!m_dev_memory_busy_percent_get) {
            return -1;
        }
        return m_dev_memory_busy_percent_get(dv_ind, busy_percent);
    }

    rsmi_status_t DeviceGetPerfLevel(uint32_t dv_ind, uint32_t *perf_level) {
        ScopedTraceLog log{__func__};
        if (!m_dev_perf_level_get) {
            return -1;
        }
        return m_dev_perf_level_get(dv_ind, perf_level);
    }

    std::optional<std::string> GetDriverVersion() {
        if (!m_driver_version_str_get) {
            return std::nullopt;
        }
        char version[RSMI_DRIVER_VERSION_BUFFER_SIZE] = {};
        auto result = m_driver_version_str_get(version, RSMI_DRIVER_VERSION_BUFFER_SIZE);
        if (result == RSMI_STATUS_SUCCESS && version[0] != '\0') {
            return std::string(version);
        }
        return std::nullopt;
    }

    const char *ErrorString(rsmi_status_t result) {
        const char *str = nullptr;
        if (m_status_string && m_status_string(result, &str) == RSMI_STATUS_SUCCESS && str) {
            return str;
        }
        return "unknown ROCm SMI error";
    }
};

void retrieve_and_assign_rsmi_current_temperature(RsmiApi *rsmi, uint32_t dv_ind,
                                                   DeviceStatusInfo &info) {
    unsigned int temp = 0;
    auto result = rsmi->DeviceGetTemperature(dv_ind, RSMI_TEMP_CURRENT, &temp);
    if (result == RSMI_STATUS_SUCCESS) {
        info.current_temperature = temp;
    } else {
        info.current_temperature_error = rsmi->ErrorString(result);
    }
}

void retrieve_and_assign_rsmi_threshold_temperatures(RsmiApi *rsmi, uint32_t dv_ind,
                                                      DeviceStatusInfo &info) {
    // Emergency threshold maps to NVML's SHUTDOWN threshold
    {
        unsigned int temp = 0;
        auto result = rsmi->DeviceGetTemperature(dv_ind, RSMI_TEMP_EMERGENCY, &temp);
        if (result == RSMI_STATUS_SUCCESS) {
            info.gpu_shutdown_temperature = temp;
        } else {
            info.gpu_shutdown_temperature_error = rsmi->ErrorString(result);
        }
    }
    // Critical threshold maps to NVML's SLOWDOWN threshold
    {
        unsigned int temp = 0;
        auto result = rsmi->DeviceGetTemperature(dv_ind, RSMI_TEMP_CRIT, &temp);
        if (result == RSMI_STATUS_SUCCESS) {
            info.gpu_slowdown_temperature = temp;
        } else {
            info.gpu_slowdown_temperature_error = rsmi->ErrorString(result);
        }
    }
    // Max operating threshold maps to NVML's GPU_MAX threshold
    {
        unsigned int temp = 0;
        auto result = rsmi->DeviceGetTemperature(dv_ind, RSMI_TEMP_MAX, &temp);
        if (result == RSMI_STATUS_SUCCESS) {
            info.gpu_max_operating_temperature = temp;
        } else {
            info.gpu_max_operating_temperature_error = rsmi->ErrorString(result);
        }
    }
}

void retrieve_and_assign_rsmi_power_usage(RsmiApi *rsmi, uint32_t dv_ind,
                                           DeviceStatusInfo &info) {
    unsigned int power_mw = 0;
    auto result = rsmi->DeviceGetPowerUsage(dv_ind, &power_mw);
    if (result == RSMI_STATUS_SUCCESS) {
        info.current_power_usage = power_mw;
    } else {
        info.current_power_usage_error = rsmi->ErrorString(result);
    }
}

void retrieve_and_assign_rsmi_power_cap(RsmiApi *rsmi, uint32_t dv_ind, DeviceStatusInfo &info) {
    unsigned int cap_mw = 0;
    auto result = rsmi->DeviceGetPowerCapDefault(dv_ind, &cap_mw);
    if (result == RSMI_STATUS_SUCCESS) {
        info.default_power_cap = cap_mw;
    } else {
        info.default_power_cap_error = rsmi->ErrorString(result);
    }
}

void retrieve_and_assign_rsmi_utilization(RsmiApi *rsmi, uint32_t dv_ind, DeviceStatusInfo &info) {
    uint32_t busy_percent = 0;
    auto result = rsmi->DeviceGetBusyPercent(dv_ind, &busy_percent);
    if (result != RSMI_STATUS_SUCCESS) {
        info.percentage_utilization_error = rsmi->ErrorString(result);
        return;
    }
    info.percentage_utilization_gpu = busy_percent;
    uint32_t mem_busy_percent = 0;
    result = rsmi->DeviceGetMemoryBusyPercent(dv_ind, &mem_busy_percent);
    if (result == RSMI_STATUS_SUCCESS) {
        info.percentage_utilization_memory = mem_busy_percent;
    }
    // A failure to retrieve memory utilization is non-critical; GPU utilization was retrieved.
}

void retrieve_and_assign_rsmi_perf_level(RsmiApi *rsmi, uint32_t dv_ind, DeviceStatusInfo &info) {
    uint32_t perf_level = 0;
    auto result = rsmi->DeviceGetPerfLevel(dv_ind, &perf_level);
    if (result == RSMI_STATUS_SUCCESS) {
        info.current_performance_state = perf_level;
    } else {
        info.current_performance_state_error = rsmi->ErrorString(result);
    }
}

class RsmiDeviceInfoCache final {
    std::mutex m_mutex{};
    RsmiApi m_rsmi{};
    std::unordered_map<uint32_t, std::optional<DeviceStatusInfo>> m_static_device_info{};
    unsigned int m_device_count = 0;
    std::vector<uint32_t> m_visible_device_indices;

    RsmiDeviceInfoCache() { set_device_count(); }

    void map_visible_devices(uint32_t device_count) {
        // ROCm uses HIP_VISIBLE_DEVICES (or ROCR_VISIBLE_DEVICES) similarly to CUDA_VISIBLE_DEVICES.
        const char *hip_visible_devices_env = std::getenv("HIP_VISIBLE_DEVICES");
        if (hip_visible_devices_env == nullptr) {
            hip_visible_devices_env = std::getenv("ROCR_VISIBLE_DEVICES");
        }
        if (hip_visible_devices_env != nullptr) {
            spdlog::debug("Found HIP_VISIBLE_DEVICES={}", hip_visible_devices_env);
            std::set<int> used_ids;
            auto device_ids = utils::split(hip_visible_devices_env, ',');
            if (device_ids.size() > device_count) {
                spdlog::error(
                        "HIP_VISIBLE_DEVICES={} specifies more device ids than the number of "
                        "GPUs present",
                        hip_visible_devices_env);
                throw std::runtime_error("Invalid device ids");
            }
            std::string_view last_device_id;
            try {
                for (const auto &id : device_ids) {
                    last_device_id = id;
                    int index = std::stoi(id);
                    if (index < 0 || index >= static_cast<int>(device_count)) {
                        spdlog::warn(
                                "Invalid index '{}' for GPU device - skipping further device "
                                "enumeration",
                                index);
                        break;
                    }
                    used_ids.insert(index);
                    m_visible_device_indices.push_back(static_cast<uint32_t>(index));
                }
            } catch (const std::exception &) {
                spdlog::warn(
                        "Unable to parse id '{}' for GPU device - skipping further device "
                        "enumeration",
                        last_device_id);
            }
            if (used_ids.size() != m_visible_device_indices.size()) {
                spdlog::warn("Duplicate GPU ids detected - no GPUs identified");
                m_visible_device_indices.clear();
            }
        } else {
            m_visible_device_indices.resize(device_count);
            std::iota(std::begin(m_visible_device_indices), std::end(m_visible_device_indices),
                      static_cast<uint32_t>(0));
        }
    }

    void set_device_count() {
        uint32_t device_count = 0;
        if (m_rsmi.is_loaded()) {
            device_count = m_rsmi.DeviceGetCount();
        }
        map_visible_devices(device_count);
        auto visible_count = static_cast<unsigned int>(m_visible_device_indices.size());
        if (visible_count > static_cast<unsigned int>(device_count)) {
            spdlog::warn(
                    "HIP_VISIBLE_DEVICES contains more device ids ({}) than devices found by "
                    "ROCm SMI ({}).",
                    visible_count, device_count);
        }
        m_device_count = std::min(visible_count, static_cast<unsigned int>(device_count));
    }

    std::optional<DeviceStatusInfo> create_static_device_entry(unsigned int device_index,
                                                                uint32_t dv_ind) {
        auto &info = m_static_device_info.emplace(dv_ind, DeviceStatusInfo{}).first->second;
        info->device_index = device_index;
        {
            char name[RSMI_NAME_BUFFER_SIZE] = {};
            auto result = m_rsmi.DeviceGetName(dv_ind, name, RSMI_NAME_BUFFER_SIZE);
            if (result == RSMI_STATUS_SUCCESS) {
                info->device_name = name;
            } else {
                info->device_name_error = m_rsmi.ErrorString(result);
            }
        }
        retrieve_and_assign_rsmi_threshold_temperatures(&m_rsmi, dv_ind, *info);
        retrieve_and_assign_rsmi_power_cap(&m_rsmi, dv_ind, *info);
        return info;
    }

    std::pair<std::optional<DeviceStatusInfo>, uint32_t> get_cached_device_info(
            unsigned int device_index) {
        std::lock_guard<std::mutex> lock(m_mutex);
        const uint32_t dv_ind = m_visible_device_indices[device_index];
        auto cached = m_static_device_info.find(dv_ind);
        if (cached != m_static_device_info.end()) {
            return {cached->second, dv_ind};
        }
        return {create_static_device_entry(device_index, dv_ind), dv_ind};
    }

public:
    static RsmiDeviceInfoCache &instance() {
        static RsmiDeviceInfoCache cache;
        return cache;
    }

    RsmiApi &rsmi() { return m_rsmi; }

    unsigned int get_device_count() const { return m_device_count; }

    bool is_device_accessible(unsigned int device_index) const {
        return m_rsmi.is_loaded() && device_index < m_device_count;
    }

    std::optional<DeviceStatusInfo> get_device_info(unsigned int device_index) {
        if (!m_rsmi.is_loaded()) {
            return std::nullopt;
        }
        auto [info, dv_ind] = get_cached_device_info(device_index);
        if (!info) {
            return std::nullopt;
        }
        // Fetch dynamic (live) metrics; these are not cached.
        retrieve_and_assign_rsmi_current_temperature(&m_rsmi, dv_ind, *info);
        retrieve_and_assign_rsmi_power_usage(&m_rsmi, dv_ind, *info);
        retrieve_and_assign_rsmi_utilization(&m_rsmi, dv_ind, *info);
        retrieve_and_assign_rsmi_perf_level(&m_rsmi, dv_ind, *info);
        // Note: throttle reason bitmask has no ROCm SMI equivalent; left as nullopt.
        return info;
    }
};

std::optional<std::string> read_version_from_rsmi() {
    auto &rsmi_api = RsmiDeviceInfoCache::instance().rsmi();
    if (!rsmi_api.is_loaded()) {
        return std::nullopt;
    }
    // Try rsmi_driver_version_str_get if available in this ROCm version.
    auto version = rsmi_api.GetDriverVersion();
    if (version) {
        return version;
    }
    // Fall back to reading the amdgpu kernel module version from sysfs.
    std::ifstream ver_file("/sys/module/amdgpu/version");
    if (ver_file.is_open()) {
        std::string line;
        if (std::getline(ver_file, line) && !line.empty()) {
            return line;
        }
    }
    spdlog::warn("Failed to retrieve AMD GPU driver version");
    return std::nullopt;
}

#endif  // HAS_RSMI

#if defined(__linux__) && !HAS_RSMI
std::optional<std::string> read_version_from_proc() {
    std::ifstream version_file("/proc/driver/nvidia/version",
                               std::ios_base::in | std::ios_base::binary);
    if (!version_file.is_open()) {
        spdlog::warn("No NVIDIA version file found in /proc");
        return std::nullopt;
    }

    // Parse the file line by line.
    std::string line;
    while (std::getline(version_file, line)) {
        auto info = detail::parse_nvidia_version_line(line);
        if (info.has_value()) {
            // We only expect there to be 1 version line, so we can return it immediately.
            return info;
        }
    }

    spdlog::warn("No version line found in /proc version file");
    return std::nullopt;
}
#endif  // __linux__ && !HAS_RSMI

}  // namespace

std::optional<DeviceStatusInfo> get_device_status_info(unsigned int device_index) {
#if HAS_NVML
    return DeviceInfoCache::instance().get_device_info(device_index);
#elif HAS_RSMI
    return RsmiDeviceInfoCache::instance().get_device_info(device_index);
#else
    (void)device_index;
    return std::nullopt;
#endif
}

std::vector<std::optional<DeviceStatusInfo>> get_devices_status_info() {
#if HAS_NVML || HAS_RSMI
    std::vector<std::optional<DeviceStatusInfo>> result{};
    const auto max_devices = get_device_count();
    for (unsigned int device_index{}; device_index < max_devices; ++device_index) {
        result.push_back(get_device_status_info(device_index));
    }
    return result;
#else
    return {};
#endif
}

std::optional<std::string> get_nvidia_driver_version() {
    static auto cached_version = [] {
        std::optional<std::string> version;
#if HAS_NVML
        version = read_version_from_nvml();
#elif HAS_RSMI
        version = read_version_from_rsmi();
#endif
#if defined(__linux__) && !HAS_RSMI
        if (!version) {
            version = read_version_from_proc();
        }
#endif  // __linux__ && !HAS_RSMI
        return version;
    }();
    return cached_version;
}

unsigned int get_device_count() {
#if HAS_NVML
    return DeviceInfoCache::instance().get_device_count();
#elif HAS_RSMI
    return RsmiDeviceInfoCache::instance().get_device_count();
#else
    return 0;
#endif  // HAS_NVML
}

namespace detail {

std::optional<std::string> parse_nvidia_version_line(std::string_view line) {
    // Format is undocumented, but appears to be 2/3 parts that are double-space separated.
    // NVRM version: <module type>  <version string>  [extra info]
    constexpr std::string_view separator{"  "};

    // Check line prefix.
    constexpr std::string_view prefix{"NVRM version: "};
    if (line.rfind(prefix, 0) != 0) {
        return std::nullopt;
    }

    // Find the splits.
    auto module_begin = prefix.size();
    auto module_end = line.find(separator, module_begin);
    if (module_end == line.npos) {
        return std::nullopt;
    }
    auto version_begin = module_end + separator.size();
    auto version_end = line.find(separator, version_begin);
    if (version_end == line.npos) {
        version_end = line.size();
    }

    // We have all the info we need.
    return std::string(line.substr(version_begin, version_end - version_begin));
}

bool is_accessible_device([[maybe_unused]] unsigned int device_index) {
#if HAS_NVML
    return DeviceInfoCache::instance().get_device_handle(device_index).has_value();
#elif HAS_RSMI
    return RsmiDeviceInfoCache::instance().is_device_accessible(device_index);
#else
    return false;
#endif  // HAS_NVML
}

}  // namespace detail

}  // namespace dorado::utils::gpu_monitor
